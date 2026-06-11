# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import musa_patch
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.losses import DPOLoss
from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, load_training_state, resolve_ddp_resume_checkpoint,
    save_model, set_scheduler_step,
    wrap_cuda_model, check_modify_and_save_config)


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--ref_model', required=False, help='ref model used in dpo')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--onnx_path', required=False, help='onnx path, which is required for online feature extraction')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--resume',
                        nargs='?',
                        const='auto',
                        help='resume full training state from PATH; auto uses latest or starts fresh if none exists')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo', 'mccl'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--data_seed',
                        default=1986,
                        type=int,
                        help='per-epoch data pipeline seed used for resumable iteration')
    parser.add_argument('--save_per_step',
                        type=int,
                        help='override train_conf.save_per_step; <= 0 disables intra-epoch checkpoints')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model only or full optimizer state; full state is required for exact DDP resume')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) for distributed control collectives.')
    parser.add_argument('--early_stop_file',
                        help='stop training at the next epoch boundary when this file exists')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    os.environ['onnx_path'] = args.onnx_path
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('RAYON_NUM_THREADS', '1')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # gan train has some special initialization logic
    gan = True if args.model == 'hifigan' else False

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    if args.qwen_pretrain_path is not None:
        override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    train_args = vars(args).copy()
    save_per_step = train_args.pop('save_per_step')
    configs['train_conf'].update(train_args)
    if save_per_step is not None:
        configs['train_conf']['save_per_step'] = save_per_step

    # Init env for ddp
    init_distributed(args)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, args.dpo)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint
    if args.dpo is True:
        configs[args.model].forward = configs[args.model].forward_dpo
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    resume_checkpoint = None
    if args.resume is not None and args.train_engine == 'torch_ddp':
        try:
            resume_checkpoint = resolve_ddp_resume_checkpoint(args.resume, args.model_dir)
            logging.info('Resuming DDP training from %s', resume_checkpoint)
        except FileNotFoundError:
            if args.resume != 'auto':
                raise
            logging.info('No DDP checkpoint found in %s; starting a new run', args.model_dir)
            args.resume = None
            configs['train_conf']['resume'] = None
    elif args.resume == 'auto' and not os.path.isfile(os.path.join(args.model_dir, 'latest')):
        logging.info('No DeepSpeed checkpoint found in %s; starting a new run', args.model_dir)
        args.resume = None
        configs['train_conf']['resume'] = None
    model_checkpoint = resume_checkpoint
    if model_checkpoint is None and args.resume is None:
        model_checkpoint = args.checkpoint
    elif args.resume is not None and args.checkpoint is not None:
        logging.info('Ignoring --checkpoint %s because --resume was requested', args.checkpoint)
    if model_checkpoint is not None:
        if os.path.exists(model_checkpoint):
            state_dict = torch.load(model_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            if 'step' in state_dict:
                start_step = state_dict['step']
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch']
        else:
            logging.warning('checkpoint {} does not exist!'.format(model_checkpoint))

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    resume_state, resume_rng_state = {}, None
    if resume_checkpoint is not None:
        resume_state, resume_rng_state = load_training_state(
            resume_checkpoint, optimizer, scheduler, optimizer_d, scheduler_d, scaler)
        start_step = resume_state.get('step', start_step)
        start_epoch = resume_state.get('epoch', start_epoch)
        if not resume_state:
            set_scheduler_step(scheduler, start_step)
            if scheduler_d is not None:
                set_scheduler_step(scheduler_d, start_step)
    elif args.resume is not None:
        if args.resume == 'auto':
            load_dir, tag = args.model_dir, None
        elif os.path.isdir(args.resume) and os.path.isfile(os.path.join(args.resume, 'latest')):
            load_dir, tag = args.resume, None
        else:
            resume_path = os.path.abspath(args.resume)
            load_dir, tag = os.path.dirname(resume_path), os.path.basename(resume_path)
        load_path, client_state = model.load_checkpoint(
            load_dir, tag=tag, load_module_strict=False,
            load_optimizer_states=True, load_lr_scheduler_states=True)
        if load_path is None:
            raise FileNotFoundError('No DeepSpeed checkpoint found in {} tag {}'.format(load_dir, tag))
        resume_state = client_state or {}
        start_step = resume_state.get('step', 0)
        start_epoch = resume_state.get('epoch', -1)
        rank = int(os.environ.get('RANK', 0))
        rng_path = os.path.join(os.path.dirname(load_path), 'rank{}.rng.pt'.format(rank))
        if os.path.isfile(rng_path):
            resume_rng_state = torch.load(rng_path, map_location='cpu', weights_only=False)
        logging.info('Resuming DeepSpeed training from %s', load_path)
    else:
        set_scheduler_step(scheduler, start_step)
        if scheduler_d is not None:
            set_scheduler_step(scheduler_d, start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    info_dict['epoch_complete'] = True
    if args.resume is None:
        save_model(model, 'init', info_dict, optimizer, scheduler,
                   optimizer_d=optimizer_d, scheduler_d=scheduler_d, scaler=scaler)

    # DPO related
    if args.dpo is True:
        ref_model = deepcopy(configs[args.model])
        state_dict = torch.load(args.ref_model, map_location='cpu')
        ref_model.load_state_dict(state_dict, strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        # NOTE maybe it is not needed to wrap ref_model as ddp because its parameter is not updated
        ref_model = wrap_cuda_model(args, ref_model)
    else:
        ref_model, dpo_loss = None, None

    # Get executor
    executor = Executor(gan=gan, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.step = start_step

    print('start step {} start epoch {}'.format(start_step, start_epoch))

    # Keep one healthy Gloo group for lightweight host-side control collectives.
    control_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
    epoch_complete = resume_state.get(
        'epoch_complete',
        resume_checkpoint is None or os.path.basename(resume_checkpoint).endswith('_whole.pt'))
    if resume_checkpoint is not None and not resume_state and not epoch_complete:
        raise RuntimeError(
            'Legacy intra-epoch checkpoint {} has no .train.pt state and cannot be resumed safely. '
            'Use an epoch_*_whole.pt checkpoint instead.'.format(resume_checkpoint))
    if not epoch_complete and resume_state.get('world_size', dist.get_world_size()) != dist.get_world_size():
        raise RuntimeError(
            'Intra-epoch resume requires the original world size {}, but current world size is {}.'
            .format(resume_state['world_size'], dist.get_world_size()))
    first_epoch = start_epoch + 1 if epoch_complete else start_epoch
    resume_batch_idx = 0 if epoch_complete else resume_state.get('train_batch_idx', -1) + 1
    try:
        for epoch in range(first_epoch, info_dict['max_epoch']):
            executor.epoch = epoch
            train_dataset.set_epoch(epoch)
            dist.barrier()
            if gan is True:
                executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                            writer, info_dict, scaler, control_group,
                                            resume_batch_idx=resume_batch_idx,
                                            resume_rng_state=resume_rng_state)
            else:
                executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler,
                                        control_group, ref_model=ref_model,
                                        resume_batch_idx=resume_batch_idx,
                                        resume_rng_state=resume_rng_state)
            resume_batch_idx = 0
            resume_rng_state = None
            stop_flag = torch.tensor(
                int(dist.get_rank() == 0 and args.early_stop_file is not None and
                    os.path.exists(args.early_stop_file)),
                dtype=torch.int32,
            )
            dist.broadcast(stop_flag, src=0, group=control_group)
            if stop_flag.item():
                logging.info('Early stop requested after epoch %s by %s', epoch, args.early_stop_file)
                break
    finally:
        dist.destroy_process_group(control_group)


if __name__ == '__main__':
    main()
