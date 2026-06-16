# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import glob
import random
import torch
import json
import re
import datetime
import warnings
import yaml
import numpy as np

import deepspeed
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

from cosyvoice.dataset.dataset import Dataset
from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def set_global_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if hasattr(torch, 'musa') and torch.musa.is_available():
        torch.musa.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine == 'torch_ddp':
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend)
    else:
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    return world_size, local_rank, rank


def init_dataset_and_dataloader(args, configs, gan, dpo):
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']
    train_dataset = Dataset(args.train_data, data_pipeline=data_pipeline, mode='train', gan=gan, dpo=dpo, shuffle=True, partition=True)
    cv_dataset = Dataset(args.cv_data, data_pipeline=data_pipeline, mode='dev', gan=gan, dpo=dpo, shuffle=False, partition=False)

    rank = int(os.environ.get('RANK', 0))
    train_generator = torch.Generator()
    train_generator.manual_seed(args.data_seed + rank)
    cv_generator = torch.Generator()
    cv_generator.manual_seed(args.data_seed + 10000 + rank)
    loader_kwargs = {
        'batch_size': None,
        'pin_memory': args.pin_memory,
        'num_workers': args.num_workers,
        'worker_init_fn': seed_worker,
    }
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(train_dataset,
                                   generator=train_generator,
                                   **loader_kwargs)
    cv_data_loader = DataLoader(cv_dataset,
                                generator=cv_generator,
                                **loader_kwargs)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def check_modify_and_save_config(args, configs):
    if args.train_engine == "torch_ddp":
        configs['train_conf']["dtype"] = 'bf16' if args.use_amp is True else 'fp32'
    else:
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs['train_conf']["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs['train_conf']["dtype"] = "bf16"
        else:
            configs['train_conf']["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        # if use deepspeed, override ddp config
        configs['train_conf']['save_per_step'] = int(configs['train_conf']['save_per_step'] *
                                                     configs['train_conf']['accum_grad'] / ds_configs["gradient_accumulation_steps"])
        configs['train_conf']['accum_grad'] = ds_configs["gradient_accumulation_steps"]
        configs['train_conf']['grad_clip'] = ds_configs["gradient_clipping"]
        configs['train_conf']['log_interval'] = ds_configs["steps_per_print"]
    return configs


def wrap_cuda_model(args, model):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.musa()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
    return model


def init_optimizer_and_scheduler(args, configs, model, gan):
    if gan is False:
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        # use deepspeed optimizer for speedup
        if args.train_engine == "deepspeed":
            def scheduler(opt):
                return scheduler_type(opt, **configs['train_conf']['scheduler_conf'])
            model, optimizer, _, scheduler = deepspeed.initialize(
                args=args,
                model=model,
                optimizer=None,
                lr_scheduler=scheduler,
                model_parameters=model.parameters())

        optimizer_d, scheduler_d = None, None

    else:
        # currently we wrap generator and discriminator in one model, so we cannot use deepspeed
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        if configs['train_conf']['optim_d'] == 'adam':
            optimizer_d = optim.Adam(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf_d'])
        elif configs['train_conf']['optim_d'] == 'adamw':
            optimizer_d = optim.AdamW(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf_d'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler_d'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler_d = WarmupLR(optimizer_d, **configs['train_conf']['scheduler_d'])
        elif configs['train_conf']['scheduler_d'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler_d = NoamHoldAnnealing(optimizer_d, **configs['train_conf']['scheduler_d'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler_d = ConstantLR(optimizer_d)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])
    return model, optimizer, scheduler, optimizer_d, scheduler_d


def init_summarywriter(args):
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    return writer


def set_scheduler_step(scheduler, step):
    scheduler.set_step(step)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lrs = scheduler.get_lr()
    for param_group, lr in zip(scheduler.optimizer.param_groups, lrs):
        param_group['lr'] = lr
    scheduler._last_lr = lrs


def seed_dataloader_for_epoch(data_loader, epoch, data_seed):
    rank = int(os.environ.get('RANK', 0))
    epoch_seed = data_seed + epoch * 100000 + rank
    if data_loader.generator is not None:
        data_loader.generator.manual_seed(epoch_seed)
    if data_loader.num_workers == 0:
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % (2 ** 32))


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    try:
        if hasattr(torch, 'musa') and torch.musa.is_available():
            state['accelerator'] = torch.musa.get_rng_state()
            state['accelerator_type'] = 'musa'
        elif torch.cuda.is_available():
            state['accelerator'] = torch.cuda.get_rng_state()
            state['accelerator_type'] = 'cuda'
    except Exception as ex:
        logging.warning('Failed to capture accelerator RNG state: %s', ex)
    return state


def restore_rng_state(state):
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    else:
        logging.warning('Python RNG state is missing; resume will not be bit-exact.')
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    else:
        logging.warning('NumPy RNG state is missing; resume will not be bit-exact.')
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    try:
        if state.get('accelerator_type') == 'musa':
            torch.musa.set_rng_state(state['accelerator'])
        elif state.get('accelerator_type') == 'cuda':
            torch.cuda.set_rng_state(state['accelerator'])
    except Exception as ex:
        logging.warning('Failed to restore accelerator RNG state: %s', ex)


def _atomic_torch_save(obj, path):
    tmp_path = '{}.tmp.{}'.format(path, os.getpid())
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_write_text(text, path):
    tmp_path = '{}.tmp.{}'.format(path, os.getpid())
    try:
        with open(tmp_path, 'w') as fout:
            fout.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def training_state_path(checkpoint_path):
    return re.sub(r'\.pt$', '.train.pt', checkpoint_path)


def find_latest_ddp_checkpoint(model_dir):
    latest_path = os.path.join(model_dir, 'latest_ddp')
    if os.path.isfile(latest_path):
        with open(latest_path, 'r') as fin:
            checkpoint_name = fin.read().strip()
        checkpoint_path = os.path.join(model_dir, checkpoint_name)
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
        logging.warning('Ignoring stale latest_ddp entry %s', checkpoint_name)

    candidates = [
        path for path in glob.glob(os.path.join(model_dir, 'epoch_*.pt'))
        if not path.endswith('.train.pt')
    ]
    if not candidates:
        raise FileNotFoundError('No resumable DDP checkpoint found in {}'.format(model_dir))
    return max(candidates, key=os.path.getmtime)


def resolve_ddp_resume_checkpoint(resume, model_dir):
    if resume == 'auto':
        return find_latest_ddp_checkpoint(model_dir)
    if os.path.isdir(resume):
        return find_latest_ddp_checkpoint(resume)
    checkpoint_path = os.path.abspath(resume)
    if checkpoint_path.endswith('.train.pt'):
        checkpoint_path = checkpoint_path[:-len('.train.pt')] + '.pt'
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('Resume checkpoint not found: {}'.format(checkpoint_path))
    return checkpoint_path


def load_training_state(checkpoint_path, optimizer, scheduler,
                        optimizer_d=None, scheduler_d=None, scaler=None):
    state_path = training_state_path(checkpoint_path)
    if not os.path.isfile(state_path):
        logging.warning(
            'Training state %s is missing; resuming weights, epoch, and step only. '
            'Optimizer momentum and AMP scaler will be reset.', state_path)
        return {}, None

    state = torch.load(state_path, map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    if optimizer_d is not None and state.get('optimizer_d') is not None:
        optimizer_d.load_state_dict(state['optimizer_d'])
    if scheduler_d is not None and state.get('scheduler_d') is not None:
        scheduler_d.load_state_dict(state['scheduler_d'])
    if scaler is not None and state.get('scaler') is not None:
        scaler.load_state_dict(state['scaler'])

    rank = int(os.environ.get('RANK', 0))
    rng_path = re.sub(r'\.pt$', '.rank{}.rng.pt'.format(rank), checkpoint_path)
    rng_state = None
    if os.path.isfile(rng_path):
        rng_state = torch.load(rng_path, map_location='cpu', weights_only=False)
    else:
        logging.warning('RNG state %s is missing; resume will not be bit-exact.', rng_path)
    return state, rng_state


def save_model(model, model_name, info_dict, optimizer=None, scheduler=None,
               optimizer_d=None, scheduler_d=None, scaler=None):
    rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))

    if info_dict["train_engine"] == "torch_ddp":
        if rank == 0:
            _atomic_torch_save(
                {**model.module.state_dict(),
                 'epoch': info_dict['epoch'],
                 'step': info_dict['step']},
                save_model_path)
            if info_dict.get('save_states') == 'model+optimizer' and optimizer is not None:
                state = {
                    'format_version': 1,
                    'model_checkpoint': os.path.basename(save_model_path),
                    'epoch': info_dict['epoch'],
                    'step': info_dict['step'],
                    'train_batch_idx': info_dict.get('train_batch_idx', -1),
                    'epoch_complete': info_dict.get('epoch_complete', False),
                    'world_size': dist.get_world_size(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer_d': optimizer_d.state_dict() if optimizer_d is not None else None,
                    'scheduler_d': scheduler_d.state_dict() if scheduler_d is not None else None,
                    'scaler': scaler.state_dict() if scaler is not None else None,
                }
                _atomic_torch_save(state, training_state_path(save_model_path))
        if info_dict.get('save_states') == 'model+optimizer' and optimizer is not None:
            rng_path = re.sub(r'\.pt$', '.rank{}.rng.pt'.format(rank), save_model_path)
            _atomic_torch_save(capture_rng_state(), rng_path)
    else:
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir,
                                  tag=model_name,
                                  client_state=info_dict)
        rng_path = os.path.join(model_dir, model_name, 'rank{}.rng.pt'.format(rank))
        _atomic_torch_save(capture_rng_state(), rng_path)
    if rank == 0:
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        _atomic_write_text(yaml.dump(info_dict), info_path)
        if info_dict["train_engine"] == "torch_ddp" and model_name != 'init':
            _atomic_write_text(os.path.basename(save_model_path),
                               os.path.join(model_dir, 'latest_ddp'))
        logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(rank, save_model_path))


def distributed_batch_available(has_batch, control_group):
    """Stop all ranks at the shortest input without poisoning the process group."""
    if dist.get_world_size() == 1:
        return has_batch

    available = torch.tensor(int(has_batch), dtype=torch.int32)
    dist.all_reduce(available, op=dist.ReduceOp.MIN, group=control_group)
    return bool(available.item())


def batch_forward(model, batch, scaler, info_dict, ref_model=None, dpo_loss=None):
    device = int(os.environ.get('LOCAL_RANK', 0))

    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32

    if info_dict['train_engine'] == 'torch_ddp':
        autocast = torch.cuda.amp.autocast(enabled=scaler is not None, dtype=dtype)
    else:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)

    with autocast:
        info_dict['loss_dict'] = model(batch, device)
        if ref_model is not None and dpo_loss is not None:
            chosen_logps = info_dict['loss_dict']["chosen_logps"]
            rejected_logps = info_dict['loss_dict']["rejected_logps"]
            sft_loss = info_dict['loss_dict']['loss']
            with torch.no_grad():
                ref_loss_dict = ref_model(batch, device)
            reference_chosen_logps = ref_loss_dict["chosen_logps"]
            reference_rejected_logps = ref_loss_dict["rejected_logps"]
            preference_loss, chosen_reward, reject_reward = dpo_loss(
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            dpo_acc = (chosen_reward > reject_reward).float().mean()
            info_dict['loss_dict']["loss"] = preference_loss + sft_loss
            info_dict['loss_dict']["sft_loss"] = sft_loss
            info_dict['loss_dict']["dpo_loss"] = preference_loss
            info_dict['loss_dict']["dpo_acc"] = dpo_acc
            info_dict['loss_dict']["chosen_reward"] = chosen_reward.mean()
            info_dict['loss_dict']["reject_reward"] = reject_reward.mean()
    return info_dict


def batch_backward(model, scaler, info_dict):
    if info_dict["train_engine"] == "deepspeed":
        scaled_loss = model.backward(info_dict['loss_dict']['loss'])
    else:
        scaled_loss = info_dict['loss_dict']['loss'] / info_dict['accum_grad']
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    info_dict['loss_dict']['loss'] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict):
    grad_norm = 0.0
    if info_dict['train_engine'] == "deepspeed":
        info_dict["is_gradient_accumulation_boundary"] = model.is_gradient_accumulation_boundary()
        model.step()
        grad_norm = model.get_global_grad_norm()
    elif (info_dict['batch_idx'] + 1) % info_dict["accum_grad"] == 0:
        # Use mixed precision training
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            # We don't check grad here since that if the gradient
            # has inf/nan values, scaler.step will skip
            # optimizer.step().
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                logging.warning('get infinite grad_norm, check your code/data if it appears frequently')
            scaler.update()
        else:
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                logging.warning('get infinite grad_norm, check your code/data if it appears frequently')
        optimizer.zero_grad()
        scheduler.step()
    info_dict["lr"] = optimizer.param_groups[0]['lr']
    info_dict["grad_norm"] = grad_norm
    return info_dict


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict.get('epoch', 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    rank = int(os.environ.get('RANK', 0))

    # only rank 0 write to tensorboard to avoid multi-process write
    if writer is not None:
        if (info_dict['train_engine'] == 'deepspeed' and info_dict['is_gradient_accumulation_boundary'] is True) or \
           (info_dict['train_engine'] == 'torch_ddp' and (info_dict['batch_idx'] + 1) % info_dict['accum_grad'] == 0):
            for k in ['epoch', 'lr', 'grad_norm']:
                writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
            for k, v in loss_dict.items():
                writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)

    # TRAIN & CV, Shell log (stdout)
    if (info_dict['batch_idx'] + 1) % info_dict['log_interval'] == 0:
        completed_step = step
        if tag == 'TRAIN':
            if info_dict['train_engine'] == 'deepspeed':
                completed_step += int(info_dict.get('is_gradient_accumulation_boundary', False))
            else:
                completed_step += int((batch_idx + 1) % info_dict['accum_grad'] == 0)
        log_str = '{} Epoch {} Batch {} Step {} '.format(
            tag, epoch, batch_idx + 1, completed_step)
        for name, value in loss_dict.items():
            log_str += '{} {:.6f} '.format(name, value)
        if tag == "TRAIN":
            log_str += 'lr {:.8f} grad_norm {:.6f}'.format(
                info_dict["lr"], info_dict['grad_norm'])
        log_str += ' rank {}'.format(rank)
        logging.debug(log_str)


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict['lr']
    rank = int(os.environ.get('RANK', 0))
    logging.info(
        'Epoch {} Step {} CV info lr {} {} rank {}'.format(
            epoch, step, lr, ' '.join(['{} {}'.format(k, v) for k, v in loss_dict.items()]), rank))

    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step)
        for k, v in loss_dict.items():
            writer.add_scalar('{}/{}'.format(tag, k), v, step)
