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
import hashlib
from contextlib import nullcontext
from copy import deepcopy
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import platform
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
from cosyvoice.utils.device import is_gpu_available
from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR


SUPPORTED_SDPA_BACKENDS = ('auto', 'flash', 'math')


def normalize_sdpa_backend(sdpa_backend):
    if sdpa_backend is None:
        return 'auto'
    if not isinstance(sdpa_backend, str):
        raise ValueError('sdpa_backend must be one of {}, got {!r}'.format(
            SUPPORTED_SDPA_BACKENDS, sdpa_backend))
    sdpa_backend = sdpa_backend.lower().strip()
    if sdpa_backend not in SUPPORTED_SDPA_BACKENDS:
        raise ValueError('unsupported sdpa_backend {!r}; expected one of {}'.format(
            sdpa_backend, SUPPORTED_SDPA_BACKENDS))
    return sdpa_backend


def sdpa_kernel_context(sdpa_backend):
    sdpa_backend = normalize_sdpa_backend(sdpa_backend)
    if sdpa_backend == 'auto':
        return nullcontext()

    if hasattr(torch.nn, 'attention') and hasattr(torch.nn.attention, 'sdpa_kernel') and \
       hasattr(torch.nn.attention, 'SDPBackend'):
        sdp_backend = torch.nn.attention.SDPBackend
        selected_backend = {
            'flash': sdp_backend.FLASH_ATTENTION,
            'math': sdp_backend.MATH,
        }[sdpa_backend]
        return torch.nn.attention.sdpa_kernel(selected_backend)

    if hasattr(torch.backends.cuda, 'sdp_kernel'):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=sdpa_backend == 'flash',
            enable_math=sdpa_backend == 'math',
            enable_mem_efficient=False,
            enable_cudnn=False,
        )

    raise RuntimeError('This PyTorch build does not support selecting the SDPA backend.')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def set_global_random_seed(seed, deterministic=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if hasattr(torch, 'musa') and torch.musa.is_available():
            torch.musa.manual_seed_all(seed)
        elif torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception as ex:
        logging.warning('Failed to seed accelerator RNG: %s', ex)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
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


def seed_dataloader_for_epoch(data_loader, epoch, data_seed):
    rank = int(os.environ.get('RANK', 0))
    epoch_seed = data_seed + epoch * 100000 + rank
    if data_loader.generator is not None:
        data_loader.generator.manual_seed(epoch_seed)
    if data_loader.num_workers == 0:
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % (2 ** 32))


def check_modify_and_save_config(args, configs):
    configs['train_conf']['sdpa_backend'] = normalize_sdpa_backend(
        configs['train_conf'].get('sdpa_backend', 'auto'))
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
        assert is_gpu_available(), 'torch_ddp training requires a CUDA or MUSA accelerator'
        model.cuda()
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
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    try:
        if state.get('accelerator_type') == 'musa':
            torch.musa.set_rng_state(state['accelerator'])
        elif state.get('accelerator_type') == 'cuda':
            torch.cuda.set_rng_state(state['accelerator'])
        elif state.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state['cuda'])
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


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _onnxruntime_version():
    for package in ['onnxruntime', 'onnxruntime-gpu', 'onnxruntime-musa']:
        package_version = _package_version(package)
        if package_version is not None:
            return package_version
    return None


def _onnxruntime_providers():
    try:
        return import_module('onnxruntime').get_available_providers()
    except Exception as ex:
        logging.debug('Failed to query ONNX Runtime providers: %s', ex)
        return None


def _device_name(device_type):
    try:
        if device_type == 'musa':
            return torch.musa.get_device_name(int(os.environ.get('LOCAL_RANK', 0)))
        if device_type == 'cuda':
            return torch.cuda.get_device_name(int(os.environ.get('LOCAL_RANK', 0)))
    except Exception as ex:
        logging.debug('Failed to query %s device name: %s', device_type, ex)
    return None


def _accelerator_signature():
    requested_accelerator = os.environ.get('ACCELERATOR_BACKEND', '').lower()
    musa_active = requested_accelerator == 'musa'
    cuda_active = requested_accelerator == 'cuda'
    if not musa_active and not cuda_active:
        musa_active = hasattr(torch, 'musa') and torch.musa.is_available()
        cuda_active = not musa_active and torch.cuda.is_available()

    if musa_active:
        try:
            mccl_version = torch.musa.mccl.version()
        except Exception:
            mccl_version = None
        return {
            'accelerator': 'musa',
            'accelerator_runtime_version': getattr(torch.version, 'musa', None),
            'accelerator_device_name': _device_name('musa'),
            'musa_version': getattr(torch.version, 'musa', None),
            'torch_musa_version': _package_version('torch_musa'),
            'torchada_version': _package_version('torchada'),
            'mccl_version': mccl_version,
            'cuda_version': None,
            'cudnn_version': None,
            'nccl_version': None,
        }
    if cuda_active:
        try:
            nccl_version = torch.cuda.nccl.version()
        except Exception:
            nccl_version = None
        return {
            'accelerator': 'cuda',
            'accelerator_runtime_version': torch.version.cuda,
            'accelerator_device_name': _device_name('cuda'),
            'musa_version': None,
            'torch_musa_version': None,
            'torchada_version': _package_version('torchada'),
            'mccl_version': None,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'nccl_version': nccl_version,
        }
    signature = {
        'accelerator': 'cpu',
        'accelerator_runtime_version': None,
        'accelerator_device_name': platform.processor() or None,
        'musa_version': None,
        'torch_musa_version': _package_version('torch_musa'),
        'torchada_version': _package_version('torchada'),
        'mccl_version': None,
        'cuda_version': None,
        'cudnn_version': None,
        'nccl_version': None,
    }
    return signature


def resume_signature(info_dict):
    signature = {
        key: info_dict.get(key)
        for key in [
            'model', 'train_engine', 'seed', 'data_seed', 'deterministic',
            'num_workers', 'prefetch', 'accum_grad', 'use_amp', 'dtype',
            'sdpa_backend', 'save_per_step', 'grad_clip', 'optim', 'optim_conf',
            'scheduler', 'scheduler_conf', 'qwen_pretrain_path', 'onnx_path',
            'dist_backend',
        ]
    }
    signature['signature_version'] = 2
    signature['python_version'] = platform.python_version()
    signature['platform'] = platform.platform()
    signature['torch_version'] = str(torch.__version__)
    signature['onnxruntime_version'] = _onnxruntime_version()
    signature['onnxruntime_providers'] = _onnxruntime_providers()
    signature['onnx_provider_request'] = os.environ.get('COSYVOICE_ONNX_PROVIDER', 'auto')
    signature['world_size'] = int(os.environ.get('WORLD_SIZE', 1))
    accelerator_signature = _accelerator_signature()
    signature['visible_devices'] = (
        os.environ.get('MUSA_VISIBLE_DEVICES')
        if accelerator_signature['accelerator'] == 'musa'
        else os.environ.get('CUDA_VISIBLE_DEVICES')
    )
    signature.update(accelerator_signature)
    for key in ['config', 'train_data', 'cv_data']:
        path = info_dict.get(key)
        if path:
            path = os.path.abspath(path)
            signature[key] = path
            if os.path.isfile(path):
                signature['{}_sha256'.format(key)] = _file_sha256(path)
    return signature


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
        if re.search(r'epoch_\d+_(?:whole|step_\d+)\.pt$', os.path.basename(path))
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


def _validate_resume_signature(saved_signature, current_info):
    if not saved_signature:
        logging.warning('Checkpoint has no resume signature; exact data/config alignment cannot be verified.')
        return
    current_signature = resume_signature(current_info)
    _log_resume_signatures(saved_signature, current_signature)
    mismatches = []
    for key, saved_value in saved_signature.items():
        current_value = current_signature.get(key)
        if current_value != saved_value:
            mismatches.append('{}: saved={!r}, current={!r}'.format(key, saved_value, current_value))
    if mismatches:
        raise RuntimeError(
            'Resume configuration does not match the checkpoint:\n  {}'
            .format('\n  '.join(mismatches)))


def _log_resume_signatures(saved_signature, current_signature):
    if int(os.environ.get('RANK', 0)) != 0:
        return
    logging.info(
        'Checkpoint resume signature:\n%s',
        yaml.safe_dump(saved_signature, sort_keys=True).rstrip())
    logging.info(
        'Current resume signature:\n%s',
        yaml.safe_dump(current_signature, sort_keys=True).rstrip())


def _validate_cross_platform_signature(saved_signature, current_info):
    if not saved_signature:
        logging.warning('Checkpoint has no resume signature; cross-platform compatibility cannot be verified.')
        return
    current_signature = resume_signature(current_info)
    _log_resume_signatures(saved_signature, current_signature)
    required_keys = {
        'model', 'train_engine', 'accum_grad', 'dtype', 'optim', 'optim_conf',
        'scheduler', 'scheduler_conf', 'train_data_sha256', 'cv_data_sha256',
    }
    incompatible = []
    ignored = []
    missing = '<missing>'
    for key in sorted(set(saved_signature) | set(current_signature)):
        saved_value = saved_signature.get(key, missing)
        current_value = current_signature.get(key, missing)
        if current_value == saved_value:
            continue
        mismatch = '{}: saved={!r}, current={!r}'.format(key, saved_value, current_value)
        if key in required_keys:
            incompatible.append(mismatch)
        else:
            ignored.append(mismatch)
    if incompatible:
        raise RuntimeError(
            'Cross-platform resume has incompatible training state:\n  {}'
            .format('\n  '.join(incompatible)))
    if ignored and int(os.environ.get('RANK', 0)) == 0:
        logging.warning(
            'Cross-platform resume ignores platform/path differences:\n  %s',
            '\n  '.join(ignored))


def _move_state_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_state_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state_to_device(item, device) for item in value)
    return value


def _load_optimizer_state_cross_platform(optimizer, state_dict, name):
    if optimizer is None or state_dict is None:
        return False
    initial_state = deepcopy(optimizer.state_dict())
    try:
        optimizer.load_state_dict(state_dict)
        for parameter, state in list(optimizer.state.items()):
            optimizer.state[parameter] = _move_state_to_device(state, parameter.device)
        logging.info('Restored %s state on the current accelerator', name)
        return True
    except Exception as ex:
        optimizer.load_state_dict(initial_state)
        logging.warning('Failed to restore %s state; using a fresh %s: %s', name, name, ex)
        return False


def load_cross_platform_training_state(checkpoint_path, optimizer, scheduler, current_info,
                                       optimizer_d=None, scheduler_d=None, scaler=None):
    state_path = training_state_path(checkpoint_path)
    if not os.path.isfile(state_path):
        logging.warning(
            'Training state %s is missing; only model weights, epoch, and step will be restored.',
            state_path)
        return {}, None

    state = torch.load(state_path, map_location='cpu', weights_only=False)
    _validate_cross_platform_signature(state.get('resume_signature'), current_info)
    _load_optimizer_state_cross_platform(optimizer, state.get('optimizer'), 'optimizer')
    _load_optimizer_state_cross_platform(optimizer_d, state.get('optimizer_d'), 'discriminator optimizer')

    try:
        scheduler.load_state_dict(state['scheduler'])
        logging.info('Restored scheduler state')
    except Exception as ex:
        set_scheduler_step(scheduler, state.get('step', 0))
        logging.warning('Failed to restore scheduler state; restored step only: %s', ex)
    if scheduler_d is not None and state.get('scheduler_d') is not None:
        try:
            scheduler_d.load_state_dict(state['scheduler_d'])
            logging.info('Restored discriminator scheduler state')
        except Exception as ex:
            set_scheduler_step(scheduler_d, state.get('step', 0))
            logging.warning('Failed to restore discriminator scheduler state; restored step only: %s', ex)
    if scaler is not None and state.get('scaler') is not None:
        try:
            scaler.load_state_dict(state['scaler'])
            logging.info('Restored AMP GradScaler state')
        except Exception as ex:
            logging.warning('Failed to restore AMP GradScaler state; using a fresh scaler: %s', ex)

    rank = int(os.environ.get('RANK', 0))
    rng_path = re.sub(r'\.pt$', '.rank{}.rng.pt'.format(rank), checkpoint_path)
    rng_state = None
    if os.path.isfile(rng_path):
        saved_rng = torch.load(rng_path, map_location='cpu', weights_only=False)
        rng_state = {
            key: saved_rng[key]
            for key in ['python', 'numpy', 'torch']
            if key in saved_rng
        }
        logging.warning(
            'Restored portable RNG state from %s; accelerator RNG is reset for the current platform.',
            rng_path)
    else:
        logging.warning('RNG state %s is missing; all RNG streams use their new-run state.', rng_path)
    return state, rng_state


def load_training_state(checkpoint_path, optimizer, scheduler, current_info,
                        optimizer_d=None, scheduler_d=None, scaler=None):
    state_path = training_state_path(checkpoint_path)
    if not os.path.isfile(state_path):
        logging.warning(
            'Training state %s is missing; resuming weights, epoch, and step only. '
            'Optimizer momentum, AMP scaler, and exact RNG continuity will be reset.',
            state_path)
        return {}, None

    state = torch.load(state_path, map_location='cpu', weights_only=False)
    _validate_resume_signature(state.get('resume_signature'), current_info)
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
    if not os.path.isfile(rng_path):
        raise RuntimeError('RNG state {} is missing; exact resume is impossible.'.format(rng_path))
    rng_state = torch.load(rng_path, map_location='cpu', weights_only=False)
    missing_rng = [key for key in ['python', 'numpy', 'torch'] if key not in rng_state]
    if is_gpu_available() and 'accelerator' not in rng_state and 'cuda' not in rng_state:
        missing_rng.append('accelerator')
    if missing_rng:
        if state.get('format_version', 0) >= 2:
            raise RuntimeError(
                'RNG state {} is missing {}; exact resume is impossible.'
                .format(rng_path, ', '.join(missing_rng)))
        logging.warning(
            'RNG state %s is missing %s; resuming legacy checkpoint without bit-exact RNG continuity.',
            rng_path, ', '.join(missing_rng))
    return state, rng_state


def save_model(model, model_name, info_dict, optimizer=None, scheduler=None,
               optimizer_d=None, scheduler_d=None, scaler=None):
    rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))

    if info_dict["train_engine"] == "torch_ddp":
        if rank == 0:
            _atomic_torch_save(
                {
                    **model.module.state_dict(),
                    'epoch': info_dict['epoch'],
                    'step': info_dict['step'],
                },
                save_model_path)
            if info_dict.get('save_states') == 'model+optimizer' and optimizer is not None:
                state = {
                    'format_version': 2,
                    'model_checkpoint': os.path.basename(save_model_path),
                    'epoch': info_dict['epoch'],
                    'step': info_dict['step'],
                    'train_batch_idx': info_dict.get('train_batch_idx', -1),
                    'epoch_complete': info_dict.get('epoch_complete', False),
                    'world_size': dist.get_world_size(),
                    'resume_signature': resume_signature(info_dict),
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
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        _atomic_write_text(yaml.dump(info_dict), info_path)
        if info_dict["train_engine"] == "torch_ddp" and model_name != 'init':
            _atomic_write_text(
                os.path.basename(save_model_path),
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

    with sdpa_kernel_context(info_dict.get('sdpa_backend', 'auto')), autocast:
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
            epoch, step, lr,
            ' '.join(['{} {}'.format(k, v) for k, v in loss_dict.items()]),
            rank))

    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step)
        for k, v in loss_dict.items():
            writer.add_scalar('{}/{}'.format(tag, k), v, step)
