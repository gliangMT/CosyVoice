# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
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

import logging
from contextlib import nullcontext
import os
import time

import torch
import torch.distributed as dist

from cosyvoice.utils.train_utils import (
    batch_backward,
    batch_forward,
    distributed_batch_available,
    distributed_rng_alignment_limit_exceeded,
    log_per_save,
    log_per_step,
    restore_rng_state,
    save_model,
    seed_dataloader_for_epoch,
    update_parameter_and_lr,
)


class Executor:

    def __init__(self, gan: bool = False, ref_model: torch.nn.Module = None, dpo_loss: torch.nn.Module = None):
        self.gan = gan
        self.ref_model = ref_model
        self.dpo_loss = dpo_loss
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.rank))

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler,
                       control_group, ref_model=None, resume_batch_idx=0, resume_rng_state=None):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
        seed_dataloader_for_epoch(train_data_loader, self.epoch, info_dict['data_seed'])
        data_iter = iter(train_data_loader)
        batch_idx = self._skip_batches(data_iter, resume_batch_idx, control_group)
        restore_rng_state(resume_rng_state)
        skip_accumulation_window = False
        while True:
            try:
                batch_dict = next(data_iter)
                has_batch = True
            except StopIteration:
                batch_dict = None
                has_batch = False

            if not distributed_batch_available(has_batch, control_group):
                if not has_batch:
                    logging.info('Epoch %s input exhausted first on rank %s', self.epoch, self.rank)
                if info_dict['train_engine'] == 'torch_ddp' and batch_idx % info_dict["accum_grad"] != 0:
                    optimizer.zero_grad()
                    logging.info('Epoch %s discarded an incomplete gradient accumulation on rank %s',
                                 self.epoch, self.rank)
                break

            info_dict["tag"] = "TRAIN"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            rng_alignment_limit = info_dict.get('rng_alignment_max_speech_feat_numel', 0)
            if distributed_rng_alignment_limit_exceeded(
                    batch_dict, rng_alignment_limit, control_group):
                skip_accumulation_window = True
                logging.warning(
                    'Epoch %s batch %s exceeds CUDA/MUSA RNG alignment limit %s; '
                    'discarding this gradient-accumulation window on rank %s '
                    '(local speech_feat.numel=%s)',
                    self.epoch, batch_idx, rng_alignment_limit, self.rank,
                    batch_dict['speech_feat'].numel())
            if skip_accumulation_window:
                optimizer.zero_grad()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    skip_accumulation_window = False
                batch_idx += 1
                continue

            # Disable gradient synchronizations across DDP processes.
            if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                context = model.no_sync
            else:
                context = nullcontext

            with context():
                info_dict = batch_forward(model, batch_dict, scaler, info_dict, ref_model=self.ref_model, dpo_loss=self.dpo_loss)
                info_dict = batch_backward(model, scaler, info_dict)

            info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
            log_per_step(writer, info_dict)
            if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                self.step += 1
            # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
            if info_dict['save_per_step'] > 0 and self.step % info_dict['save_per_step'] == 0 and \
               (batch_idx + 1) % info_dict["accum_grad"] == 0:
                dist.barrier()
                info_dict['train_batch_idx'] = batch_idx
                self.cv(
                    model, optimizer, scheduler, cv_data_loader, writer,
                    info_dict, scaler, control_group=control_group, on_batch_end=False)
                model.train()
            batch_idx += 1
        # Longer ranks may still own prefetched batches; release their workers before CV.
        del data_iter
        dist.barrier()
        info_dict['train_batch_idx'] = batch_idx - 1
        self.cv(
            model, optimizer, scheduler, cv_data_loader, writer,
            info_dict, scaler, control_group=control_group, on_batch_end=True)

    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, control_group,
                           resume_batch_idx=0, resume_rng_state=None):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        model.train()
        seed_dataloader_for_epoch(train_data_loader, self.epoch, info_dict['data_seed'])
        data_iter = iter(train_data_loader)
        batch_idx = self._skip_batches(data_iter, resume_batch_idx, control_group)
        restore_rng_state(resume_rng_state)
        while True:
            try:
                batch_dict = next(data_iter)
                has_batch = True
            except StopIteration:
                batch_dict = None
                has_batch = False

            if not distributed_batch_available(has_batch, control_group):
                if not has_batch:
                    logging.info('Epoch %s input exhausted first on rank %s', self.epoch, self.rank)
                if info_dict['train_engine'] == 'torch_ddp' and batch_idx % info_dict["accum_grad"] != 0:
                    optimizer.zero_grad()
                    optimizer_d.zero_grad()
                    logging.info('Epoch %s discarded an incomplete gradient accumulation on rank %s',
                                 self.epoch, self.rank)
                break

            info_dict["tag"] = "TRAIN"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            # Disable gradient synchronizations across DDP processes.
            if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                context = model.no_sync
            else:
                context = nullcontext

            with context():
                batch_dict['turn'] = 'discriminator'
                info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                info_dict = batch_backward(model, scaler, info_dict)
            info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
            optimizer.zero_grad()
            log_per_step(writer, info_dict)
            with context():
                batch_dict['turn'] = 'generator'
                info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                info_dict = batch_backward(model, scaler, info_dict)
            info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
            optimizer_d.zero_grad()
            log_per_step(writer, info_dict)
            if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                self.step += 1
            # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
            if info_dict['save_per_step'] > 0 and self.step % info_dict['save_per_step'] == 0 and \
               (batch_idx + 1) % info_dict["accum_grad"] == 0:
                dist.barrier()
                info_dict['train_batch_idx'] = batch_idx
                self.cv(
                    model, optimizer, scheduler, cv_data_loader, writer,
                    info_dict, scaler, optimizer_d=optimizer_d,
                    scheduler_d=scheduler_d, control_group=control_group,
                    on_batch_end=False)
                model.train()
            batch_idx += 1
        # Longer ranks may still own prefetched batches; release their workers before CV.
        del data_iter
        dist.barrier()
        info_dict['train_batch_idx'] = batch_idx - 1
        self.cv(
            model, optimizer, scheduler, cv_data_loader, writer,
            info_dict, scaler, optimizer_d=optimizer_d,
            scheduler_d=scheduler_d, control_group=control_group,
            on_batch_end=True)

    @torch.inference_mode()
    def cv(self, model, optimizer, scheduler, cv_data_loader, writer, info_dict, scaler,
           optimizer_d=None, scheduler_d=None, control_group=None, on_batch_end=True):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(
            self.epoch, self.step, on_batch_end, self.rank))
        model.eval()
        seed_dataloader_for_epoch(cv_data_loader, self.epoch, info_dict['data_seed'] + 10000000)
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            rng_alignment_limit = info_dict.get('rng_alignment_max_speech_feat_numel', 0)
            if distributed_rng_alignment_limit_exceeded(
                    batch_dict, rng_alignment_limit, control_group):
                logging.warning(
                    'Epoch %s CV batch %s exceeds CUDA/MUSA RNG alignment limit %s; '
                    'skipping on rank %s (local speech_feat.numel=%s)',
                    self.epoch, batch_idx, rng_alignment_limit, self.rank,
                    batch_dict['speech_feat'].numel())
                continue

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict['turn'] = 'generator'
            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.mean().item() * num_utts)
            log_per_step(None, info_dict)
        if total_num_utts == 0:
            raise RuntimeError(
                'All CV batches were skipped by rng_alignment_max_speech_feat_numel={}.'
                .format(info_dict.get('rng_alignment_max_speech_feat_numel', 0)))
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        info_dict['step'] = self.step
        info_dict['epoch_complete'] = on_batch_end
        info_dict['world_size'] = dist.get_world_size()
        log_per_save(writer, info_dict)
        model_name = (
            'epoch_{}_whole'.format(self.epoch)
            if on_batch_end
            else 'epoch_{}_step_{}'.format(self.epoch, self.step)
        )
        save_model(
            model, model_name, info_dict, optimizer, scheduler,
            optimizer_d=optimizer_d, scheduler_d=scheduler_d, scaler=scaler)

    @staticmethod
    def _skip_batches(data_iter, resume_batch_idx, control_group):
        if resume_batch_idx <= 0:
            return 0
        logging.info('Skipping %s already completed training batches', resume_batch_idx)
        start_time = time.monotonic()
        for batch_idx in range(resume_batch_idx):
            try:
                next(data_iter)
                has_batch = True
            except StopIteration:
                has_batch = False
            if not distributed_batch_available(has_batch, control_group):
                raise RuntimeError(
                    'Checkpoint requests skipping {} batches, but epoch input ended after {}.'
                    .format(resume_batch_idx, batch_idx))
            completed = batch_idx + 1
            if completed % 100 == 0 or completed == resume_batch_idx:
                elapsed = time.monotonic() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (resume_batch_idx - completed) / rate if rate > 0 else 0
                logging.info(
                    'Resume skip progress %s/%s batches, elapsed %.1fs, ETA %.1fs',
                    completed, resume_batch_idx, elapsed, remaining)
        return resume_batch_idx
