import os
from typing import Any, Dict
import json

import torch
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl

from models.mol_llama import MolLLaMA
from trainer.optims import LinearWarmupCosineLRScheduler
from trainer.reference_loader import get_reference_model
from copy import deepcopy


def batch_to_device(batch, device):
    """
    Recursively move everything in `batch` to `device`.
    Works for:
     - torch.Tensor
     - dicts of things
     - PyG Data / DataBatch (they have a .to() method)
     - lists/tuples of things
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        cls = type(batch)
        return cls(batch_to_device(x, device) for x in batch)
    if hasattr(batch, "to"):
        # PyG Data and most PyTorch objects that implement .to()
        return batch.to(device)
    return batch  # anything else, leave it alone

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)
    

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class Stage3Trainer(pl.LightningModule):
    def __init__(self, vocab_size, train_config):
        super().__init__()
        self.train_config = train_config
        if train_config.precision == 'bf16-mixed':
            torch_dtype = "bfloat16"
        elif train_config.precision == '16':
            torch_dtype = "float16"
        elif train_config.precision == '32':
            torch_dtype = "float32"
        self.torch_dtype = torch_dtype
        self.vocab_size = vocab_size

        self.mol_llama = MolLLaMA.from_pretrained(train_config.stage2_path, vocab_size=vocab_size, torch_dtype=torch_dtype, enable_flash=train_config.enable_flash)

        self.beta = 0.1 ###
        self.ref_device = train_config.reference_device

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.train_config.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.init_lr, weight_decay=self.train_config.weight_decay)
        if self.train_config.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.train_config.max_epochs, self.train_config.min_lr, self.train_config.init_lr, warmup_steps, self.train_config.warmup_lr)
        elif self.train_config.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def _get_log_probs(self, model, graph_batch, text_batch):
        outputs = model(graph_batch, text_batch)
        batch_size = text_batch.input_ids.size(0)
        logits = outputs.logits[:, :-1, :]
        labels = text_batch.input_ids[:, 1:]
        log_probs = -F.cross_entropy(
            logits.contiguous().view(-1, logits.size(-1)),
            labels.contiguous().view(-1),
            reduction='none',
        ).view(batch_size, -1)
        return log_probs.sum(dim=1)

    def training_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch              
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        # lazy load
        reference_model = get_reference_model(self.train_config.stage2_path, vocab_size=self.vocab_size, torch_dtype=self.torch_dtype, enable_flash=self.train_config.enable_flash, device=self.ref_device)

        batch_size = text_batch.input_ids.size(0) // 2

        ###============== Overall Loss ===================###
        pi_rj, pi_ch = torch.chunk(self._get_log_probs(self.mol_llama, graph_batch, text_batch), 2)

        ref_dev = torch.device(f"cuda:{self.ref_device}")    
        gb_ref = batch_to_device(graph_batch, ref_dev)
        txt_ref = text_batch.to(ref_dev)
        
        with torch.no_grad():
            rr, rc = torch.chunk(self._get_log_probs(reference_model, gb_ref, txt_ref), 2)
        
        rc = rc.to(self.device)
        rr = rr.to(self.device)
        
        logits = pi_ch - pi_rj - (rc - rr)
        dpo_loss = -F.logsigmoid(logits * self.beta).mean()

        loss = {'loss': dpo_loss}

        self.log("molecule dpo loss", float(loss['loss']), batch_size=batch_size, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss['loss']