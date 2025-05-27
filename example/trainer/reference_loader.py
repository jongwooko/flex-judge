# reference_loader.py
import torch
from models.mol_llama import MolLLaMA

_REF = None

def get_reference_model(path, vocab_size, torch_dtype, enable_flash, device=1):
    global _REF
    if _REF is None:
        _REF = MolLLaMA.from_pretrained(path, vocab_size=vocab_size, torch_dtype=torch_dtype, enable_flash=enable_flash)
        _REF = _REF.to(torch.device(f"cuda:{device}")).eval()
        for p in _REF.parameters():
            p.requires_grad = False
    return _REF
