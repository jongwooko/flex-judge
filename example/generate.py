import os
import argparse
import warnings
from collections import defaultdict
import yaml
import json
from easydict import EasyDict as edict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger

from transformers import AutoTokenizer

from models.mol_llama import MolLLaMA
from utils.configuration_mol_llama import MolLLaMAConfig
from data_provider.stage2_dm import Stage2DM, Stage2Collater

from utils.dist_funs import MyDeepSpeedStrategy

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium')
device = "cuda:0"


def parse_tasks(tasks):
    tasks = tasks.split(',')
    out = defaultdict(list)
    for task in tasks:
        split = task.split('_')
        if len(split) == 1:
            out[task] = []
        elif len(split) == 2:
            out[task.split('_')[0]].append(task.split('_')[1])

    return out

def edict_to_dict(config):
    """
    Convert an EasyDict object to a regular dictionary.
    """
    if isinstance(config, edict):
        return {k: edict_to_dict(v) for k, v in config.items()}
    else:
        return config
    

def main(model_config, generate_config, data_config):
    # Load model and tokenizer
    llama_version = 'llama3' if 'Llama-3' in generate_config.pretrained_model_name_or_path else 'llama2'
    tokenizer = AutoTokenizer.from_pretrained(generate_config.pretrained_model_name_or_path)
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]
    tokenizer.padding_side = 'left'
    if llama_version == 'llama3':
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    elif llama_version == 'llama2':
        terminators = tokenizer.eos_token_id

    model = MolLLaMA.from_pretrained(generate_config.pretrained_model_name_or_path, vocab_size=len(tokenizer)).to(device)

    dataset = Stage2DM(
        tokenizer=tokenizer,
        llama_version=llama_version,
        batch_size=data_config.batch_size,
        root=data_config.root,
        num_workers=data_config.num_workers,
        unimol_dictionary=model.encoder.unimol_dictionary,
        encoder_types=model.encoder.encoder_types,
        data_types=data_config.data_types,
        single_turn=True,
    )
    collater = Stage2Collater(tokenizer, llama_version, model.encoder.unimol_dictionary.pad(), model.encoder.encoder_types)
    dataloader = DataLoader(dataset.train_dataset, batch_size=data_config.batch_size, collate_fn=collater, shuffle=False)

    if generate_config.resume:
        with open(generate_config.output_path, 'r') as f:
            lines = f.readlines()
            start_index = len(lines) // 16 # batch_size
    
    responses, infos = [], []
    for idx, (graph_batch, text_batch, other_infos) in tqdm(enumerate(dataloader)):
        if generate_config.resume:
            if idx < start_index:
                continue
            else:
                generate_config.resume = False
        for key in graph_batch.keys():
            if key == 'unimol':
                for key_ in graph_batch[key].keys():
                    graph_batch[key][key_] = graph_batch[key][key_].to(device)
            elif key == 'moleculestm':
                graph_batch[key] = graph_batch[key].to(device)
        text_batch = text_batch.to(device)

        # Generate
        outputs = model.generate(
            graph_batch = graph_batch,
            text_batch = text_batch,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = terminators,
            max_new_tokens=256,
            do_sample=True,
            temperature=generate_config.temperature,
        )
        
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Save responses and infos as jsonl file
        with open(generate_config.output_path, 'a') as f:
            for i, response in enumerate(generated_texts):
                save = {}
                for key in other_infos.keys():
                    save[key] = other_infos[key][i]
                save['response'] = response
                f.write(json.dumps(save) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2 generating')
    parser.add_argument('--generate_config_path', type=str, default='configs/stage2/generate_config_structure.yaml')
    parser.add_argument('--output_path', type=str, default='results_generate_structure.jsonl')
    parser.add_argument('--resume', action='store_true', help='Resume from the last checkpoint')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')

    args = parser.parse_args()

    model_config = MolLLaMAConfig()
    generate_config = edict(yaml.load(open(args.generate_config_path), Loader=yaml.FullLoader))
    data_config = generate_config.dataset
    generate_config.output_path = args.output_path
    generate_config.resume = args.resume
    generate_config.temperature = args.temperature

    print('-'*60)
    print(f'batch_size: {data_config.batch_size}\tnum_devices: {len(generate_config.devices)}\taccumulate_grad_batches: {generate_config.accumulate_grad_batches}')
    print(f'Total batch size: {data_config.batch_size * len(generate_config.devices) * generate_config.accumulate_grad_batches}')
    print('-'*60)
    print(f'Data Types:')
    for data_type in data_config.data_types:
        print(f'  - {data_type}')
    print('-'*60)

    main(model_config, generate_config, data_config)