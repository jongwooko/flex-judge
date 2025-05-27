import json
import argparse
from tqdm import tqdm
import re
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from transformers import AutoTokenizer, LlamaForCausalLM
from models.mol_llama import MolLLaMA

from pampa.dataset_judge import PAMPADataset, PAMPACollater
from pampa.prompts import PROMPTS


def main(args):
    # Load model and tokenizer
    llama_version = 'llama3' if 'Llama-3' in args.pretrained_model_name_or_path else 'llama2'
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]
    
    tokenizer.padding_side = 'left'
    if llama_version == 'llama3':
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    elif llama_version == 'llama2':
        terminators = tokenizer.eos_token_id

    model = MolLLaMA.from_pretrained(args.pretrained_model_name_or_path, vocab_size=len(tokenizer))
    model.import_judge(args.judge_llm_path)
    model = model.to(args.device)

    dataset = PAMPADataset(json_path='pampa/data/pampa.json', 
                        split='test', prompt_type=args.prompt_type, 
                        unimol_dictionary=model.encoder.unimol_dictionary,
                        prompts=PROMPTS[args.prompt_option],
                        response_path=args.judge_input_path,)

    collater = PAMPACollater(tokenizer, model.encoder.unimol_dictionary, llama_version)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collater, shuffle=False)

    pattern = r"[Ff]inal [Aa]nswer:"

    responses, answers, smiles_list = [], [], []
    for graph_batch, text_batch, answer, smiles, pre_response in tqdm(dataloader):
        for key in graph_batch.keys():
            if key == 'unimol':
                for key_ in graph_batch[key].keys():
                    graph_batch[key][key_] = graph_batch[key][key_].to(args.device)
            elif key == 'moleculestm':
                graph_batch[key] = graph_batch[key].to(args.device)
        text_batch = text_batch.to(args.device)

        # Generate
        outputs = model.generate(
            graph_batch = graph_batch,
            text_batch = text_batch,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = terminators,
        )
        
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        original_texts = tokenizer.batch_decode(text_batch['input_ids'], skip_special_tokens=False)
            
        responses.extend(generated_texts)
        answers.extend(answer)
        smiles_list.extend(smiles)

        with open(args.judge_output_path, 'a', encoding='utf-8') as f:
            for i in range(len(generated_texts)):
                pre_response[i]['Judge'] = generated_texts[i]
                f.write(json.dumps(pre_response[i], ensure_ascii=False) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='DongkiKim/Mol-Llama-3.1-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--prompt_type', type=str, default='default', choices=['default', 'rationale', 'task_info'],)
    parser.add_argument('--prompt_option', type=str, default='judge')
    parser.add_argument('--judge_llm_path', type=str, default='./flex_llama_8b')
    parser.add_argument('--judge_input_path', type=str, default=None)
    parser.add_argument('--judge_output_path', type=str, default='pampa/judge_output.jsonl')
    args = parser.parse_args()
    if args.judge_input_path is None:
        args.judge_input_path = 'pampa/results_llama3_{}.jsonl'.format(args.prompt_type)
    main(args)


