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

from data_provider.dataset_judge import JudgeCollater, JudgeDataset
from data_provider.prompts import PROMPTS


def extract_score(text):
    matches = re.findall(r"<answer>(\d+(?:\.\d+)?)</answer>", text)
    if matches:
        return [int(m) for m in matches]
    return None

def get_win_response(text):
    score = extract_score(text)
    if score and len(score) == 2:
        if score[0] < score[1]:
            return 1
        elif score[0] > score[1]:
            return 0
        else: # tie
            return -1
    else:
        return -1

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

    dataset = JudgeDataset(
        json_paths=[f'{args.judge_input_path}_neg.jsonl',
                    f'{args.judge_input_path}_pos.jsonl'], 
        unimol_dictionary=model.encoder.unimol_dictionary,
        encoder_types=model.encoder.encoder_types,
        prompts=PROMPTS[args.prompt_option],
        root='data/',
    )

    collater = JudgeCollater(tokenizer, model.encoder.unimol_dictionary, llama_version)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collater, shuffle=False)
    
    for idx, (graph_batch, text_batch_forwad, text_batch_reverse, other_infos) in tqdm(enumerate(dataloader)):
        if args.shard_idx:
            if not (args.shard_idx[0] < idx < args.shard_idx[1]):
                continue

        for key in graph_batch.keys():
            if key == 'unimol':
                for key_ in graph_batch[key].keys():
                    graph_batch[key][key_] = graph_batch[key][key_].to(args.device)
            elif key == 'moleculestm':
                graph_batch[key] = graph_batch[key].to(args.device)

        text_batch_forwad = text_batch_forwad.to(args.device)
        text_batch_reverse = text_batch_reverse.to(args.device)

        # Generate
        outputs_forward = model.generate(
            graph_batch = graph_batch,
            text_batch = text_batch_forwad,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = terminators,
        )
        outputs_reverse = model.generate(
            graph_batch = graph_batch,
            text_batch = text_batch_reverse,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = terminators,
        )

        generated_texts_forward = tokenizer.batch_decode(outputs_forward, skip_special_tokens=True)
        generated_texts_reverse = tokenizer.batch_decode(outputs_reverse, skip_special_tokens=True)

        with open(args.judge_output_path, 'a', encoding='utf-8') as f:
            for i in range(len(generated_texts_forward)):
                pre_response = {}
                for key in other_infos[i].keys():
                    pre_response[key] = other_infos[i][key]

                pre_response['judge_forward'] = generated_texts_forward[i]
                pre_response['judge_reverse'] = generated_texts_reverse[i]
                pre_response['win_forward'] = get_win_response(generated_texts_forward[i])
                pre_response['win_reverse'] = get_win_response(generated_texts_reverse[i])
                f.write(json.dumps(pre_response, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='DongkiKim/Mol-Llama-3.1-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--prompt_option', type=str, default='judge')
    parser.add_argument('--judge_llm_path', type=str, default='./flex_llama_8b')
    parser.add_argument('--judge_input_path', type=str, default='results_generate_structure')
    parser.add_argument('--judge_output_path', type=str, default='judge_output_structure.jsonl')
    parser.add_argument('--shard_idx', nargs='+', type=int, default=[])
    args = parser.parse_args()
    main(args)


