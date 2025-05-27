import json, re
import argparse

from transformers import AutoProcessor
from datasets import load_dataset, Features, Value

from vllm import LLM, SamplingParams
from PIL import Image
from torchvision import transforms

def main(args):
    llm = LLM(
        args.ckpt,
        tensor_parallel_size=4,
        enforce_eager=True
    )
    processor = AutoProcessor.from_pretrained(args.ckpt, use_fast=True)
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.k,
        stop=["</answer>", "<|im_end|>"] if args.split == "score" else ["<|im_end|>"],
        include_stop_str_in_output=True
    )

    if args.split == "pair":
        features = Features({
            'id': Value(dtype='int64', id=None), 'pair_id': Value(dtype='int64', id=None), 
            'image_path': Value(dtype='string', id=None), 'original_dataset': Value(dtype='string', id=None), 
            'instruction': Value(dtype='string', id=None), 
            'answer1': {'name': Value(dtype='string', id=None), 'answer': Value(dtype='string', id=None)}, 
            'answer2': {'name': Value(dtype='string', id=None), 'answer': Value(dtype='string', id=None)}, 
            'human': Value(dtype='string', id=None), "true_answers": Value(dtype='string', id=None)
        })
        dataset = load_dataset('json', data_files='data/mllm-judge/Benchmark/pair.jsonl', split='train', features=features)

        SYSTEM_PROMPT = (
            "You are a helpful assistant. The assistant first performs a detailed, "
            "step-by-step reasoning process in its mind and then provides the user with "
            "the answer. The reasoning process and answer are enclosed within <think> "
            "</think> and <answer> </answer> tags, respectively, i.e., <think> detailed "
            "reasoning process here, explaining each step of your evaluation for both "
            "assistants </think><answer> answer here </answer>. Now the user asks you "
            "to judge the performance of two AI assistants in response to the question. "
            "Score assistants 1-10 (higher=better). Criteria includes helpfulness, "
            "relevance, accuracy, and level of detail. Avoid order, length, style or "
            "other bias. After thinking, when you finally reach a conclusion, clearly "
            "provide your evaluation scores within <answer> </answer> tags, i.e., for "
            "example,<answer>3</answer><answer>5</answer>"
        )

    elif args.split == "score":
        features = Features({
            'id': Value(dtype='int64', id=None), 'score_id': Value(dtype='int64', id=None), 
            'image_path': Value(dtype='string', id=None), 'original_dataset': Value(dtype='string', id=None), 
            'instruction': Value(dtype='string', id=None), 
            'answer': Value(dtype='string', id=None), 'name': Value(dtype='string', id=None),
            'human': Value(dtype='int64', id=None), "true_answers": Value(dtype='string', id=None)
        })
        dataset = load_dataset('json', data_files='data/mllm-judge/Benchmark/score.jsonl', split='train', features=features)

        SYSTEM_PROMPT = (
            "You are a helpful assistant. The assistant first performs a detailed, "
            "step-by-step reasoning process in its mind and then provides the user with "
            "the answer. The reasoning process and answer are enclosed within <think> "
            "</think> and <answer> </answer> tags, respectively, i.e., <think> detailed "
            "reasoning process here, explaining each step of your evaluation for both "
            "assistants </think><answer> answer here </answer>. Now the user asks you "
            "to judge the performance of an AI assistants in response to the question. "
            "Score assistants 1-5 (higher=better). Criteria includes helpfulness, "
            "relevance, accuracy, and level of detail. Avoid length, style or "
            "other bias. After thinking, when you finally reach a conclusion, clearly "
            "provide your evaluation scores within <answer> </answer> tags, i.e., for "
            "example, <answer>3</answer>"
        )

    elif args.split == "batch":
        dataset = load_dataset('json', data_files='data/mllm-judge/Benchmark/batch.jsonl', split='train')

        SYSTEM_PROMPT = (
            "You are a helpful assistant. The assistant first performs a detailed, "
            "step-by-step reasoning process in its mind and then provides the user with "
            "the answer. The reasoning process and answer are enclosed within <think> "
            "</think> and <answer> </answer> tags, respectively, i.e., <think> detailed "
            "reasoning process here, explaining each step of your evaluation for both "
            "assistants </think><answer> answer here </answer>. Now the user asks you "
            "to judge the performance of multiple AI assistants in response to the question. "
            "Score assistants 1-10 (higher=better). Criteria includes helpfulness, "
            "relevance, accuracy, and level of detail. DO NOT assign the same score to multiple assistants. " 
            "Avoid length, style or other bias. After thinking, when you finally reach a conclusion, clearly "
            "provide your evaluation scores within <answer> </answer> tags, i.e., for "
            "example, <answer>3</answer><answer>8</answer><answer>6</answer>"
        )

    else:
        raise NotImplementedError

    ps = []
    
    for example in dataset:

        q = example["instruction"]
        im_path = example["image_path"]
        im = Image.open(f"./data/mllm-judge/image/{im_path}").convert("RGB")
        im = im.resize((args.resize, args.resize))

        if args.split == "pair":
            a1, a2 = example["answer1"]["answer"], example["answer2"]["answer"]
            text = f"<|vision_start|><|image_pad|><|vision_end|>\n\n[Question]\n{q}\n\n[Assistant 1's Answer]\n{a1}\n\n[Assistant 2's Answer]\n{a2}"

        elif args.split == "score":
            a1 = example["answer"]
            text = f"<|vision_start|><|image_pad|><|vision_end|>\n\n[Question]\n{q}\n\n[Assistant's Answer]\n{a1}"

        elif args.split == "batch":
            text = f"<|vision_start|><|image_pad|><|vision_end|>\n\n[Question]\n{q}\n\n"
            a = example["answers"]
            for i in range(len(a)):
                text += f"[Assistant {i+1}'s Answer]\n{a[i]['answer']}\n\n"
            text = text.strip()
            
        msg = [
            {
                "role": "system", "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": text
                    }
                ]
            },
        ]


        text = processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        p = {
            "prompt": text, 
            "multi_modal_data": {"image": [im]}
        }
        ps.append(p)

    outputs = llm.generate(ps, sampling_params=sampling_params)

    new_ckpt = args.ckpt.split('/')[-1] if '/' in args.ckpt else args.ckpt
    with open(f"results/mllm-judge/{new_ckpt}_{args.k}_{args.split}.jsonl", "w") as ans_file:

        for o, p in zip(outputs, ps):
            try:
                score = list(map(int, re.findall(r"<answer>(\d+)</answer>", o.outputs[0].text)))
                if score[0] > score[1]: pred = "A"
                elif score[0] < score[1]: pred = "B"
                else: pred = "C"
            except:
                pred = "C"

            ans_file.write(json.dumps({
                "text": o.outputs[0].text.strip(),
                "pred": pred,
            }) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode with vLLM')
    parser.add_argument('--temperature', type=float, default=0.4,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.8,
                        help='Top-p probability for sampling')
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--k', type=int, default=0,
                        help="Number of trial")
    parser.add_argument('--ckpt', type=str, default="ckpts/vl",
                        help="directory for checkpoint")
    parser.add_argument('--split', type=str, default="pair",
                        help="evaluation type")
    parser.add_argument('--resize', type=int, default=448,
                        help="image resize for computation constraints")
    args = parser.parse_args()
    main(args)