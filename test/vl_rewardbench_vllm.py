import json
import re
import argparse
import os

from transformers import AutoProcessor
from datasets import load_dataset

from vllm import LLM, SamplingParams
import numpy as np

def main(args):
    np.random.seed(args.k)

    llm = LLM(
        args.ckpt,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": 1},  # The maximum number to accept
    )
    processor = AutoProcessor.from_pretrained(args.ckpt, use_fast=True)
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.k
    )

    dataset = load_dataset('MMInstruction/VL-RewardBench', split='test')

    ps, ss = [], []
    SYSTEM_PROMPT = (
        "You are a helpful assistant. The assistant first performs a detailed, "
        "step-by-step reasoning process in its mind and then provides the user with"
        "the answer. The reasoning process and answer are enclosed within <think> "
        "reasoning process here, explaining each step of your evaluation for both "
        "assistants </think><answer> answer here </answer>. Now the user asks you "
        "to judge the performance of two AI assistants in response to the question. "
        "Score assistants 1-10 (higher=better). Criteria includes helpfulness, "
        "relevance, accuracy, and level of detail. Avoid order, length, style or "
        "other bias. After thinking, when you finally reach a conclusion, clearly "
        "provide your evaluation scores within <answer> </answer> tags, i.e., for "
        "example, <answer>3</answer><answer>5</answer>"
    )
    
    for example in dataset:
        q = example["query"]
        if np.random.uniform() < 0.5:
            a1, a2 = example["response"]
            swap = False
        else:
            a2, a1 = example["response"]
            swap = True
        im = example["image"]

        msg = [
            {
                "role": "system", "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": f"<|vision_start|><|image_pad|><|vision_end|>\n\n[Question]\n{q}\n\n[Assistant 1's Answer]\n{a1}\n\n[Assistant 2's Answer]\n{a2}"
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
        ss.append(swap)

    for p in ps:
        assert "<|vision_start|><|image_pad|><|vision_end|>" in p["prompt"]
        assert p["prompt"].count("<|vision_start|><|image_pad|><|vision_end|>") == len(p["multi_modal_data"]["image"])

    outputs = llm.generate(ps, sampling_params=sampling_params)
    
    os.makedirs("results/vl_rewardbench", exist_ok=True)
    new_ckpt = args.ckpt.split('/')[-1] if '/' in args.ckpt else args.ckpt
    with open(f"results/vl_rewardbench/{new_ckpt}_{args.k}.jsonl", "w") as ans_file:

        for o, p, s in zip(outputs, ps, ss):
            try:
                score = list(map(int, re.findall(r"<answer>(\d+)</answer>", o.outputs[0].text)))
                if score[0] > score[1]: pred = [0, 1] if not s else [1, 0]
                elif score[0] < score[1]: pred = [1, 0] if not s else [0, 1]
                else: pred = [0, 0]
            except:
                pred = [0, 0]

            ans_file.write(json.dumps({
                "text": o.outputs[0].text.strip(),
                "pred": pred,
            }) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode with vLLM')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p probability for sampling')
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--k', type=int, default=0,
                        help="Number of trial")
    parser.add_argument('--ckpt', type=str, default="ckpts/vl",
                        help="directory for checkpoint")
    args = parser.parse_args()
    main(args)