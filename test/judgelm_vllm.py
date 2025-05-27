import json, re
import argparse
import os

from transformers import AutoTokenizer
from datasets import load_dataset

from vllm import LLM, SamplingParams


def main(args):
    llm = LLM(
        args.ckpt,
        tensor_parallel_size=4,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.k
    )

    dataset = load_dataset('json', data_files='data/judgelm/judgelm_val_5k.jsonl', split='train')
    answers = load_dataset('json', data_files='data/judgelm/judgelm_val_5k_gpt4.jsonl', split='train')

    ps = []
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
        "example, <answer>3</answer><answer>5</answer>"
    )

    for example, answer in zip(dataset, answers):
        assert example["question_id"] == answer["question_id"]
        q = example["question_body"]
        a1 = example["answer1_body"]
        a2 = example["answer2_body"]

        msg = [
            {
                "role": "system", "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""[Question]\n{q}\n\n[Assistant 1's Answer]\n{a1}\n\n[Assistant 2's Answer]\n{a2}"""
            },
        ]

        p = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        ps.append(p)

    outputs = llm.generate(ps, sampling_params=sampling_params)

    os.makedirs("results/judgelm", exist_ok=True)
    new_ckpt = args.ckpt.split('/')[-1] if '/' in args.ckpt else args.ckpt
    with open(f"results/judgelm/{new_ckpt}_{args.k}.jsonl", "w") as ans_file:
        for o in outputs:
            try:
                pred = re.findall(r"<answer>(\[\[.*?\]\])</answer>", o.outputs[0].text)[0]
            except:
                pred = "None"

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
