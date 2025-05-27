from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import json, re
import argparse


def main(args):
    ckpt = "nuojohnchen/JudgeLRM-7B"
    model = LLM(ckpt, tensor_parallel_size=2)
    tok = AutoTokenizer.from_pretrained(ckpt)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
    )

    # ds = load_dataset('BAAI/JudgeLM-100K', split='train')
    ds = load_dataset('json', data_files='data/judgelm/judgelm_train_100k.jsonl', split='train')
    question = ds['question_body']
    answer1 = ds['answer1_body']
    answer2 = ds['answer2_body']

    ps = []
    SYSTEM_PROMPT = """You are a helpful assistant. The assistant first performs a detailed, \
step-by-step reasoning process in its mind and then provides the user with \
the answer. The reasoning process and answer are enclosed within <think> \
</think> and <answer> </answer> tags, respectively, i.e., <think> detailed \
reasoning process here, explaining each step of your evaluation for both \
assistants </think><answer> answer here </answer>. Now the user asks you \
to judge the performance of two AI assistants in response to the question. \
Score assistants 1-10 (higher=better). Criteria includes helpfulness, \
relevance, accuracy, and level of detail. Avoid order, length, style or \
other bias. After thinking, when you finally reach a conclusion, clearly \
provide your evaluation scores within <answer> </answer> tags, i.e., for \
example,<answer>3</answer><answer>5</answer>"""

    ps = []
    for q, a1, a2 in zip(question, answer1, answer2):
        messages = [
            {
                "role": "system", "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""[Question]
{q}

[Assistant 1's Answer]
{a1}

[Assistant 2's Answer]
{a2}"""
            },
        ]

        p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>\n"
        ps.append(p)

    outputs = model.generate(ps, sampling_params=sampling_params)

    with open(f"judgelrm_7b_{args.seed}_{args.temperature}.jsonl", "w") as ans_file:
        for o, p in zip(outputs, ps):
            ans_file.write(json.dumps({
                "prompt": p,
                "text": o.outputs[0].text.strip(),
                "tokens": o.outputs[0].token_ids,
            }) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode with vLLM')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p probability for sampling')
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    # 13, 21, 42, 79, 100
    args = parser.parse_args()
    main(args)