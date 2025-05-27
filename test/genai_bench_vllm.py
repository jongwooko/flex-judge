import json, re
import argparse
import os

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from datasets import load_dataset

from vllm import LLM, SamplingParams
import time

def main(args):
    llm = LLM(
        args.ckpt,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"video": 2, "image": 3},  # The maximum number to accept
        enforce_eager=True,
        dtype="bfloat16",
    )
    processor = AutoProcessor.from_pretrained(args.ckpt, use_fast=True)
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.k
    )

    ps = []

    if args.split == 'video':
        dataset = load_dataset('TIGER-Lab/GenAI-Bench', 'video_generation', split='test_v1')
        SYSTEM_PROMPT = (
            "You are a helpful assistant. The assistant first performs a detailed, "
            "step-by-step reasoning process in its mind and then provides the user with "
            "the answer. The reasoning process and answer are enclosed within <think> "
            "</think> and <answer> </answer> tags, respectively, i.e., <think> detailed "
            "reasoning process here, explaining each step of your evaluation for both "
            "assistants </think><answer> answer here </answer>. Now the user asks you "
            "to judge the performance of two AI assistants. You have only FOUR Option:\n"
            "Option 1. Model A is better: [[A>B]]\n"
            "Option 2. Model B is better: [[B>A]]\n"
            "Option 3. Tie, relatively the same acceptable quality: [[A=B=Good]]\n"
            "Option 4. Both are bad: [[A=B=Bad]]\n"
            "Assess the quality of generated videos. Consider inappropriateness the following sub-dimensions: "
            "Alignment with prompt, Naturalness, Visual Quality, Consistent and Coherent"
            "and Dynamic, Visual Appealing, are correctly represented. Avoid order, length, style or "
            "other bias. After thinking, when you finally reach a conclusion, clearly "
            "provide your evaluation option within <answer> </answer> tags. i.e., for "
            "example, <answer>[[B>A]]</answer>."
        )

        input_type = "Video"
        placeholder = "<|video_pad|>"

    elif args.split == "editing":
        dataset = load_dataset('TIGER-Lab/GenAI-Bench', 'image_edition', split='test_v1')
        SYSTEM_PROMPT = (
            "You are a helpful assistant. The assistant first performs a detailed, "
            "step-by-step reasoning process in its mind and then provides the user with "
            "the answer. The reasoning process and answer are enclosed within <think> "
            "</think> and <answer> </answer> tags, respectively, i.e., <think> detailed "
            "reasoning process here, explaining each step of your evaluation for both "
            "assistants </think><answer> answer here </answer>. Now the user asks you "
            "to judge the performance of two AI assistants. You have only FOUR Option:\n"
            "Option 1. Model A is better: [[A>B]]\n"
            "Option 2. Model B is better: [[B>A]]\n"
            "Option 3. Tie, relatively the same acceptable quality: [[A=B=Good]]\n"
            "Option 4. Both are bad: [[A=B=Bad]]\n"
            "Assess the quality of editied images. Consider inappropriateness the following sub-dimensions: "
            "Alignment with editing prompt, Overedited, Naturalness, Artifact, "
            "and Visual Appealing, are correctly represented. Avoid order, length, style or "
            "other bias. After thinking, when you finally reach a conclusion, clearly "
            "provide your evaluation option within <answer> </answer> tags. i.e., for "
            "example, <answer>[[A=B=Bad]]</answer>."
        )

        input_type = "Image"
        placeholder = "<|image_pad|>"

    elif args.split == "image":
        dataset = load_dataset('TIGER-Lab/GenAI-Bench', 'image_generation', split='test_v1')
        SYSTEM_PROMPT = (
            "You are a helpful assistant. The assistant first performs a detailed, "
            "step-by-step reasoning process in its mind and then provides the user with "
            "the answer. The reasoning process and answer are enclosed within <think> "
            "</think> and <answer> </answer> tags, respectively, i.e., <think> detailed "
            "reasoning process here, explaining each step of your evaluation for both "
            "assistants </think><answer> answer here </answer>. Now the user asks you "
            "to judge the performance of two AI assistants. You have only FOUR Option:\n"
            "Option 1. Model A is better: [[A>B]]\n"
            "Option 2. Model B is better: [[B>A]]\n"
            "Option 3. Tie, relatively the same acceptable quality: [[A=B=Good]]\n"
            "Option 4. Both are bad: [[A=B=Bad]]\n"
            "Assess the quality of generated images. Consider inappropriateness the following sub-dimensions: "
            "Alignment with prompt, Naturalness, Artifact, "
            "and Visual Appealing, are correctly represented. Avoid order, length, style or "
            "other bias. After thinking, when you finally reach a conclusion, clearly "
            "provide your evaluation option within <answer> </answer> tags. i.e., for "
            "example, <answer>[[B>A]]</answer>."
        )

        input_type = "Image"
        placeholder = "<|image_pad|>"

    else:
        raise NotImplementedError
    
    for example in dataset:
        

        if args.split == "video":
            q = example["prompt"]
            im1_path, im2_path = example["left_video"].split('/')[-1], example["right_video"].split('/')[-1]

            folder = "" # You have to change `folder`
            im1_path = f"{folder}/{im1_path}"
            im2_path = f"{folder}/{im2_path}"
            
            msg = [
                {
                    "role": "system", "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video", "video": im1_path, "max_pixels": 224 * 224, "fps": 1.0,
                        },
                        {
                            "type": "video", "video": im2_path, "max_pixels": 224 * 224, "fps": 1.0,
                        },
                    ]
                },
            ]

        elif args.split == "image":
            q = example["prompt"]
            im1, im2 = example["left_image"], example["right_image"]
            im1, im2 = im1.resize((448, 448)), im2.resize((448, 448))


        elif args.split == "editing":
            q1, q2, q3 = example["source_prompt"], example["target_prompt"], example["instruct_prompt"]
            im1, im2, im3 = example["source_image"], example["left_output_image"], example["right_output_image"]
            im1, im2, im3 = im1.resize((448, 448)), im2.resize((448, 448)), im3.resize((448, 448))

        if args.split == "editing":
            msg2 = [
                {
                    "role": "system", "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": (
                                f"[Source Image Prompt]\n{q1}\n\n[Target Image Prompt]\n{q2}\n\n[Editing Prompt]\n{q3}\n\n[Source Image]\n<|vision_start|>{placeholder}<|vision_end|>\n\n"
                                f"[Assistant A's Edited {input_type}]\n<|vision_start|>{placeholder}<|vision_end|>\n\n[Assistant Edited B's {input_type}]\n<|vision_start|>{placeholder}<|vision_end|>"
                            )
                        }
                    ]
                },
            ]
            
        else:
            msg2 = [
                {
                    "role": "system", "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": f"[Question]\n{q}\n\n[Assistant A's {input_type}]\n<|vision_start|>{placeholder}<|vision_end|>\n\n[Assistant B's {input_type}]\n<|vision_start|>{placeholder}<|vision_end|>"
                            
                        }
                    ]
                },
                        
            ]

        text = processor.apply_chat_template(
            msg2, tokenize=False, add_generation_prompt=True
        )

        if args.split == "video":

            _, video_inputs = process_vision_info(msg)

            p = {
                "prompt": text,
                "multi_modal_data": {"video": video_inputs}
            }

        elif args.split == "image":
            p = {
                "prompt": text,
                "multi_modal_data": {"image": [im1, im2]}
            }

        elif args.split == "editing":
            p = {
                "prompt": text,
                "multi_modal_data": {"image": [im1, im2, im3]}
            }
        
        ps.append(p)

    outputs = llm.generate(ps, sampling_params=sampling_params)

    os.makedirs("results/genai_bench", exist_ok=True)
    new_ckpt = args.ckpt.split('/')[-1] if '/' in args.ckpt else args.ckpt
    with open(f"results/genai_bench/{new_ckpt}_{args.split}_{args.k}.jsonl", "w") as ans_file:
        for o in outputs:
            try:
                score = re.findall(r"<answer>(\[\[.*?\]\])</answer>", o.outputs[0].text)[0]

                if score in ["[[A>B]]", "[[B<A]]"]:
                    pred = 'leftvote'
                elif score in ["[[A<B]]", "[[B>A]]"]:
                    pred = 'rightvote'
                elif score in ["[[A=B=Good]]"]:
                    pred = 'tievote'
                else:
                    pred = 'bothbad_vote'

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
    parser.add_argument('--split', type=str, default='alignment',
                        help="image, video, editing")
    args = parser.parse_args()
    main(args)