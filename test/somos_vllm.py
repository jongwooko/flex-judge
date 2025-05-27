import json, re
import argparse
import os
import numpy as np
import soundfile as sf
import librosa
from transformers import AutoProcessor

from vllm import LLM, SamplingParams

def load_audio_from_file(audio_path):
    try:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(y=audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None, None

def main(args):
    somos_txt_path = args.somos_txt_path
    somos_dir = args.somos_dir

    if not os.path.exists(somos_txt_path):
        print(f"Error: SOMOS list file not found: {somos_txt_path}")
        return

    print(f"Loading SOMOS dataset from {somos_txt_path}...")
    audio_paths = []
    scores = []

    with open(somos_txt_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue

            split = line.find(',')
            if split == -1:
                print(f"Warning: Invalid line format: {line}")
                continue

            wav = line[:split]
            score = line[split+1:]

            audio_path = os.path.join(somos_dir, wav)
            audio_paths.append(audio_path)
            scores.append(float(score))

    print(f"Loaded {len(audio_paths)} entries from SOMOS dataset.")


    if args.max_samples > 0 and len(audio_paths) > args.max_samples:
        print(f"Limiting to {args.max_samples} samples")
        audio_paths = audio_paths[:args.max_samples]
        scores = scores[:args.max_samples]


    llm = LLM(
        args.ckpt,
        tensor_parallel_size=args.gpu_count,
        max_num_seqs=16,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=True,
        dtype="bfloat16",
    )
    processor = AutoProcessor.from_pretrained(args.ckpt, use_fast=True)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    system_prompt = """You are a helpful assistant. The assistant first performs a detailed, \
step-by-step reasoning process in its mind and then provides the user with the answer. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> \
tags, respectively, i.e., <think> detailed reasoning process here, explaining each step of \
your evaluation for an AI assistant </think><answer> answer here </answer>. Now the user asks \
you to judge the performance of an audio generative AI assistant in response to the question. \
Listen to the generated speech audio, and score this speech on a scale from 1.0 to 5.0 in FISRT DECIMAL. \
Consider the following criteria when scoring:\n1 - Very Bad: The speech is very unnatural, has poor audio \
quality, and is nearly impossible to understand.\n2 - Poor: The speech sounds unnatural and/or noisy. \
Only a few words are understandable.\n3 - Fair: The speech is somewhat unnatural or contains noticeable \
noise, but the overall meaning is understandable.\n4 - Good: The speech is generally natural and clear, \
with most of the content easy to understand.\n5 - Excellent: The speech is very natural, high in audio \
quality, and fully intelligible.\nDo NOT consider the content of the speech. After thinking, when you \
finally reach a conclusion, clearly provide your evaluation scores within <answer> </answer> tags, \
i.e., for example, <answer>3.8</answer>."""

    user_prompt = """\n\n[Question]\nGenerate clear, natural, \
and understandable high-quality speech audio.\n\n[Assistant's Answer]\
\nHere is the speech I generated: <|audio_bos|><|AUDIO|><|audio_eos|>"""

    predictions = []
    ground_truths = []
    ps = []

    for idx, (audio_path, score) in enumerate(zip(audio_paths, scores)):
        print(f"Processing audio {idx+1}/{len(audio_paths)}: {audio_path}")

        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        audio_data, sr = load_audio_from_file(audio_path)
        if audio_data is None:
            continue

        prompt = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
              f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
              "<|im_start|>assistant\n")

        p = {
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio_data]}
        }
        ps.append(p)
        ground_truths.append(round(float(score), 1))

    print(f"Running inference on {len(ps)} audio samples...")
    outputs = llm.generate(ps, sampling_params=sampling_params)

    os.makedirs(args.output_dir, exist_ok=True)
    new_ckpt = args.ckpt.split('/')[-1] if '/' in args.ckpt else args.ckpt

    with open(f"{args.output_dir}/{new_ckpt}.jsonl", "w") as ans_file:
        for i, o in enumerate(outputs):
            try:
                answer_match = re.search(r"<answer>(\d+\.?\d*)</answer>", o.outputs[0].text)
                if answer_match:
                    pred = float(answer_match.group(1))
                else:
                    pred = 3.0
            except Exception as e:
                print(f"Error extracting score for sample {i}: {e}")
                pred = 3.0

            predictions.append(pred)

            ans_file.write(json.dumps({
                "text": o.outputs[0].text.strip(),
                "pred": pred,
                "gt": ground_truths[i],
                "audio_path": audio_paths[i] if i < len(audio_paths) else "",
            }) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SOMOS audio quality using vLLM')
    parser.add_argument('--root', type=str, default="/path/to/somos",
                        help="Root directory for SOMOS dataset")
    parser.add_argument('--somos_txt_path', type=str, default="training_files/split1/clean/test_mos_list.txt",
                        help='Path to SOMOS metadata TXT file')
    parser.add_argument('--somos_dir', type=str, default="audios",
                        help='Directory containing SOMOS audio files')
    parser.add_argument('--output_dir', type=str, default="results/somos",
                        help='Directory to save results')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p probability for sampling')
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--ckpt', type=str, default="ckpts/omni",
                        help="directory for checkpoint")
    parser.add_argument('--max_samples', type=int, default=0,
                        help="Maximum number of samples to process (0 for all)")
    parser.add_argument('--gpu_count', type=int, default=2,
                        help="Number of GPUs to use for inference")
    args = parser.parse_args()
    args.somos_txt_path = os.path.join(args.root, args.somos_txt_path)
    args.somos_dir = os.path.join(args.root, args.somos_dir)
    main(args)