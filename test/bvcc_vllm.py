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
    bvcc_txt_path = args.bvcc_txt_path
    bvcc_dir = args.bvcc_dir

    if not os.path.exists(bvcc_txt_path):
        print(f"Error: BVCC list file not found: {bvcc_txt_path}")
        return

    print(f"Loading BVCC dataset from {bvcc_txt_path}...")
    audio_paths = []
    scores = []

    with open(bvcc_txt_path, 'r') as f:
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

            audio_path = os.path.join(bvcc_dir, wav)
            audio_paths.append(audio_path)
            scores.append(float(score))

    print(f"Loaded {len(audio_paths)} entries from BVCC dataset.")


    if args.max_samples > 0 and len(audio_paths) > args.max_samples:
        print(f"Limiting to {args.max_samples} samples")
        audio_paths = audio_paths[:args.max_samples]
        scores = scores[:args.max_samples]

    llm = LLM(
        args.ckpt,
        tensor_parallel_size=args.gpu_count,
        max_num_seqs=4096,
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

    SYSTEM_PROMPT = """You are a helpful assistant. The assistant first performs a detailed, \
step-by-step reasoning process in its mind and then provides the user with \
the answer. The reasoning process and answer are enclosed within <think> \
</think> and <answer> </answer> tags, respectively, i.e., <think> detailed \
reasoning process here, explaining each step of your evaluation for both \
assistants </think><answer> answer here </answer>. Now the user asks you \
to judge the score of the quality of this speech sample. \
Score the audio on a mean opinion score (MOS) scale from 1 to 5 (higher = better) based on overall quality. \
Please consider the quality of the speech, the naturalness, and the intelligibility of the speech. \
1 – Very Bad: The speech is very unnatural, has poor audio quality, and is nearly impossible to understand. \
2 – Poor: The speech sounds unnatural and/or noisy. Only a few words are understandable. \
3 – Fair: The speech is somewhat unnatural or contains noticeable noise, but the overall meaning is understandable. \
4 – Good: The speech is generally natural and clear, with most of the content easy to understand. \
5 – Excellent: The speech is very natural, high in audio quality, and fully intelligible. \
After thinking, when you finally reach a conclusion, clearly \
provide your evaluation score within <answer> </answer> tags, i.e., for \
example,<answer>3.1</answer>"""

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

        question = "Rate this audio on a mean opinion score (MOS) scale from 1 to 5. YOU MUST end with <answer>X</answer> where X is your score."
        text = f"<|audio_bos|><|AUDIO|><|audio_eos|>\n\n{question}\n\nRemember to end with <answer>X</answer>."

        prompt = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
              f"<|im_start|>user\n{text}<|im_end|>\n"
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
                    score_match = re.search(r"(?:score|rating)(?:\s+is|\s*[:=]\s*)?\s*(\d+\.?\d*)", o.outputs[0].text, re.IGNORECASE)
                    if score_match:
                        pred = float(score_match.group(1))
                    else:
                        num_match = re.search(r"\b[1-5](?:\.\d*)?\b", o.outputs[0].text)
                        pred = float(num_match.group(0)) if num_match else 0.0
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
    parser = argparse.ArgumentParser(description='Evaluate BVCC audio quality using vLLM')
    parser.add_argument('--root', type=str, default="/path/to/bvcc",
                        help="Root directory for BVCC dataset")
    parser.add_argument('--bvcc_txt_path', type=str, default="main/DATA/sets/test_mos_list.txt",
                        help='Path to BVCC metadata TXT file')
    parser.add_argument('--bvcc_dir', type=str, default="main/DATA/wav",
                        help='Directory containing BVCC audio files')
    parser.add_argument('--output_dir', type=str, default="results/bvcc",
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
    args.bvcc_txt_path = os.path.join(args.root, args.bvcc_txt_path)
    args.bvcc_dir = os.path.join(args.root, args.bvcc_dir)
    main(args)