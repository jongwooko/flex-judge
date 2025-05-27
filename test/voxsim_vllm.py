import json, re
import argparse
import os
import numpy as np
import soundfile as sf
import librosa
from transformers import AutoProcessor
from datasets import load_dataset
from tqdm import tqdm

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
    voxsim_list_path = args.voxsim_txt_path
    if not os.path.exists(voxsim_list_path):
        raise FileNotFoundError(f"List file not found: {voxsim_list_path}")

    data = []
    with open(voxsim_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 5:
                print(f"Warning: Malformed line: {line}")
                continue
            data.append({
                'audio_path1': parts[0],
                'audio_path2': parts[1],
                'is_same_speaker': parts[2],
                'similarity_score': parts[4]
            })

    from datasets import Dataset
    dataset = Dataset.from_list(data)

    def process_row(row):
        row['full_audio_path1'] = os.path.join(args.voxsim_dir, row['audio_path1'])
        row['full_audio_path2'] = os.path.join(args.voxsim_dir, row['audio_path2'])

        row['is_same_speaker'] = int(row['is_same_speaker'])
        row['similarity_score'] = float(row['similarity_score'])

        return row

    dataset = dataset.map(process_row)
    print(f"Loaded {len(dataset)} samples from {voxsim_list_path}")

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    llm = LLM(
        args.ckpt,
        tensor_parallel_size=args.gpu_count,
        max_num_seqs=16,
        limit_mm_per_prompt={"audio": 2},
        enforce_eager=True,
        dtype="bfloat16"
    )

    processor = AutoProcessor.from_pretrained(args.ckpt, use_fast=True)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    system_prompt = """You are a helpful assistant. The assistant first performs a detailed, \
step-by-step reasoning process in its mind and then provides the user with the answer. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, \
respectively, i.e., <think> detailed reasoning process here, explaining each step of your evaluation \
for an AI assistant </think><answer> answer here </answer>. Now the user asks you to judge the performance \
of an audio AI assistant. Score the assistant from 1.0 to 5.0 in first decimal. After thinking, \
when you finally reach a conclusion, clearly provide your evaluation scores within <answer> </answer> tags, \
i.e., for example, <answer>3.8</answer>."""

    user_prompt = """\n\n[Question]\nEvaluate if these two speech audios share the same speaker. \
Do NOT consider the content of the speech. ONLY focus on the speaker-related characteristics, \
such as vocal, tone, pitch, accent, intonation.\n(1) <|audio_bos|><|AUDIO|><|audio_eos|>\n(2) \
<|audio_bos|><|AUDIO|><|audio_eos|>\n\n[Assistant's Answer]\nThe given two audios are from the same speaker."""

    predictions = []
    ground_truths = []
    ps = []

    for idx, sample in tqdm(enumerate(dataset)):
        print(f"Processing sample {idx+1}/{len(dataset)}: {sample['full_audio_path1']} vs {sample['full_audio_path2']}")

        audio_data1, sr1 = load_audio_from_file(sample['full_audio_path1'])
        if audio_data1 is None:
            print(f"Warning: Could not load first audio file: {sample['full_audio_path1']}")
            continue
        
        audio_data2, sr2 = load_audio_from_file(sample['full_audio_path2'])
        if audio_data2 is None:
            print(f"Warning: Could not load second audio file: {sample['full_audio_path2']}")
            continue

        prompt = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
              f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
              "<|im_start|>assistant\n")

        p = {
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio_data1, audio_data2]}
        }
        ps.append(p)
        ground_truths.append(sample['similarity_score'])

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
                    pred = 3.5
            except Exception as e:
                print(f"Error extracting score for sample {i}: {e}")
                pred = 3.5

            predictions.append(pred)

            sample_data = {}
            if i < len(dataset):
                sample_data = {
                    "audio_path1": dataset[i]['audio_path1'],
                    "audio_path2": dataset[i]['audio_path2'],
                    "is_same_speaker": dataset[i]['is_same_speaker'],
                }

            ans_file.write(json.dumps({
                "text": o.outputs[0].text.strip(),
                "pred": pred,
                "gt": ground_truths[i],
                **sample_data,
            }) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate speaker similarity using vLLM')
    parser.add_argument('--root', type=str, default="/path/to/voxsim",
                        help="Root directory for VoxSim dataset")
    parser.add_argument('--voxsim_txt_path', type=str, default="voxsim_test_list.txt",
                        help='Path to VoxSim test list file')
    parser.add_argument('--voxsim_dir', type=str, default="wav",
                        help='Base directory containing audio files')
    parser.add_argument('--output_dir', type=str, default="results/voxsim",
                        help='Directory to save results')
    parser.add_argument('--temperature', type=float, default=0.6,
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
    args.voxsim_txt_path = os.path.join(args.root, args.voxsim_txt_path)
    args.voxsim_dir = os.path.join(args.root, args.voxsim_dir)
    main(args)