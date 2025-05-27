import json, re
import argparse
import os
import numpy as np
import soundfile as sf
import librosa
from transformers import AutoProcessor
from datasets import load_dataset

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
    csv_path = args.nisqa_csv_path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    dataset = load_dataset('csv', data_files=csv_path, split='train')

    def process_row(row):
        row['audio_path'] = os.path.join(args.nisqa_dir, row['filepath_deg'])
        audio_data, sr = load_audio_from_file(row['audio_path'])
        if audio_data is not None:
            row['audio'] = {'array': audio_data, 'sampling_rate': sr}
        return row

    dataset = dataset.map(process_row)

    print(f"Loaded {len(dataset)} entries from NISQA metadata.")

    llm = LLM(
        args.ckpt,
        tensor_parallel_size=args.gpu_count,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=True,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=16,
    )

    processor = AutoProcessor.from_pretrained(args.ckpt, use_fast=True)
    sampling_params = SamplingParams(
        max_tokens=4096,
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

    if args.max_samples > 0:
        dataset = dataset.select(range(args.max_samples))

    for idx, row in enumerate(dataset):
        print(f"Processing audio {idx+1}/{len(dataset)}: {row['audio_path']}")
        audio_data = row['audio']['array']
        sr = row['audio']['sampling_rate']

        if not os.path.exists(row['audio_path']):
            print(f"Warning: Audio file not found: {row['audio_path']}")
            continue

        prompt = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
              f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
              "<|im_start|>assistant\n")

        p = {
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio_data]}
        }
        ps.append(p)
        ground_truths.append(round(float(row['mos']), 1))

    print(f"Running inference on {len(ps)} audio samples...")
    outputs = llm.generate(ps, sampling_params=sampling_params)

    os.makedirs(args.output_dir, exist_ok=True)

    new_ckpt = args.ckpt.split('/')[-1] if '/' in args.ckpt else args.ckpt
    dataset_name = "LIVETALK" if "LIVETALK" in args.nisqa_csv_path else "FOR" if "FOR" in args.nisqa_csv_path else "P501" if "P501" in args.nisqa_csv_path else ""
    with open(f"{args.output_dir}/{new_ckpt}_{dataset_name}.jsonl", "w") as ans_file:
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
                "audio_path": dataset[i]['audio_path'] if i < len(dataset) else "",
            }) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate NISQA audio quality using vLLM')
    parser.add_argument('--root', type=str, default="/path/to/nisqa",
                        help='Root directory for NISQA dataset')
    parser.add_argument('--nisqa_csv_path', type=str, default="NISQA_TEST_LIVETALK/NISQA_TEST_LIVETALK_file.csv",
                        help='Path to NISQA metadata CSV file')
    parser.add_argument('--nisqa_dir', type=str, default="NISQA_TEST_LIVETALK",
                        help='Directory containing NISQA audio files')
    parser.add_argument('--output_dir', type=str, default="results/nisqa",
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
    args.nisqa_csv_path = os.path.join(args.root, args.nisqa_csv_path)
    args.nisqa_dir = os.path.join(args.root, args.nisqa_dir)
    main(args)