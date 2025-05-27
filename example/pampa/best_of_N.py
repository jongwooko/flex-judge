import json
import re
import os
import random
import numpy as np
from collections import defaultdict

# Config
jsonl_dir = "./"  # Path where all judge_output_*.jsonl files are stored
file_prefix = "judge_output_default"
file_suffix = ".jsonl"
num_files = 16  # Adjust based on number of response variants per sample
N = 16

iters = 10
accs = []

def extract_score(text):
    match = re.search(r"<answer>(\d+(?:\.\d+)?)</answer>", text)
    if match:
        return float(match.group(1))
    return None

for i in range(iters):

    # Load all JSONL files
    uid_candidates = defaultdict(list)
    
    for idx in random.sample(range(num_files), N):
        path = os.path.join(jsonl_dir, f"{file_prefix}_{idx}{file_suffix}")
        with open(path, "r") as f:
            num_lines = sum(1 for _ in f)
        print(f"Number of rows: {num_lines}")
        with open(path, "r") as f:
            for l, line in enumerate(f):
                item = json.loads(line)
                score = extract_score(item.get("Judge", ""))
                if score is not None:
                    uid_candidates[l].append({
                        "score": score,
                        "prediction": item.get("Prediction"),
                        "label": item.get("Label")
                    })

    # Best-of-N selection and accuracy computation
    correct = 0
    total = 0

    for uid, candidates in uid_candidates.items():
        # Select the candidate(s) with the highest score
        max_score = max(c["score"] for c in candidates)
        best_candidates = [c for c in candidates if c["score"] == max_score]
        chosen = random.choice(best_candidates)

        if chosen["prediction"] == chosen["label"]:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Best-of-{num_files} Accuracy: {accuracy:.4f} ({correct}/{total})")

    accs.append(accuracy)

mean = np.mean(accs)
std = np.std(accs)
print(mean, std)