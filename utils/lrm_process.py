import json
import re
from datasets import load_dataset
from tqdm.auto import tqdm

dataset = load_dataset('json', data_files='data/judgelm/judgelm_train_100k.jsonl', split='train')
preds = [
    # load_dataset('json', data_files=f'judgelrm_7b_{i}.jsonl', split='train') for i in [2, 21, 79, 100]
    load_dataset('json', data_files=f'judgelrm_7b_{i}_0.1.jsonl', split='train') for i in [13, 79]
]

ds = {}
for pred in preds:
    for p, s in tqdm(zip(pred, dataset['score'])):
        p1 = list(map(int, re.findall(r"<answer>(\d+)</answer>", p['text'])))
        try:
            if s[0] == p1[0] and s[1] == p1[1]:
                if p['prompt'] in ds:
                    ds[p['prompt']].append([p['text'], len(p['tokens'])])
                else:
                    ds[p['prompt']] = [[p['text'], len(p['tokens'])]]
        except:
            continue

with open("judgelrm_7b_0.1_final.jsonl", "w") as ans_file:
    for k, v in ds.items():
        ans_file.write(json.dumps({
            "prompt": k, "text": sorted(v, key=lambda x: x[1], reverse=True)[0][0]
        }) + "\n")