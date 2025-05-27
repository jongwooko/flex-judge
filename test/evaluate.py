from datasets import load_dataset, Features, Value
import argparse
import json
import glob
import os
import numpy as np
import re
from pathlib import Path

def mllm_judge_eval(args):
    assert args.split in ["pair", "score", "batch"]

    if args.split == 'pair':
        features = Features({
            'id': Value(dtype='int64', id=None), 'pair_id': Value(dtype='int64', id=None), 
            'image_path': Value(dtype='string', id=None), 'original_dataset': Value(dtype='string', id=None), 
            'instruction': Value(dtype='string', id=None), 
            'answer1': {'name': Value(dtype='string', id=None), 'answer': Value(dtype='string', id=None)}, 
            'answer2': {'name': Value(dtype='string', id=None), 'answer': Value(dtype='string', id=None)}, 
            'human': Value(dtype='string', id=None), "true_answers": Value(dtype='string', id=None)
        })
        ds = load_dataset('json', data_files='data/mllm-judge/Benchmark/pair.jsonl', split='train', features=features)

    elif args.split == 'score':
        features = Features({
            'id': Value(dtype='int64', id=None), 'score_id': Value(dtype='int64', id=None), 
            'image_path': Value(dtype='string', id=None), 'original_dataset': Value(dtype='string', id=None), 
            'instruction': Value(dtype='string', id=None), 
            'answer': Value(dtype='string', id=None), 'name': Value(dtype='string', id=None),
            'human': Value(dtype='int64', id=None), "true_answers": Value(dtype='string', id=None)
        })
        ds = load_dataset('json', data_files='data/mllm-judge/Benchmark/score.jsonl', split='train', features=features)

    elif args.split == "batch":
        ds = load_dataset('json', data_files='data/mllm-judge/Benchmark/batch.jsonl', split='train')

    ds1 = load_dataset('json', data_files=f'results/mllm-judge/{args.model}_0_{args.split}.jsonl', split='train')
    ds2 = load_dataset('json', data_files=f'results/mllm-judge/{args.model}_1_{args.split}.jsonl', split='train')
    ds3 = load_dataset('json', data_files=f'results/mllm-judge/{args.model}_2_{args.split}.jsonl', split='train')


    if args.split == "pair":
        p2n = {"A": -1, "B": 1, "C": 0}
        d = {k: [0, 0] for k in list(set(ds['original_dataset']))}
        d2 = {k: [0, 0] for k in list(set(ds['original_dataset']))}

        for e1, e2, e3, l, o in zip(ds1["pred"], ds2["pred"], ds3["pred"], ds["human"], ds['original_dataset']):
            d[o][1] += 1; e = p2n[e1] + p2n[e2] + p2n[e3]
            if e < 0: e = "A"
            elif e > 0: e = "B"
            else: e = "C"
            
            if e == l: d[o][0] += 1
            if l != 'C':
                d2[o][1] += 1
                if e == l: d2[o][0] += 1

        v0, v1 = 0, 0
        results = []
        for k, v in d.items():
            results.append([k, round(v[0] / v[1], 3)])
            v0 += v[0]; v1 += v[1]

        print ("*** Pair w. Tie ***")
        ave = []
        for i in sorted(results, key=lambda x: x[0]):
            print (i[0] + ':', i[1])
            ave.append(i[1])
        print ("Ave:", round(sum(ave)/len(ave), 3))

        v0, v1 = 0, 0
        results = []
        for k, v in d2.items():
            results.append([k, round(v[0] / v[1], 3)])
            v0 += v[0]; v1 += v[1]

        print ()
        print ("*** Pair w.o Tie ***")
        ave = []
        for i in sorted(results, key=lambda x: x[0]):
            print (i[0] + ':', i[1])
            ave.append(i[1])
        print ("Ave:", round(sum(ave)/len(ave), 3))

    elif args.split == "score":
        from scipy.stats import pearsonr

        alls = {k: [] for k in list(set(ds['original_dataset']))}
        preds = {k: [] for k in list(set(ds['original_dataset']))}
        humans = {k: [] for k in list(set(ds['original_dataset']))}

        all = [ds1, ds2, ds3]

        for dsi in all:
            for t, o, h in zip(dsi['text'], ds['original_dataset'], ds['human']):
                humans[o].append(h)

                v = list(map(int, re.findall(r"<answer>(\d+)</answer>", t)))
                if len(v) == 0: preds[o].append(-1)
                else: preds[o].extend([v[0]])

            results = []
            for (k1, v1), (k2, v2) in zip(preds.items(), humans.items()):
                assert k1 == k2
                if np.isnan(pearsonr(v1, v2)[0]):
                    results.append([k1, 0])
                else:
                    results.append([k1, pearsonr(v1, v2)[0]])

            for i in sorted(results, key=lambda x: x[0]):
                alls[i[0]].append(i[1])

        ave = []
        for k, v in alls.items():
            maxv = np.median(v)
            print (k + ':', round(maxv, 3))
            ave.append(maxv)
        print ("Ave:", round(sum(ave)/len(ave), 3))

    elif args.split == "batch":
        from Levenshtein import distance
        import string

        alls = {k: [] for k in list(set(ds['original_dataset']))}
        results = {k: [] for k in list(set(ds['original_dataset']))}

        all = [ds1, ds2, ds3]

        for dsi in all:
            for t, o, h in zip(dsi['text'], ds['original_dataset'], ds['human']):
                v = list(map(int, re.findall(r"<answer>(\d+)</answer>", t)))
                labels = list(string.ascii_uppercase[:len(v)])
                sorted_labels = [label for _, label in sorted(zip(v, labels), reverse=True)]
                result = ''.join(sorted_labels)
                dist = distance(result, h) / max(len(result), len(h))
                results[o].append(dist)
            
            for k, v in results.items():
                alls[k].append(np.mean(v))

        ave = []
        for k, v in alls.items():
            minv = np.max(v)
            print (k + ':', round(minv, 3))
            ave.append(minv)
        print ("Ave:", round(sum(ave)/len(ave), 3))


def genai_bench_eval(args):
    assert args.split in ["image", "editing", "video"]

    if args.split == "image":
        dataset = load_dataset('TIGER-Lab/GenAI-Bench', 'image_generation', split='test_v1')
        preds = [
            load_dataset('json', data_files=f'results/genai_bench/{args.model}_image_{i}.jsonl', split='train') \
            for i in range(7)
        ]

    elif args.split == "editing":
        dataset = load_dataset('TIGER-Lab/GenAI-Bench', 'image_edition', split='test_v1')
        preds = [
            load_dataset('json', data_files=f'results/genai_bench/{args.model}_editing_{i}.jsonl', split='train') \
            for i in range(7)
        ]

    elif args.split == "video":
        dataset = load_dataset('TIGER-Lab/GenAI-Bench', 'video_generation', split='test_v1')
        preds = [
            load_dataset('json', data_files=f'results/genai_bench/{args.model}_video_{i}.jsonl', split='train') \
            for i in range(7)
        ]

    r = []
    pp = [[] for _ in range(len(preds[0]))]

    for pred in preds:
        i = 0
        for j, (v, text) in enumerate(zip(dataset['vote_type'], pred['text'])):
            try:
                score = re.findall(r"<answer>(\[\[.*?\]\])</answer>", text)[0]
                if score in ["[[A>B]]", "[[B<A]]"]:
                    p = 'leftvote'
                elif score in ["[[A<B]]", "[[B>A]]"]:
                    p = 'rightvote'
                elif score in ["[[A=B=Good]]"]:
                    p = 'tievote'
                else:
                    p = 'bothbad_vote'

            except:
                p = 'none'

            pp[j].append(p)
            if v == p:
                i += 1

        r.append(i / len(dataset))

    from collections import Counter

    i = 0
    for p, v in zip(pp, dataset['vote_type']):
        counter = Counter(p)
        vp, _ = counter.most_common(1)[0]
        
        if vp == v: i += 1

    print (sum(r) / len(r))
    print (i / len(pp))


def audio_eval(args):
    from scipy.stats import pearsonr, spearmanr
    assert args.split in ["nisqa", "bvcc", "somos", "voxsim"]

    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    if args.split == "nisqa":
        result_files = [
            os.path.join(args.result_dir, args.split, f'{model_name}_FOR.jsonl'),
            os.path.join(args.result_dir, args.split, f'{model_name}_LIVETALK.jsonl'),
            os.path.join(args.result_dir, args.split, f'{model_name}_P501.jsonl'),
        ]
    else:
        result_files = [
            os.path.join(args.result_dir, args.split, f'{model_name}.jsonl'),
        ]
    
    all_preds = []
    all_labels = []
    all_system_preds = {}
    all_system_labels = {}

    system_gt_data = {}
    if args.mode == "system":
        if args.split == "bvcc":
            system_gt_file = os.path.join(args.audio_root, 'bvcc/main/DATA/mydata_system.csv') 
        elif args.split == "somos":
            system_gt_file = os.path.join(args.audio_root, 'somos/training_files/split1/clean/test_system.csv')
        else:
            raise NotImplementedError("System GT file not defined for this dataset")
        
        try:
            with open(system_gt_file, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    if ',' in line:
                        system_id, mean_score = line.strip().split(',')
                        system_gt_data[system_id] = float(mean_score)
        except Exception as e:
            print(f"Error: Failed to load system GT file {system_gt_file}: {e}")
            return

    for result_file in result_files:
        with open(result_file, 'r') as f:
            dataset_preds = []
            dataset_labels = []
            system_preds = {}
            system_labels = {}

            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'pred' in item and 'gt' in item:
                        pred = item['pred']
                        gt = item['gt']
                        if isinstance(pred, (int, float)) and isinstance(gt, (int, float)):
                            dataset_preds.append(pred)
                            dataset_labels.append(gt)
                            if 'audio_path' in item:
                                audio_path = item['audio_path']
                                if args.split == 'bvcc':
                                    system_id_match = Path(audio_path).stem.split('-')[0]
                                elif args.split == 'somos':
                                    system_id_match = Path(audio_path).stem.split('_')[-1]
                                else:
                                    system_id_match = None
                                system_id = system_id_match
                                if system_id not in system_preds:
                                    system_preds[system_id] = []
                                    system_labels[system_id] = []
                                system_preds[system_id].append(pred)
                                system_labels[system_id].append(gt)

                except json.JSONDecodeError:
                    print(f"Warning: Error parsing line in {result_file}")
                    continue

            if len(dataset_preds) > 1 and len(dataset_labels) > 1:
                if args.mode == "utterance":
                    dataset_preds = np.array(dataset_preds)
                    dataset_labels = np.array(dataset_labels)

                    lcc = round(pearsonr(dataset_preds, dataset_labels)[0], 3)
                    srcc = round(spearmanr(dataset_preds, dataset_labels)[0], 3)
                    mse = round(np.mean((dataset_preds - dataset_labels) ** 2), 3)

                    print(f"\nUtterance-level results for {os.path.basename(result_file)}:")
                    print(f"  total utterances: {len(dataset_preds)}")
                    print(f"  LCC: {lcc}")
                    print(f"  SRCC: {srcc}")
                    print(f"  MSE: {mse}")

                elif args.mode == "system":
                    file_system_avg_preds = {}
                    for sys_id in system_preds:
                        file_system_avg_preds[sys_id] = np.mean(system_preds[sys_id])

                    system_pred_values = []
                    system_gt_values = []

                    print(f"\nSystem-level results for {os.path.basename(result_file)}:")
                    print(f"{'System ID':<15} {'Predicted MOS':<15} {'Ground Truth MOS':<15}")
                    print("-" * 45)

                    for sys_id in file_system_avg_preds:
                        if sys_id in system_gt_data:
                            pred_mos = file_system_avg_preds[sys_id]
                            gt_mos = system_gt_data[sys_id]
                            system_pred_values.append(pred_mos)
                            system_gt_values.append(gt_mos)
                            print(f"{sys_id:<15} {pred_mos:<15.3f} {gt_mos:<15.3f}")

                            if sys_id not in all_system_preds:
                                all_system_preds[sys_id] = pred_mos
                                all_system_labels[sys_id] = gt_mos
                            else:
                                all_system_preds[sys_id] = (all_system_preds[sys_id] + pred_mos) / 2
                        else:
                            print(f"Warning: System ID {sys_id} not found in GT data")

                    if len(system_pred_values) > 1:
                        lcc = round(pearsonr(system_pred_values, system_gt_values)[0], 3)
                        srcc = round(spearmanr(system_pred_values, system_gt_values)[0], 3)
                        mse = round(np.mean((np.array(system_pred_values) - np.array(system_gt_values)) ** 2), 3)

                        print("\nSummary:")
                        print(f"  total systems: {len(system_pred_values)}")
                        print(f"  LCC: {lcc}")
                        print(f"  SRCC: {srcc}")
                        print(f"  MSE: {mse}")
                    else:
                        print(f"\nWarning: Not enough matching systems for evaluation in {result_file}")

            all_preds.extend(dataset_preds)
            all_labels.extend(dataset_labels)

    # Calculate overall combined metrics
    if len(all_preds) > 1 and len(all_labels) > 1:
        if args.mode == "utterance":
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            lcc = round(pearsonr(all_preds, all_labels)[0], 3)
            srcc = round(spearmanr(all_preds, all_labels)[0], 3)
            mse = round(np.mean((all_preds - all_labels) ** 2), 3)

            print(f"\n=== Combined Utterance-level Results ===")
            print(f"Total utterances: {len(all_preds)}")
            print(f"LCC: {lcc}")
            print(f"SRCC: {srcc}")
            print(f"MSE: {mse}")
    else:
        print("Error: Not enough data for evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Omni-RM")
    parser.add_argument('--benchmark', type=str, default="mllm_judge",
                        choices=["mllm_judge", "mjbench", "vl_rewardbench", "genai_bench", "audio"], help='Name of benchmark')
    parser.add_argument('--model', type=str, default="ckpts/vl",
                        help='Name of evaluated model')
    parser.add_argument('--split', type=str, default="pair",
                        help='Specific split for some benchmark')
    parser.add_argument('--audio_root', type=str, default='/path/to/audio/dataset',
                        help='Root directory for audio files')
    parser.add_argument('--mode', type=str, default="utterance", choices=["utterance", "system"],
                        help='Audio evaluation mode: utterance-level or system-level')
    parser.add_argument('--result_dir', type=str, default="results", help="Directory for saved results")
    args = parser.parse_args()

    if args.benchmark == "mllm_judge":
        mllm_judge_eval(args)

    elif args.benchmark == "mjbench":
        # mj_bench_eval(args)
        raise NotImplementedError

    elif args.benchmark == "vl_rewardbench":
        # vl_rewardbench_eval(args)
        raise NotImplementedError

    elif args.benchmark == "genai_bench":
        genai_bench_eval(args)

    elif args.benchmark == "audio":
        audio_eval(args)

    else:
        raise NotImplementedError
