# Case Study: Flex-Mol-LLaMA

## 🗓️ To-Dos
- [x] Upload best-of-N evaluation code
- [x] Upload DPO training code
- [x] Release DPO-trained Mol-LLaMA checkpoint: [link](https://www.dropbox.com/scl/fi/q4mo6zzq806a6valh4bsn/dpo_mol_llama.ckpt?rlkey=ay0zfby4z9zpzndb56mjumi92&dl=0)

## Install Requirements

**Flex-Mol-LLaMA** is a molecular judge model built on top of [Mol-LLaMA](https://github.com/DongkiKim95/Mol-LLaMA). 

Please follow their repository for base installation and dataset preparation.
- [Dependencies](https://github.com/DongkiKim95/Mol-LLaMA?tab=readme-ov-file#dependencies)
- [Dataset Preparation](https://github.com/DongkiKim95/Mol-LLaMA?tab=readme-ov-file#dataset-preparation)

## Prepare the Flex-Judge Model

You need a Flex-Judge model based on [Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), which will be embedded into Flex-Mol-LLaMA later.
- Follow the [Flex-Judge training guide](../README.md#training-flex-judge)
- Place the final checkpoint under `./flex_llama_8b`

## Best-of-N Evaluation (PAMPA)

### Step 1: Generate N Candidate Responses
```bash
NUM_ITER=16  # N=16
for ((i=0; i<NUM_ITER; i++)); do
    python -m pampa.inference --prompt_type default  --exp_name $i 
done
```
The output responses will be recorded at `pampa/results_llama3_default_$i.jsonl`. You can also try other prompt types (i.e., `rationale` or `task_info`).

### Step 2: Score Responses with Flex-Mol-LLaMA
```sh
for ((i=0; i<NUM_ITER; i++)); do
    python -m pampa.inference_judge \
      --prompt_type default \
      --judge_llm_path ./flex_llama_8b \
      --judge_input_path ./pampa/results_llama3_default_$i.jsonl \
      --judge_output_path ./pampa/judge_output_default_$i.jsonl
done
```
This will integrate the reasoning backbone (`./flex_llama_8b`) with Mol-LLaMA's molecular modules including LoRA, Q-Formers, and molecule encoders.

### Step 3: Select Best-of-N Response
```sh
python pampa/best_of_N.py
```

## DPO Training

### Prepare Preference Data
Before DPO training, we need to setup the preference data. 
Your `data/` folder should look like this:
```
data/
├── pubchem-molecules.json
├── detailed_structural_descriptions.json
├── structure2biological_features_relationships.json
└── structure2chemical_features_relationships.json
```

- Run `generate.py` twice, one with temp 0.8 and one with temp 1.2
- Generate responses with all three datasets: structure, biological, and chemical. (Refer to `configs/stage2/`)
- Use `judge.py` to score the response pairs
- Filter high-quality pairs with consistent winning responses

> ⚠️ Processing over the entire datasets may take time. We used 4.3K samples, available at `data/judge_output.json`, which was sufficiently useful.

### Run DPO Training (stage3 Only)
We skip stage1 & stage2. Start from the pretrained model, [Mol-LLaMA](https://huggingface.co/DongkiKim/Mol-Llama-3.1-8B-Instruct), and further fine-tune on the DPO training set we have constructed.

Run:
```sh
python stage3.py
```
and evaluate the model checkpoint by:
```sh
python -m pampa.inference \
  --finetuned_ckpt ./checkpoints/stage3/v1.0/{last_epoch}.ckpt \ 
  --prompt_type default
```

We have released the DPO-trained Mol-LLaMA checkpoint at [this link](https://www.dropbox.com/scl/fi/q4mo6zzq806a6valh4bsn/dpo_mol_llama.ckpt?rlkey=ay0zfby4z9zpzndb56mjumi92&dl=0).
You may reproduce our results by evaluating with this model: `--finetuned_ckpt ./dpo_mol_llama.ckpt`.
