import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
import torch


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-7B-VL-Instruct")
    block_size: int = field(default=8192)
    train_file_path: Optional[str] = field(default='./data/train.jsonl')
    dagger: bool = field(default=False)

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        if config.model_name in ["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"]:
            model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model_name, torch_dtype=torch.bfloat16
            )
        elif config.model_name in ["Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen2.5-Omni-3B"]:
            model = transformers.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                config.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            )
        elif config.model_name in ["meta-llama/Llama-3.1-8B-Instruct"]:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                config.model_name, torch_dtype=torch.bfloat16,
            )
    

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True, truncation=True, model_max_length=config.block_size)
    processor = transformers.AutoProcessor.from_pretrained(config.model_name, use_fast=True, truncation=True, model_max_length=config.block_size)

    try:
        dataset = load_dataset(config.train_file_path)
    except:
        dataset = load_dataset('json', data_files=config.train_file_path)
        lenx = [len(tokenizer(i)['input_ids']) for i in dataset['train']['text']]

        texts = []
        for example in dataset['train']:
            question = example['prompt']
            text = example['text']
            texts.append(question + text)
        dataset['train'] = dataset['train'].remove_columns("text")
        dataset['train'] = dataset['train'].add_column("text", texts)

    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"
    else:
        raise NotImplementedError

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    trainer.train()
    if config.model_name in ["Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen2.5-Omni-3B"]:
        try:
            model = transformers.Qwen2_5OmniForConditionalGeneration.from_pretrained(
                config.model_name, torch_dtype=torch.bfloat16, 
            )
            model.thinker = trainer.model
            model.save_pretrained(args.output_dir)
        except:
            trainer.save_model(output_dir=args.output_dir)

    else:
        trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    train()