from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import os

from transformers import BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch

import re
import random
from multiprocessing import cpu_count

import determined as det
from determined.transformers import DetCallback

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs["model"].config.to_json_file(f"{checkpoint_folder}/config.json")
        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control

def main(training_args, det_callback, hparams):
    # based on config
    dataset_id = hparams["dataset"]
    raw_datasets = load_dataset(dataset_id)

    if hparams["dryrun"]: # enable this to debug
        indices = range(0,100) 

        dataset_dict = {"train": raw_datasets["train_sft"].select(indices),
                        "test": raw_datasets["test_sft"].select(indices)}
    else:
        dataset_dict = {"train": raw_datasets["train_sft"],
                        "test": raw_datasets["test_sft"]}

    raw_datasets = DatasetDict(dataset_dict)

    model_id = hparams["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    # Set chat template
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    def apply_chat_template(example, tokenizer):
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

        return example

    column_names = list(raw_datasets["train"].features)
    raw_datasets = raw_datasets.map(apply_chat_template,
                                    num_proc=cpu_count(),
                                    fn_kwargs={"tokenizer": tokenizer},
                                    remove_columns=column_names,
                                    desc="Applying chat template",)

    # create the splits
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    model_kwargs = dict(
        torch_dtype="auto",
        use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map=device_map,
    )

    ##### PEFT #####
    # based on config
    peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = SFTTrainer(
            model=model_id,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            tokenizer=tokenizer,
            packing=True,
            peft_config=peft_config,
            max_seq_length=tokenizer.model_max_length,
    )

    trainer.add_callback(det_callback)
    trainer.add_callback(SavePeftModelCallback)
    
    trainer.train()
    
    checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
    trainer.model.config.to_json_file(f"{checkpoint_folder}/config.json")
    # Save fine-tuned model
    trainer_filepath= f"trainer/{train_util.get_time()}"
    trainer.model.save_pretrained(os.path.join(checkpoint_folder, trainer_filepath))

    # reload base model
    base_model= AutoModelForCausalLM.from_pretrained(model_name)

    # merge base model and fine-tuned model
    merged_model= PeftModel.from_pretrained(base_model, os.path.join(checkpoint_folder, trainer_filepath))
    merged_model= merged_model.merge_and_unload()

    # save merged model
    merged_model.save_pretrained(checkpoint_folder)
    
    
if __name__ == "__main__":
    info = det.get_cluster_info()
    hparams = info.trial.hparams
    distributed = det.core.DistributedContext.from_torch_distributed()
    
    # path where the Trainer will save its checkpoints and logs
    # output_dir = "data/Mistral-7B-v0.1"
    
    with det.core.init(distributed=distributed) as core_context:
        training_args = TrainingArguments(**hparams["training_args"])
        det_callback = DetCallback(core_context, training_args)
        main(training_args, det_callback, hparams)