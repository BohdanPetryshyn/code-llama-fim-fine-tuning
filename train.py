# Adapted from: https://github.com/pacman100/LLM-Workshop/blob/0ba41561ce6ea16d3993069c03ec1dca3ab6769d/personal_copilot/training/train.py#L144

import gc
import os
import sys
from typing import Optional
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    replace_lora_weights_loftq,
)
from dataset import ConstantLengthDataset


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_qunatization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )
    use_loftq: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables LoftQ init for the LoRA adapters when using QLoRA."},
    )
    use_loftq_callback: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enables LoftQ callback comparing logits of base model to the ones from LoftQ init. Provides better init."
        },
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="smangrul/hug_stack",
        metadata={"help": "The preference dataset to use."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=4096)
    test_size: Optional[float] = field(default=0.1)
    fim_rate: Optional[float] = field(default=0.5)
    fim_spm_rate: Optional[float] = field(default=0.5)
    splits: Optional[str] = field(
        default="train",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def create_datasets(tokenizer, args, seed):
    dataset = load_dataset(args.dataset_name, split=args.splits)
    dataset = dataset.train_test_split(
        test_size=args.test_size, seed=seed, shuffle=True
    )
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.dataset_text_field)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=seed,
        shuffle=True,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=seed,
    )
    print(f"A sample of valid dataset: {next(iter(valid_dataset))}")
    return train_dataset, valid_dataset


def get_mae(x, y):
    return (x - y).abs().mean()


def get_mse(x, y):
    return torch.pow(x - y, 2).mean()


def error_report(x, y):
    mae = get_mae(x, y)
    mse = get_mse(x, y)
    print(f"Mean absolute error: {mae:>8.5f}\n" f"Mean squared error:  {mse:>8.5f}")


def loftq_init(model, tokenizer, train_dataset, max_seq_length, args):
    if args.use_loftq_callback:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=compute_dtype
        )
        base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        random_input_ids = (
            torch.randint(0, len(train_dataset), size=(1,)).numpy().tolist()
        )
        random_inputs = [train_dataset[i]["content"] for i in random_input_ids]
        random_inputs = tokenizer(
            random_inputs,
            return_tensors="pt",
            padding=True,
            truncation="max_length",
            max_length=max_seq_length,
        )
        logits_base = base_model(**random_inputs).logits
        del base_model
        gc.collect()

        def loftq_callback(model, module_name):
            """Callable to replace weights with LoFTQ if the mse is lower than the current best one."""
            global current_mse
            logits = model(**random_inputs).logits
            mse = get_mse(logits_base, logits)
            if mse < current_mse:
                current_mse = mse
                print(f"MSE improved for module {module_name}")
                return True
            print(f"MSE did not improve for module {module_name}")
            return False

        replace_lora_weights_loftq(model, callback=loftq_callback)
        logits_loftq_callback = model(**random_inputs).logits
        error_report(logits_base, logits_loftq_callback)
    else:
        replace_lora_weights_loftq(model)


def create_and_prepare_model(args, data_args, training_args):
    device_map = None
    bnb_config = None

    load_in_8bit = args.use_8bit_qunatization
    load_in_4bit = args.use_4bit_quantization

    if args.use_unsloth:
        from unsloth import FastLanguageModel

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    if args.use_4bit_quantization or args.use_8bit_qunatization:
        device_map = (
            int(os.environ.get("LOCAL_RANK", -1))
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else "auto"
        )  # {"": 0}

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=load_in_8bit,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        )

    if (
        (args.use_4bit_quantization or args.use_8bit_qunatization)
        and args.use_peft_lora
        and not args.use_unsloth
    ):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": model_args.use_reentrant},
        )

    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                args.lora_target_modules.split(",")
                if args.lora_target_modules != "all-linear"
                else args.lora_target_modules
            ),
        )
        model = get_peft_model(model, peft_config)
    elif args.use_peft_lora and args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=(
                args.lora_target_modules.split(",")
                if args.lora_target_modules != "all-linear"
                else args.lora_target_modules
            ),
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )
    return model


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # load the datasets
    train_dataset, eval_dataset = create_datasets(
        tokenizer, data_args, training_args.seed
    )
    train_dataset.start_iteration = 0

    model = create_and_prepare_model(model_args, data_args, training_args)
    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = (
        training_args.gradient_checkpointing and not model_args.use_unsloth
    )
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if model_args.use_peft_lora:
        trainer.model.print_trainable_parameters()

    # LoftQ initialization when using QLoRA
    if model_args.use_4bit_quantization and model_args.use_loftq:
        loftq_init(
            trainer.model,
            tokenizer,
            train_dataset,
            data_args.max_seq_length,
            model_args,
        )

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
