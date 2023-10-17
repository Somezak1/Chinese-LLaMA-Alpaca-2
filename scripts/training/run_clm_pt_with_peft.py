#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import numpy as np
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import datasets
import torch
from datasets import load_dataset, concatenate_datasets

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    BitsAndBytesConfig
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from sklearn.metrics import accuracy_score
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# 该代码的运行参数为
# torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
#     --deepspeed ds_zero2_no_offload.json \
#     --model_name_or_path /data1/csw_model_weights/Llama-2-13b-chat-hf \
#     --tokenizer_name_or_path /data1/csw_model_weights/chinese-llama-2-13b \
#     --dataset_dir ../../data \
#     --data_cache_dir temp_data_cache_dir \
#     --per_device_train_batch_size 2 \
#     --do_train \
#     --seed 14 \
#     --fp16 \
#     --num_train_epochs 1 \
#     --lr_scheduler_type cosine \
#     --learning_rate 2e-4 \
#     --warmup_ratio 0.05 \
#     --weight_decay 0.01 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --save_strategy steps \
#     --save_total_limit 3 \
#     --save_steps 500 \
#     --gradient_accumulation_steps 1 \
#     --preprocessing_num_workers 8 \
#     --block_size 512 \
#     --output_dir output_dir \
#     --overwrite_output_dir \
#     --ddp_timeout 30000 \
#     --logging_first_step True \
#     --lora_rank 64 \
#     --lora_alpha 128 \
#     --trainable "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
#     --modules_to_save "embed_tokens,lm_head" \
#     --lora_dropout 0.05 \
#     --torch_dtype float16 \
#     --load_in_kbits 16 \
#     --gradient_checkpointing \
#     --ddp_find_unused_parameters False

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "pt_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8/INT4 parameters to fp32
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


def accuracy(predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError: # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None)
    flash_attn : Optional[bool] = field(default=False)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)


logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.flash_attn:
        from flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)
    # def send_example_telemetry(example_name, *example_args, framework="pytorch"):
    #     """
    #     Sends telemetry that helps tracking the examples use.
    #
    #     Args:
    #         example_name (`str`): The name of the example.
    #         *example_args (dataclasses or `argparse.ArgumentParser`): The arguments to the script.
    #             This function will only try to extract the model and dataset name from those. Nothing else is tracked
    #         framework (`str`, *optional*, defaults to `"pytorch"`): The framework for the example.
    #     """

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)

    if training_args.should_log:
        # training_args.should_log: True
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    # log_level: 20, INFO
    # get_process_log_level():
    """
    Returns the log level to be used depending on whether this process is the main process of node 0, main process
    of node non-0, or a non-main process.

    For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn't do
    anything) unless overridden by `log_level` argument.

    For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
    argument.

    The choice between the main and replica process settings is made according to the return value of `should_log`.
    """
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    # Set the verbosity level for the 🤗 Transformers's root logger.
    transformers.utils.logging.set_verbosity(log_level)
    # Enable the default handler of the HuggingFace Transformers's root logger.
    transformers.utils.logging.enable_default_handler()
    # Enable explicit formatting for every HuggingFace Transformers's logger.
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    # training_args.output_dir: 'output'. The output directory where model predictions and checkpoints will be written.
    # training_args.do_train: True. Whether to run training.
    # training_args.overwrite_output_dir: True. If `True`, overwrite the content of the output directory.
    # Use this to continue training if `output_dir` points to a checkpoint directory.
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    # Helper function for reproducible behavior to set seed in `random`, `numpy`, `torch` and/or `tf` (if installed).
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        # model_args.cache_dir: None
        "revision": model_args.model_revision,
        # model_args.model_revision: 'main'
        "use_auth_token": True if model_args.use_auth_token else None,
        # model_args.use_auth_token: False
    }
    # config_kwargs: key/value pairs with which to update the configuration object after loading
    if model_args.config_name:
        # model_args.config_name: None
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        # model_args.model_name_or_path: '/data/model_weights/chinese-llama-plus-7b-official'
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        # model_args.cache_dir: None
        "use_fast": model_args.use_fast_tokenizer,
        # model_args.use_fast_tokenizer: True, # Indicate if transformers
        # Indicate if transformers should try to load the fast version of the tokenizer (True) or use the Python one (False).
        "revision": model_args.model_revision,
        # model_args.model_revision: 'main'
        "use_auth_token": True if model_args.use_auth_token else None,
        # model_args.use_auth_token: False
    }
    if model_args.tokenizer_name:
        # model_args.tokenizer_name: None
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        # model_args.tokenizer_name_or_path: '/data/model_weights/chinese-llama-plus-7b-official'
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.add_eos_token = True

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        # examples:
        # {
        #   'text': [
        #       '4. 把剩饭剩菜利用起来，做成其他菜肴。',
        #       '5. 把无法在食用前熟食的水果和蔬菜冷冻或保存。',
        #       '创建一个包含10个字段的表单，用于收集客户的订单信息。客户订单表格',
        #       ...共1000条
        #    ]
        # }

        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output


    if data_args.block_size is None:
        # data_args.block_size: 512
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            # tokenizer.model_max_length: 1000000000000000019884624838656
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
        # block_size: 512

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        # examples:
        # {
        #   'input_ids': [input_ids_random_len_list_1, ..., input_ids_random_len_list_1000],
        #   'attention_mask': [attention_mask_random_len_list_1, ..., attention_mask_random_len_list_1000]
        # }

        # tokenizer.convert_ids_to_tokens(input_ids_random_len_list_1):
        # ['<s>', '▁4.', '▁', '把', '剩', '饭', '剩', '菜', '利用', '起来', '，', '做成', '其他', '菜', '肴', '。']
        # tokenizer.convert_ids_to_tokens(input_ids_random_len_list_2):
        # ['<s>', '▁5.', '▁', '把', '无法', '在', '食用', '前', '熟', '食', '的', '水果', '和', '蔬菜', '冷', '冻', '或', '保存', '。']

        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # concatenated_examples:
        # {
        #   'input_ids': [input_ids_random_len_list_1+...+input_ids_random_len_list_1000],
        #   'attention_mask': [attention_mask_random_len_list_1+...+attention_mask_random_len_list_1000]
        # }

        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # result:
        # {
        #   'input_ids': [input_ids_512_len_list_1, ..., input_ids_512_len_list_N],
        #   'attention_mask': [attention_mask_512_len_list_1, ..., attention_mask_512_len_list_N]
        # }
        # N = total_length // block_size
        result["labels"] = result["input_ids"].copy()
        return result


    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        lm_datasets = []
        path = Path(data_args.dataset_dir)
        files = [file.name for file in path.glob("*.txt")]
        # files: ['pt_sample_data.txt']
        if training_args.debug_mode is True:
            # training_args.debug_mode: False
            files = [files[0]]

        for idx, file in enumerate(files):
            data_file = os.path.join(path, file)
            # data_file: '../../data/pt_sample_data.txt'
            filename = ''.join(file.split(".")[:-1])
            # filename: 'pt_sample_data'
            cache_path = os.path.join(data_args.data_cache_dir, filename+f"_{block_size}")
            # cache_path: 'temp_data_cache_dir/pt_sample_data_512'
            os.makedirs(cache_path, exist_ok=True)
            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'training datasets-{filename} has been loaded from disk')
            except Exception:
                cache_dir = os.path.join(data_args.data_cache_dir, filename+f"_text_{block_size}")
                # cache_dir: 'temp_data_cache_dir/pt_sample_data_text_512'
                os.makedirs(cache_dir, exist_ok=True)
                raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
                # raw_dataset:
                # DatasetDict({
                #     train: Dataset({
                #         features: ['text'],
                #         num_rows: 125987
                #     })
                # })

                # txt文件中一行就是如下一条样本
                # raw_dataset['train'][0]:
                # {'text': '我们如何在日常生活中减少用水？1. 使用节水装置，如节水淋浴喷头和水龙头。 '}
                # raw_dataset['train'][1]:
                # {'text': '2. 使用水箱或水桶收集家庭废水，例如洗碗和洗浴。 '}
                # raw_dataset['train'][2]:
                # {'text': '3. 在社区中提高节水意识。 '}

                logger.info(f"{file} has been loaded")
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    # 一次处理1000条
                    num_proc=data_args.preprocessing_num_workers,
                    # data_args.preprocessing_num_workers: 8
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names = {k: os.path.join(cache_dir, 'tokenized.arrow') for k in raw_dataset},
                    # {'train': 'temp_data_cache_dir/pt_sample_data_text_512/tokenized.arrow'}
                    desc="Running tokenizer on dataset",
                )
                # tokenized_dataset:
                # DatasetDict({
                #     train: Dataset({
                #         features: ['input_ids', 'attention_mask'],
                #         num_rows: 125987
                #     })
                # })

                # tokenized_dataset['train'][0]:
                # {
                # 'input_ids': [1, 35646, 32175, 30505, 45566, 32800, 41332, 30882, 32902, 29871, 32049, 31669,
                #               30716, 36353, 30214, 30847, 31669, 30716, 38135, 37230, 33252, 31584, 30503, 30716,
                #               38460, 30267, 29871],
                # 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                # }
                # tokenizer.convert_ids_to_tokens(tokenized_dataset['train'][0]["input_ids"]):
                # ['<s>', '▁我们', '如何', '在', '日常生活中', '减少', '用水', '？', '1.', '▁', '使用', '节', '水', '装置',
                #  '，', '如', '节', '水', '淋', '浴', '喷', '头', '和', '水', '龙头', '。', '▁']

                # tokenized_dataset['train'][1]:
                # {
                # 'input_ids': [1, 33549, 29871, 32049, 30716, 33453, 31391, 30716, 35407, 36277, 32525, 33463,
                #               30716, 30214, 33725, 32711, 35100, 30503, 32711, 37230, 30267, 29871],
                # 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                # }
                # tokenizer.convert_ids_to_tokens(tokenized_dataset['train'][1]["input_ids"])
                # ['<s>', '▁2.', '▁', '使用', '水', '箱', '或', '水', '桶', '收集', '家庭', '废', '水', '，', '例如',
                # '洗', '碗', '和', '洗', '浴', '。', '▁']

                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    # 一次处理1000条
                    num_proc=data_args.preprocessing_num_workers,
                    # data_args.preprocessing_num_workers: 8
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names = {k: os.path.join(cache_dir, 'grouped.arrow') for k in tokenized_dataset},
                    # {'train': 'temp_data_cache_dir/pt_sample_data_text/grouped.arrow'}
                    desc=f"Grouping texts in chunks of {block_size}",
                )
                # grouped_datasets:
                # DatasetDict({
                #     train: Dataset({
                #         features: ['input_ids', 'attention_mask', 'labels'],
                #         num_rows: 6913
                #     })
                # })

                # >>> tokenizer.decode(grouped_datasets['train'][0]["input_ids"])
                # '<s> 我们如何在日常生活中减少用水？1. 使用节水装置，如节水淋浴喷头和水龙头。 \
                #  <s> 2. 使用水箱或水桶收集家庭废水，例如洗碗和洗浴。 \
                #  <s> 3. 在社区中提高节水意识。 \
                #  <s> 4. 检查水管和灌溉系统的漏水情况，并及时修复它们。 \
                #  <s> 5. 洗澡时间缩短，使用低流量淋浴头节约用水。 \
                #  <s> 6. 收集雨水，用于园艺或其他非饮用目的。 \
                #  <s> 7. 刷牙或擦手时关掉水龙头。 \
                #  <s> 8. 减少浇水草坪的时间。 \
                #  <s> 9. 尽可能多地重复使用灰水（来自洗衣机、浴室水槽和淋浴的水）。 \
                #  <s> 10. 只购买能源效率高的洗碗机和洗衣机。\
                #  <s> 编辑文章，使其更吸引读者。自主机器人是计算机控制的机器，被编程执行特定任务而不需要任何人类输入。\
                #  自主机器人在各个行业中被越来越广泛地应用，从制造业到医疗保健再到安全。自主机器人是计算机控制的机器，\
                #  被编程执行特定任务而不需要任何人类输入，从而实现了新的效率、精确度和可靠性水平。\
                #  自主机器人在各个行业中被越来越广泛地应用，从制造业，它们可以使用精度和一致的质量组装复杂的组件，\
                #  到医疗保健，可以协助进行医疗测试和处理，再到安全，可以监控大面积地区，保障人们和财产的安全。\
                #  自主机器人还可以减少在危险或有害环境中的错误和增加安全，在工业流程的检查或维修期间等。\
                #  由于其多样性，自主机器人将彻底改变我们工作方式的方式，使任务变得更加简单、快速，最终更加愉悦。\
                #  <s> 政府可以采取哪些策略来减少空气污染？\
                #  1. 实施强制的车辆排放标准和基于激励的计划，以降低车辆的碳足迹。\
                #  <s> 2. 增加公共交通工具，减少公众对车辆的依赖。\
                #  <s> 3. 增加对空气污染的影响的认识，鼓励市民减少污染物的生成。\
                #  <s> 4. 投资于可再生能源的研究和开发，如太阳能和风能。\
                #  <s> 5. 在工厂和发电厂安装空气污染控制装置，例如洗涤器。\
                #  <s> 6. 对车辆和工厂使用清洁燃料。\
                #  <s> 7. 实施更好的城市规划和控制拓展。\
                #  <s>8. 改善农业效率，减少化肥和杀虫'

                # >>> tokenizer.decode(grouped_datasets['train'][1]["input_ids"])
                # ' 剂的使用。\
                # <s> 9. 种植更多的树木以减少空气污染。\
                # <s> 10. 减少木材、煤炭和生物质的燃烧。\
                # <s> 可再生能源的存在对环境有什么影响？可再生能源的存在可以帮助减少空气污染和温室气体排放，\
                # 因为它们几乎不会排放二氧化碳、二氧化硫等空气污染物。此外，使用可再生能源可以促进能源效率的进一步提高和能源利用的改善。\
                # 可再生能源也可以帮助减少对化石燃料的依赖，这不仅可以减少排放，而且还可以帮助减少全球气候变化的风险。\
                # 最后，可再生能源可以帮助保护自然资源，减少为了能源生产而开发更多土地和资源的需要。\
                # <s> 解释神经网络如何学习。神经网络是一种机器学习算法，它使用连接的节点集合来近似可以将输入变量映射到输出的函数。\
                # 为了学习神经网络的参数，计算机需要调整节点之间连接的权重，以便网络为给定输入产生正确的输出。这个调整过程称为学习，\
                # 通过比较网络产生的输出和期望的结果，然后使用优化算法来调整权重，使得网络输出逼近期望的结果。\
                # 这个过程在多个输入和期望的输出上重复进行多次迭代。最终，连接节点之间的权重将被调整，以便神经网络的输出与期望的结果相匹配，\
                # 学习过程将完成。<s> 给出一个机器学习算法的例子，并解释它是如何工作的。一个流行的机器学习算法的例子是支持向量机（SVM）。\
                # 它是一个用于分类和回归任务的监督学习算法。它通过在n维空间中绘制数据点，由空间中的决策边界或超平面进行分离。\
                # 该算法使用最大边距，这些边距尽可能远离两类数据点。这些边距有助于创建最优的决策超平面。然后，\
                # 算法通过考虑分类任务中发生的错误来调整决策超平面，并相应地修改超平面。<s>\
                # <s> 最终，支持向量机可以使用最优的决策超平面执行分类任务，预测数据点的类别。\
                # <s>描述推荐系统的工作原理推荐系统是一种信息过滤系统，它使用用户过去的行为或偏好来建议用户可能感兴趣的新项目。\
                # 该系统首先收集用户行为和偏好的数据，例如他们经常在线购买或查看哪些项目。\
                # 然后使用这些数据基于他们过去的行为为用户创建个性化的个人资料。通过分析相似用户购买或查看的其他项目，\
                # 系统可以向用户推荐量身定制的项目'

                # >>> grouped_datasets['train'][0]: ('input_ids'里的内容与'labels'一模一样)
                # {
                # 'input_ids': [1, 35646, 32175, 30505, 45566, 32800, 41332, 30882, 32902, 29871, 32049, 31669,
                #               30716, 36353, 30214, 30847, 31669, 30716, 38135, 37230, 33252, 31584, 30503, 30716,
                #               38460, 30267, 29871, 1, 33549, 29871, 32049, 30716, 33453, 31391, 30716, 35407, 36277,
                #               32525, 33463, 30716, 30214, 33725, 32711, 35100, 30503, 32711, 37230, 30267, 29871, 1,
                #               34134, 32061, 33348, 30275, 32338, 31669, 30716, 33939, 30267, 29871, 1, 35566, 29871,
                #               32752, 30716, 31624, 30503, 42755, 36573, 35149, 30716, 32263, 30214, 31666, 33385,
                #               36064, 33392, 30267, 29871, 1, 37231, 29871, 40132, 32036, 41576, 30214, 32049, 32079,
                #               35155, 38135, 37230, 31584, 41065, 41332, 30267, 29871, 1, 39243, 29871, 36277, 32400,
                #               30716, 30214, 33262, 32553, 33438, 42373, 31838, 39665, 34287, 30267, 29871, 1, 41111,
                #               29871, 33092, 32973, 31391, 35152, 30880, 30594, 31057, 32430, 30716, 38460, 30267,
                #               29871, 1, 35400, 29889, 29871, 32800, 38528, 30716, 31710, 36088, 33173, 30267, 29871,
                #               1, 35703, 29889, 29871, 39443, 30923, 30533, 36102, 32049, 33644, 30716, 30419, 32701,
                #               46167, 30330, 37230, 32434, 30716, 37258, 30503, 38135, 37230, 30210, 30716, 30409,
                #               30267, 29871, 1, 34363, 29889, 29871, 31557, 32856, 34745, 34323, 30528, 30210, 32711,
                #               35100, 31429, 30503, 46167, 30267, 1, 29871, 32024, 33335, 30214, 37954, 31100, 34064,
                #               36197, 30267, 35364, 36880, 30392, 33482, 32287, 30210, 35454, 30214, 31407, 38531,
                #               33106, 36444, 32885, 35161, 32054, 41114, 30832, 34485, 30267, 35364, 36880, 30505,
                #               34350, 32260, 30275, 31407, 31844, 30805, 31844, 33818, 30533, 32596, 30214, 31594,
                #               38254, 30780, 33885, 36836, 31733, 30780, 32225, 30267, 35364, 36880, 30392, 33482,
                #               32287, 30210, 35454, 30214, 31407, 38531, 33106, 36444, 32885, 35161, 32054, 41114,
                #               30832, 34485, 30214, 33241, 37721, 33077, 34323, 30330, 40263, 30898, 30503, 45161,
                #               32387, 30267, 35364, 36880, 30505, 34350, 32260, 30275, 31407, 31844, 30805, 31844,
                #               33818, 30533, 32596, 30214, 31594, 38254, 30214, 33392, 32003, 32049, 42186, 30503,
                #               34199, 40793, 31263, 31905, 38848, 46158, 30214, 30780, 33885, 36836, 30214, 32003,
                #               37319, 32019, 33885, 33628, 30503, 32391, 30214, 31733, 30780, 32225, 30214, 32003,
                #               36927, 30257, 32719, 32178, 30214, 33579, 32482, 30503, 35707, 30210, 32225, 30267,
                #               35364, 36880, 34402, 32800, 30505, 34222, 31391, 39401, 42087, 30210, 33910, 30503,
                #               32332, 32225, 30214, 30505, 32545, 35641, 30210, 32752, 31391, 36682, 32520, 31184,
                #               30267, 32196, 31149, 40893, 30952, 30214, 35364, 36880, 30998, 34785, 32733, 32007,
                #               32017, 32290, 33784, 30214, 30785, 32885, 44844, 30666, 32862, 30330, 33238, 30214,
                #               32469, 32744, 46384, 30267, 1, 29871, 32128, 32003, 33505, 33217, 34216, 30805, 32800,
                #               34458, 34377, 30882, 32902, 29871, 32879, 38008, 30210, 33675, 39716, 32404, 30503,
                #               34791, 38866, 30210, 32282, 30214, 30651, 33139, 33675, 30210, 36206, 31722, 34629,
                #               30267, 1, 33549, 29871, 32332, 33428, 32747, 33427, 30214, 32800, 35755, 30783, 33675,
                #               30210, 36895, 30267, 1, 34134, 29871, 32332, 30783, 34458, 34377, 33462, 30210, 33153,
                #               30214, 34955, 34000, 32800, 46069, 30210, 36501, 30267, 1, 35566, 29871, 32140, 30909,
                #               30682, 42093, 34745, 35570, 30503, 32548, 30214, 30847, 41390, 30503, 32037, 30815,
                #               30267, 1, 37231, 32061, 35530, 30503, 37440, 32958, 33455, 34458, 34377, 32287, 36353,
                #               30214, 33725, 44117, 30943, 30267, 1, 39243, 29871, 30783, 33675, 30503, 35530, 32049,
                #               35949, 38520, 30267, 1, 41111, 29871, 32879, 35779, 32192, 32839, 30503, 32287, 37981,
                #               30267, 1, 35400, 29889, 29871, 33670, 32987, 34323, 30214, 32800, 30705, 33613, 30503,
                #               32324, 33231],
                # 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                # 'labels': [1, 35646, 32175, 30505, 45566, 32800, 41332, 30882, 32902, 29871, 32049, 31669, 30716,
                #            36353, 30214, 30847, 31669, 30716, 38135, 37230, 33252, 31584, 30503, 30716, 38460, 30267,
                #            29871, 1, 33549, 29871, 32049, 30716, 33453, 31391, 30716, 35407, 36277, 32525, 33463,
                #            30716, 30214, 33725, 32711, 35100, 30503, 32711, 37230, 30267, 29871, 1, 34134, 32061,
                #            33348, 30275, 32338, 31669, 30716, 33939, 30267, 29871, 1, 35566, 29871, 32752, 30716,
                #            31624, 30503, 42755, 36573, 35149, 30716, 32263, 30214, 31666, 33385, 36064, 33392, 30267,
                #            29871, 1, 37231, 29871, 40132, 32036, 41576, 30214, 32049, 32079, 35155, 38135, 37230,
                #            31584, 41065, 41332, 30267, 29871, 1, 39243, 29871, 36277, 32400, 30716, 30214, 33262,
                #            32553, 33438, 42373, 31838, 39665, 34287, 30267, 29871, 1, 41111, 29871, 33092, 32973,
                #            31391, 35152, 30880, 30594, 31057, 32430, 30716, 38460, 30267, 29871, 1, 35400, 29889,
                #            29871, 32800, 38528, 30716, 31710, 36088, 33173, 30267, 29871, 1, 35703, 29889, 29871,
                #            39443, 30923, 30533, 36102, 32049, 33644, 30716, 30419, 32701, 46167, 30330, 37230, 32434,
                #            30716, 37258, 30503, 38135, 37230, 30210, 30716, 30409, 30267, 29871, 1, 34363, 29889,
                #            29871, 31557, 32856, 34745, 34323, 30528, 30210, 32711, 35100, 31429, 30503, 46167, 30267,
                #            1, 29871, 32024, 33335, 30214, 37954, 31100, 34064, 36197, 30267, 35364, 36880, 30392,
                #            33482, 32287, 30210, 35454, 30214, 31407, 38531, 33106, 36444, 32885, 35161, 32054, 41114,
                #            30832, 34485, 30267, 35364, 36880, 30505, 34350, 32260, 30275, 31407, 31844, 30805, 31844,
                #            33818, 30533, 32596, 30214, 31594, 38254, 30780, 33885, 36836, 31733, 30780, 32225, 30267,
                #            35364, 36880, 30392, 33482, 32287, 30210, 35454, 30214, 31407, 38531, 33106, 36444, 32885,
                #            35161, 32054, 41114, 30832, 34485, 30214, 33241, 37721, 33077, 34323, 30330, 40263, 30898,
                #            30503, 45161, 32387, 30267, 35364, 36880, 30505, 34350, 32260, 30275, 31407, 31844, 30805,
                #            31844, 33818, 30533, 32596, 30214, 31594, 38254, 30214, 33392, 32003, 32049, 42186, 30503,
                #            34199, 40793, 31263, 31905, 38848, 46158, 30214, 30780, 33885, 36836, 30214, 32003, 37319,
                #            32019, 33885, 33628, 30503, 32391, 30214, 31733, 30780, 32225, 30214, 32003, 36927, 30257,
                #            32719, 32178, 30214, 33579, 32482, 30503, 35707, 30210, 32225, 30267, 35364, 36880, 34402,
                #            32800, 30505, 34222, 31391, 39401, 42087, 30210, 33910, 30503, 32332, 32225, 30214, 30505,
                #            32545, 35641, 30210, 32752, 31391, 36682, 32520, 31184, 30267, 32196, 31149, 40893, 30952,
                #            30214, 35364, 36880, 30998, 34785, 32733, 32007, 32017, 32290, 33784, 30214, 30785, 32885,
                #            44844, 30666, 32862, 30330, 33238, 30214, 32469, 32744, 46384, 30267, 1, 29871, 32128,
                #            32003, 33505, 33217, 34216, 30805, 32800, 34458, 34377, 30882, 32902, 29871, 32879, 38008,
                #            30210, 33675, 39716, 32404, 30503, 34791, 38866, 30210, 32282, 30214, 30651, 33139, 33675,
                #            30210, 36206, 31722, 34629, 30267, 1, 33549, 29871, 32332, 33428, 32747, 33427, 30214,
                #            32800, 35755, 30783, 33675, 30210, 36895, 30267, 1, 34134, 29871, 32332, 30783, 34458,
                #            34377, 33462, 30210, 33153, 30214, 34955, 34000, 32800, 46069, 30210, 36501, 30267, 1,
                #            35566, 29871, 32140, 30909, 30682, 42093, 34745, 35570, 30503, 32548, 30214, 30847, 41390,
                #            30503, 32037, 30815, 30267, 1, 37231, 32061, 35530, 30503, 37440, 32958, 33455, 34458,
                #            34377, 32287, 36353, 30214, 33725, 44117, 30943, 30267, 1, 39243, 29871, 30783, 33675,
                #            30503, 35530, 32049, 35949, 38520, 30267, 1, 41111, 29871, 32879, 35779, 32192, 32839,
                #            30503, 32287, 37981, 30267, 1, 35400, 29889, 29871, 33670, 32987, 34323, 30214, 32800,
                #            30705, 33613, 30503, 32324, 33231]
                # }

                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
            if idx == 0:
                lm_datasets = processed_dataset['train']
            else:
                assert lm_datasets.features.type == processed_dataset["train"].features.type
                lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])
        lm_datasets = lm_datasets.train_test_split(test_size = data_args.validation_split_percentage)
        # data_args.validation_split_percentage: 0.001

    if training_args.do_train:
        # training_args.do_train: True
        train_dataset = lm_datasets['train']
        if data_args.max_train_samples is not None:
            # data_args.max_train_samples: None
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(train_dataset)}")
        # len(train_dataset): 6906
        logger.info("Training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    if training_args.do_eval:
        # training_args.do_eval: False
        eval_dataset = lm_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("Evaluation example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if training_args.load_in_kbits in [4, 8]:
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8
        if training_args.modules_to_save is not None:
            load_in_8bit_skip_modules = training_args.modules_to_save.split(',')
        else:
            load_in_8bit_skip_modules = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=6.0,
            load_in_8bit_skip_modules=load_in_8bit_skip_modules,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
        )
    else:
        load_in_4bit = False
        load_in_8bit = False
        quantization_config = None
    if quantization_config is not None:
        logger.info(f"quantization_config:{quantization_config.to_dict()}")
    if model_args.model_name_or_path:
        # model_args.model_name_or_path: '/data/model_weights/chinese-llama-plus-7b-official'
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        # torch_dtype: torch.float16
        device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            # model_args.cache_dir: None
            revision=model_args.model_revision,
            # model_args.model_revision: 'main'
            use_auth_token=True if model_args.use_auth_token else None,
            # model_args.use_auth_token: False
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    if training_args.load_in_kbits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model.config.use_cache = False
    model_vocab_size = model.get_output_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    if tokenizer_vocab_size != 55296:
        raise ValueError(f"The vocab size of tokenizer is {tokenizer_vocab_size}, not 55296. Please use Chinese-LLaMA-2 tokenizer.")
    if model_vocab_size != tokenizer_vocab_size:
        logger.info(f"Resize model vocab size to {tokenizer_vocab_size}")
        model.resize_token_embeddings(len(tokenizer))

    if training_args.peft_path is not None:
        # training_args.peft_path: None
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, training_args.peft_path, device_map=device_map)
    else:
        logger.info("Init new peft model")
        target_modules = training_args.trainable.split(',')
        # target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
        modules_to_save = training_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
            # modules_to_save: ['embed_tokens', 'lm_head']
        lora_rank = training_args.lora_rank
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            # TaskType.CAUSAL_LM: TaskType.CAUSAL_LM
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            # lora_rank: 8, lora_alpha: 32.0
            lora_dropout=lora_dropout,
            # lora_dropout: 0.05
            modules_to_save=modules_to_save)
            # modules_to_save: ['embed_tokens', 'lm_head']
        model = get_peft_model(model, peft_config)
    if training_args.gradient_checkpointing and \
        (not model.modules_to_save or 'embed_tokens' not in model.modules_to_save):
        # enable requires_grad to avoid exception during backward pass when using gradient_checkpoint without tuning embed.
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        elif hasattr(model.base_model, "get_input_embeddings"):
            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)
            model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            if training_args.fp16:
                module = module.to(torch.float16)
        if 'norm' in name:
            module = module.to(torch.float16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                if training_args.fp16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)
    model.print_trainable_parameters()
    logger.info(f"model.modules_to_save: {model.modules_to_save}")
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )
    trainer.add_callback(SavePeftModelCallback)
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
