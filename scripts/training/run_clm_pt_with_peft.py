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

# 此代码的注释都是基于如下运行指令debug获得的
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
    # training_args.output_dir: 'output_dir'. The output directory where model predictions and checkpoints will be written.
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
        # model_args.model_name_or_path: '/data1/csw_model_weights/Llama-2-13b-chat-hf'
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
        # model_args.tokenizer_name_or_path: '/data1/csw_model_weights/chinese-llama-2-13b'
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.add_eos_token = True
    # tokenize时会在句子末尾加'</s>'标识符

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

        # input_ids_random_len_list_1:
        #     [1, 32553, 31381, 32084, 30505, 43547, 32740, 42037, 30882, 29896, 29889, 29871, 32059, 31669, 30716,
        #     37944, 30214, 30847, 31669, 30716, 40202, 37658, 33215, 31584, 30503, 30716, 38049, 30267, 29871, 2]
        # tokenizer.convert_ids_to_tokens(input_ids_random_len_list_1):
        #     ['<s>', '▁我', '们', '如何', '在', '日常生活中', '减少', '用水', '？', '1', '.', '▁', '使用', '节', '水',
        #     '装置', '，', '如', '节', '水', '淋', '浴', '喷', '头', '和', '水', '龙头', '。', '▁', '</s>']

        # input_ids_random_len_list_2:
        #     [1, 29871, 29906, 29889, 29871, 32059, 30716, 33805, 31391, 30716, 35895, 36039, 32381, 34600, 30716,
        #     30214, 33231, 49926, 30503, 32645, 37658, 30267, 29871, 2]
        # tokenizer.convert_ids_to_tokens(input_ids_random_len_list_2):
        #     ['<s>', '▁', '2', '.', '▁', '使用', '水', '箱', '或', '水', '桶', '收集', '家庭', '废', '水', '，', '例如',
        #     '洗碗', '和', '洗', '浴', '。', '▁', '</s>']

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
        # def main_process_first(self, local=True, desc="work"):
        #     """
        #     A context manager for torch distributed environment where on needs to do something on the main process,
        #     while blocking replicas, and when it's finished releasing the replicas.
        #
        #     One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main
        #     process, which upon completion saves a cached version of results and which then automatically gets
        #     loaded by the replicas.
        #
        #     Args:
        #         local (`bool`, *optional*, defaults to `True`):
        #             if `True` first means process of rank 0 of each node if `False` first means process of rank 0
        #             of node rank 0 In multi-node environment with a shared filesystem you most likely will want
        #             to use `local=False` so that only the main process of the first node will do the processing.
        #             If however, the filesystem is not shared, then the main process of each node will need to do
        #             the processing, which is the default behavior.
        #         desc (`str`, *optional*, defaults to `"work"`):
        #             a work description to be used in debug logs
        #
        #     """

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
                # 'input_ids':
                #     [1, 32553, 31381, 32084, 30505, 43547, 32740, 42037, 30882, 29896, 29889, 29871, 32059, 31669,
                #     30716, 37944, 30214, 30847, 31669, 30716, 40202, 37658, 33215, 31584, 30503, 30716, 38049,
                #     30267, 29871, 2],
                # 'attention_mask':
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                # }
                # tokenizer.convert_ids_to_tokens(tokenized_dataset['train'][0]["input_ids"]):
                #     ['<s>', '▁我', '们', '如何', '在', '日常生活中', '减少', '用水', '？', '1', '.', '▁', '使用', '节',
                #     '水', '装置', '，', '如', '节', '水', '淋', '浴', '喷', '头', '和', '水', '龙头', '。', '▁', '</s>']

                # tokenized_dataset['train'][1]:
                # {
                # 'input_ids':
                #     [1, 29871, 29906, 29889, 29871, 32059, 30716, 33805, 31391, 30716, 35895, 36039, 32381, 34600,
                #     30716, 30214, 33231, 49926, 30503, 32645, 37658, 30267, 29871, 2],
                # 'attention_mask':
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                # }
                # tokenizer.convert_ids_to_tokens(tokenized_dataset['train'][1]["input_ids"])
                #     ['<s>', '▁', '2', '.', '▁', '使用', '水', '箱', '或', '水', '桶', '收集', '家庭', '废', '水', '，',
                #     '例如', '洗碗', '和', '洗', '浴', '。', '▁', '</s>']

                # 值得注意的是, 对于pt_sample_data.txt中
                # 第8行  line = '9. 尽可能多地重复使用灰水（来自洗衣机、浴室水槽和淋浴的水）。 \n'
                # tokenizer.decode(tokenized_dataset['train'][8]["input_ids"]):
                #     '<s> 9. 尽可能多地重复使用灰水（来自洗衣机、浴室水槽和淋浴的水）。 </s>'
                # tokenizer.convert_ids_to_tokens(tokenized_dataset['train'][8]["input_ids"]):
                #     ['<s>', '▁', '9', '.', '▁', '尽可能', '多', '地', '重复', '使用', '灰', '水', '（', '来自',
                #     '洗衣机', '、', '浴室', '水', '槽', '和', '淋', '浴', '的', '水', '）', '。', '▁', '</s>']
                # 第40行 line = ' \n'
                # tokenizer.decode(tokenized_dataset['train'][40]["input_ids"]):
                #     '<s>  </s>'
                # tokenizer.convert_ids_to_tokens(tokenized_dataset['train'][40]["input_ids"]):
                #     ['<s>', '▁▁', '</s>']
                # 第42行 line = '    players = int(sys.argv[1])\n'
                # tokenizer.decode(tokenized_dataset['train'][42]["input_ids"]):
                #     '<s>     players = int(sys.argv[1])</s>'
                # tokenizer.convert_ids_to_tokens(tokenized_dataset['train'][42]["input_ids"]):
                #     ['<s>', '▁▁▁▁', '▁players', '▁=', '▁int', '(', 'sys', '.', 'argv', '[', '1', '])', '</s>']
                # 第44行 line = '\n'
                # tokenizer.decode(tokenized_dataset['train'][44]["input_ids"]):
                #     '<s></s>'
                # tokenizer.convert_ids_to_tokens(tokenized_dataset['train'][44]["input_ids"]):
                #     ['<s>', '</s>']

                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    # 一次处理1000条
                    num_proc=data_args.preprocessing_num_workers,
                    # data_args.preprocessing_num_workers: 8
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names = {k: os.path.join(cache_dir, 'grouped.arrow') for k in tokenized_dataset},
                    # {'train': 'temp_data_cache_dir/pt_sample_data_text_512/grouped.arrow'}
                    desc=f"Grouping texts in chunks of {block_size}",
                )
                # grouped_datasets:
                # DatasetDict({
                #     train: Dataset({
                #         features: ['input_ids', 'attention_mask', 'labels'],
                #         num_rows: 7261
                #     })
                # })

                # >>> tokenizer.decode(grouped_datasets['train'][0]["input_ids"])
                # '<s> 我们如何在日常生活中减少用水？1. 使用节水装置，如节水淋浴喷头和水龙头。 </s>\
                # <s> 2. 使用水箱或水桶收集家庭废水，例如洗碗和洗浴。 </s>\
                # <s> 3. 在社区中提高节水意识。 </s>\
                # <s> 4. 检查水管和灌溉系统的漏水情况，并及时修复它们。 </s>\
                # <s> 5. 洗澡时间缩短，使用低流量淋浴头节约用水。 </s>\
                # <s> 6. 收集雨水，用于园艺或其他非饮用目的。 </s>\
                # <s> 7. 刷牙或擦手时关掉水龙头。 </s>\
                # <s> 8. 减少浇水草坪的时间。 </s>\
                # <s> 9. 尽可能多地重复使用灰水（来自洗衣机、浴室水槽和淋浴的水）。 </s>\
                # <s> 10. 只购买能源效率高的洗碗机和洗衣机。</s>\
                # <s> 编辑文章，使其更吸引读者。自主机器人是计算机控制的机器，被编程执行特定任务而不需要任何人类输入。\
                # 自主机器人在各个行业中被越来越广泛地应用，从制造业到医疗保健再到安全。自主机器人是计算机控制的机器，\
                # 被编程执行特定任务而不需要任何人类输入，从而实现了新的效率、精确度和可靠性水平。\
                # 自主机器人在各个行业中被越来越广泛地应用，从制造业，它们可以使用精度和一致的质量组装复杂的组件，到医疗保健，\
                # 可以协助进行医疗测试和处理，再到安全，可以监控大面积地区，保障人们和财产的安全。\
                # 自主机器人还可以减少在危险或有害环境中的错误和增加安全，在工业流程的检查或维修期间等。由于其多样性，\
                # 自主机器人将彻底改变我们工作方式的方式，使任务变得更加简单、快速，最终更加愉悦。</s>\
                # <s> 政府可以采取哪些策略来减少空气污染？1. 实施强制的车辆排放标准和基于激励的计划，以降低车辆的碳足迹。</s>\
                # <s> 2. 增加公共交通工具，减少公众对车辆的依赖。</s>\
                # <s> 3. 增加对空气污染的影响的认识，鼓励市民减少污染物的生成。</s>\
                # <s> 4. 投资于可再生能源的研究和开发，如太阳能和风能。</s>\
                # <s> 5. 在工厂和发电厂安装空气污染控制装置，例如洗涤器。</s>\
                # <s>6'

                # >>> tokenizer.decode(grouped_datasets['train'][1]["input_ids"])
                # ' . 对车辆和工厂使用清洁燃料。</s>\
                # <s> 7. 实施更好的城市规划和控制拓展。</s>\
                # <s> 8. 改善农业效率，减少化肥和杀虫剂的使用。</s>\
                # <s> 9. 种植更多的树木以减少空气污染。</s>\
                # <s> 10. 减少木材、煤炭和生物质的燃烧。</s>\
                # <s> 可再生能源的存在对环境有什么影响？可再生能源的存在可以帮助减少空气污染和温室气体排放，因为它们几乎不会排放二氧化碳、\
                # 二氧化硫等空气污染物。此外，使用可再生能源可以促进能源效率的进一步提高和能源利用的改善。\
                # 可再生能源也可以帮助减少对化石燃料的依赖，这不仅可以减少排放，而且还可以帮助减少全球气候变化的风险。\
                # 最后，可再生能源可以帮助保护自然资源，减少为了能源生产而开发更多土地和资源的需要。</s>\
                # <s> 解释神经网络如何学习。神经网络是一种机器学习算法，它使用连接的节点集合来近似可以将输入变量映射到输出的函数。\
                # 为了学习神经网络的参数，计算机需要调整节点之间连接的权重，以便网络为给定输入产生正确的输出。这个调整过程称为学习，\
                # 通过比较网络产生的输出和期望的结果，然后使用优化算法来调整权重，使得网络输出逼近期望的结果。\
                # 这个过程在多个输入和期望的输出上重复进行多次迭代。最终，连接节点之间的权重将被调整，以便神经网络的输出与期望的结果相匹配，\
                # 学习过程将完成。</s>\
                # <s> 给出一个机器学习算法的例子，并解释它是如何工作的。一个流行的机器学习算法的例子是支持向量机（SVM）。\
                # 它是一个用于分类和回归任务的监督学习算法。它通过在n维空间中绘制数据点，由空间中的决策边界或超平面进行分离。\
                # 该算法使用最大边距，这些边距尽可能远离两类数据点。这些边距有助于创建最优的决策超平面。然后，\
                # 算法通过考虑分类任务中发生的错误来调整决策超平面，并相应地修改超平面。</s>\
                # <s></s>\
                # <s> 最终，支持向量机可以使用最优的决策超平面执行分类任务，预测数据点的类别。</s>\
                # <s>描述推荐系统的工作原理推荐系统是一种信息过滤系统，它使用用户过去的行为或偏好来建议用户可能感兴趣的新项目。\
                # 该系统首先收集用户行为和偏好的数据，例如他们经常在线购买或查看哪些项目。'

                # >>> grouped_datasets['train'][0]: ('input_ids'里的内容与'labels'一模一样)
                # {
                # 'input_ids':
                #     [1, 32553, 31381, 32084, 30505, 43547, 32740, 42037, 30882, 29896, 29889, 29871, 32059,
                # 31669, 30716, 37944, 30214, 30847, 31669, 30716, 40202, 37658, 33215, 31584, 30503, 30716, 38049,
                # 30267, 29871, 2, 1, 29871, 29906, 29889, 29871, 32059, 30716, 33805, 31391, 30716, 35895, 36039,
                # 32381, 34600, 30716, 30214, 33231, 49926, 30503, 32645, 37658, 30267, 29871, 2, 1, 29871, 29941,
                # 29889, 29871, 30505, 33128, 30275, 32342, 31669, 30716, 33815, 30267, 29871, 2, 1, 29871, 29946,
                # 29889, 29871, 32520, 30716, 31624, 30503, 50117, 37132, 34948, 30716, 32195, 30214, 31666, 33040,
                # 35490, 33409, 30267, 29871, 2, 1, 29871, 29945, 29889, 29871, 37610, 32037, 42018, 30214, 32059,
                # 32147, 35044, 40202, 37658, 31584, 40329, 42037, 30267, 29871, 2, 1, 29871, 29953, 29889, 29871,
                # 36039, 44474, 30214, 33727, 32780, 33431, 43187, 31838, 39383, 34269, 30267, 29871, 2, 1, 29871,
                # 29955, 29889, 29871, 44911, 31391, 34387, 30880, 30594, 31057, 32351, 30716, 38049, 30267, 29871,
                # 2, 1, 29871, 29947, 29889, 29871, 32740, 39735, 30716, 49825, 32881, 30267, 29871, 2, 1, 29871,
                # 29929, 29889, 29871, 38823, 30923, 30533, 35322, 32059, 33987, 30716, 30419, 32501, 42925, 30330,
                # 46508, 30716, 37591, 30503, 40202, 37658, 30210, 30716, 30409, 30267, 29871, 2, 1, 29871, 29896,
                # 29900, 29889, 29871, 31557, 32513, 34685, 34046, 30528, 30210, 49926, 31429, 30503, 42925, 30267,
                # 2, 1, 29871, 32155, 32949, 30214, 39560, 31100, 33573, 36114, 30267, 36348, 35796, 30392, 33893,
                # 32357, 30210, 35077, 30214, 31407, 36480, 33136, 37203, 32739, 35892, 32042, 39647, 30832, 34168,
                # 30267, 36348, 35796, 30505, 34318, 32178, 30275, 31407, 31844, 30805, 31844, 34455, 30533, 32754,
                # 30214, 31594, 37879, 30780, 33521, 37898, 41496, 32162, 30267, 36348, 35796, 30392, 33893, 32357,
                # 30210, 35077, 30214, 31407, 36480, 33136, 37203, 32739, 35892, 32042, 39647, 30832, 34168, 30214,
                # 33477, 37647, 33179, 34046, 30330, 40752, 30898, 30503, 48001, 32316, 30267, 36348, 35796, 30505,
                # 34318, 32178, 30275, 31407, 31844, 30805, 31844, 34455, 30533, 32754, 30214, 31594, 37879, 30214,
                # 33409, 32003, 32059, 44813, 30503, 34360, 40765, 47125, 39538, 47181, 30214, 30780, 33521, 37898,
                # 30214, 32003, 38067, 32034, 33521, 33275, 30503, 32382, 30214, 41496, 32162, 30214, 32003, 35860,
                # 43781, 32326, 30214, 33343, 32568, 30503, 35703, 30210, 32162, 30267, 36348, 35796, 33663, 32740,
                # 30505, 34136, 31391, 40594, 42968, 30210, 33635, 30503, 32317, 32162, 30214, 30505, 32978, 34837,
                # 30210, 32520, 31391, 35811, 32439, 31184, 30267, 32150, 31149, 39683, 30952, 30214, 36348, 35796,
                # 30998, 34341, 32551, 32005, 32021, 32309, 33468, 30214, 30785, 32739, 46133, 32583, 30330, 33026,
                # 30214, 32408, 32505, 44258, 30267, 2, 1, 29871, 32219, 32003, 33535, 32796, 34028, 30805, 32740,
                # 34346, 34365, 30882, 29896, 29889, 29871, 32995, 37003, 30210, 33087, 39581, 32407, 30503, 34503,
                # 38575, 30210, 32277, 30214, 30651, 33112, 33087, 30210, 37504, 31722, 35337, 30267, 2, 1, 29871,
                # 29906, 29889, 29871, 32317, 48419, 33223, 30214, 32740, 35217, 30783, 33087, 30210, 36429, 30267,
                # 2, 1, 29871, 29941, 29889, 29871, 32317, 30783, 34346, 34365, 33505, 30210, 33121, 30214, 34454,
                # 33322, 32740, 47768, 30210, 37656, 30267, 2, 1, 29871, 29946, 29889, 29871, 32125, 30909, 30682,
                # 43311, 34685, 37550, 30503, 32586, 30214, 30847, 42348, 30503, 32077, 30815, 30267, 2, 1, 29871,
                # 29945, 29889, 29871, 30505, 35182, 30503, 38740, 33075, 33116, 34346, 34365, 32357, 37944, 30214,
                # 33231, 46507, 30943, 30267, 2, 1, 29871, 29953],
                # 'attention_mask':
                #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                # 'labels':
                #     [1, 32553, 31381, 32084, 30505, 43547, 32740, 42037, 30882, 29896, 29889, 29871, 32059,
                # 31669, 30716, 37944, 30214, 30847, 31669, 30716, 40202, 37658, 33215, 31584, 30503, 30716, 38049,
                # 30267, 29871, 2, 1, 29871, 29906, 29889, 29871, 32059, 30716, 33805, 31391, 30716, 35895, 36039,
                # 32381, 34600, 30716, 30214, 33231, 49926, 30503, 32645, 37658, 30267, 29871, 2, 1, 29871, 29941,
                # 29889, 29871, 30505, 33128, 30275, 32342, 31669, 30716, 33815, 30267, 29871, 2, 1, 29871, 29946,
                # 29889, 29871, 32520, 30716, 31624, 30503, 50117, 37132, 34948, 30716, 32195, 30214, 31666, 33040,
                # 35490, 33409, 30267, 29871, 2, 1, 29871, 29945, 29889, 29871, 37610, 32037, 42018, 30214, 32059,
                # 32147, 35044, 40202, 37658, 31584, 40329, 42037, 30267, 29871, 2, 1, 29871, 29953, 29889, 29871,
                # 36039, 44474, 30214, 33727, 32780, 33431, 43187, 31838, 39383, 34269, 30267, 29871, 2, 1, 29871,
                # 29955, 29889, 29871, 44911, 31391, 34387, 30880, 30594, 31057, 32351, 30716, 38049, 30267, 29871,
                # 2, 1, 29871, 29947, 29889, 29871, 32740, 39735, 30716, 49825, 32881, 30267, 29871, 2, 1, 29871,
                # 29929, 29889, 29871, 38823, 30923, 30533, 35322, 32059, 33987, 30716, 30419, 32501, 42925, 30330,
                # 46508, 30716, 37591, 30503, 40202, 37658, 30210, 30716, 30409, 30267, 29871, 2, 1, 29871, 29896,
                # 29900, 29889, 29871, 31557, 32513, 34685, 34046, 30528, 30210, 49926, 31429, 30503, 42925, 30267,
                # 2, 1, 29871, 32155, 32949, 30214, 39560, 31100, 33573, 36114, 30267, 36348, 35796, 30392, 33893,
                # 32357, 30210, 35077, 30214, 31407, 36480, 33136, 37203, 32739, 35892, 32042, 39647, 30832, 34168,
                # 30267, 36348, 35796, 30505, 34318, 32178, 30275, 31407, 31844, 30805, 31844, 34455, 30533, 32754,
                # 30214, 31594, 37879, 30780, 33521, 37898, 41496, 32162, 30267, 36348, 35796, 30392, 33893, 32357,
                # 30210, 35077, 30214, 31407, 36480, 33136, 37203, 32739, 35892, 32042, 39647, 30832, 34168, 30214,
                # 33477, 37647, 33179, 34046, 30330, 40752, 30898, 30503, 48001, 32316, 30267, 36348, 35796, 30505,
                # 34318, 32178, 30275, 31407, 31844, 30805, 31844, 34455, 30533, 32754, 30214, 31594, 37879, 30214,
                # 33409, 32003, 32059, 44813, 30503, 34360, 40765, 47125, 39538, 47181, 30214, 30780, 33521, 37898,
                # 30214, 32003, 38067, 32034, 33521, 33275, 30503, 32382, 30214, 41496, 32162, 30214, 32003, 35860,
                # 43781, 32326, 30214, 33343, 32568, 30503, 35703, 30210, 32162, 30267, 36348, 35796, 33663, 32740,
                # 30505, 34136, 31391, 40594, 42968, 30210, 33635, 30503, 32317, 32162, 30214, 30505, 32978, 34837,
                # 30210, 32520, 31391, 35811, 32439, 31184, 30267, 32150, 31149, 39683, 30952, 30214, 36348, 35796,
                # 30998, 34341, 32551, 32005, 32021, 32309, 33468, 30214, 30785, 32739, 46133, 32583, 30330, 33026,
                # 30214, 32408, 32505, 44258, 30267, 2, 1, 29871, 32219, 32003, 33535, 32796, 34028, 30805, 32740,
                # 34346, 34365, 30882, 29896, 29889, 29871, 32995, 37003, 30210, 33087, 39581, 32407, 30503, 34503,
                # 38575, 30210, 32277, 30214, 30651, 33112, 33087, 30210, 37504, 31722, 35337, 30267, 2, 1, 29871,
                # 29906, 29889, 29871, 32317, 48419, 33223, 30214, 32740, 35217, 30783, 33087, 30210, 36429, 30267,
                # 2, 1, 29871, 29941, 29889, 29871, 32317, 30783, 34346, 34365, 33505, 30210, 33121, 30214, 34454,
                # 33322, 32740, 47768, 30210, 37656, 30267, 2, 1, 29871, 29946, 29889, 29871, 32125, 30909, 30682,
                # 43311, 34685, 37550, 30503, 32586, 30214, 30847, 42348, 30503, 32077, 30815, 30267, 2, 1, 29871,
                # 29945, 29889, 29871, 30505, 35182, 30503, 38740, 33075, 33116, 34346, 34365, 32357, 37944, 30214,
                # 33231, 46507, 30943, 30267, 2, 1, 29871, 29953]
                # }

                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
            if idx == 0:
                lm_datasets = processed_dataset['train']
            else:
                assert lm_datasets.features.type == processed_dataset["train"].features.type
                lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])
        lm_datasets = lm_datasets.train_test_split(test_size = data_args.validation_split_percentage)
        # data_args.validation_split_percentage: 0.05

    if training_args.do_train:
        # training_args.do_train: True
        train_dataset = lm_datasets['train']
        if data_args.max_train_samples is not None:
            # data_args.max_train_samples: None
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(train_dataset)}")
        # len(train_dataset): 6897
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
        # model_args.model_name_or_path: '/data1/csw_model_weights/Llama-2-13b-chat-hf'
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
            # TaskType.CAUSAL_LM: <TaskType.CAUSAL_LM: 'CAUSAL_LM'>
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            # lora_rank: 64, lora_alpha: 128.0
            lora_dropout=lora_dropout,
            # lora_dropout: 0.05
            modules_to_save=modules_to_save)
            # modules_to_save: ['embed_tokens', 'lm_head']

        # class LoraConfig(PeftConfig):
        #     """
        #     This is the configuration class to store the configuration of a [`LoraModel`].
        #
        #     Args:
        #         r (`int`): Lora attention dimension.
        #         target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        #         lora_alpha (`float`): The alpha parameter for Lora scaling.
        #         lora_dropout (`float`): The dropout probability for Lora layers.
        #         fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        #         For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        #         bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        #         modules_to_save (`List[str]`): List of modules apart from LoRA layers to be set as trainable
        #             and saved in the final checkpoint.
        #     """

        model = get_peft_model(model, peft_config)
        # def get_peft_model(model, peft_config):
        #     """
        #     Returns a Peft model object from a model and a config.
        #
        #     Args:
        #         model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        #         peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
        #     """

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
    # def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    #     """
    #     Get the state dict of the Peft model.
    #
    #     Args:
    #         model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
    #         the model should be the underlying model/unwrapped model (i.e. model.module).
    #         state_dict (`dict`, *optional*, defaults to `None`):
    #             The state dict of the model. If not provided, the state dict of the model will be used.
    #     """

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
        # class LlamaForCausalLM(LlamaPreTrainedModel):
        #     _tied_weights_keys = ["lm_head.weight"]
        #
        #     def __init__(self, config):
        #         super().__init__(config)
        #         self.model = LlamaModel(config)
        #         self.vocab_size = config.vocab_size
        #         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #
        #         # Initialize weights and apply final processing
        #         self.post_init()
        #
        #     ......
        #
        #     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
        #     @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
        #     def forward(
        #         self,
        #         input_ids: torch.LongTensor = None,
        #         attention_mask: Optional[torch.Tensor] = None,
        #         position_ids: Optional[torch.LongTensor] = None,
        #         past_key_values: Optional[List[torch.FloatTensor]] = None,
        #         inputs_embeds: Optional[torch.FloatTensor] = None,
        #         labels: Optional[torch.LongTensor] = None,
        #         use_cache: Optional[bool] = None,
        #         output_attentions: Optional[bool] = None,
        #         output_hidden_states: Optional[bool] = None,
        #         return_dict: Optional[bool] = None,
        #     ) -> Union[Tuple, CausalLMOutputWithPast]:
        #         r"""
        #         Args:
        #             labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        #                 Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        #                 config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        #                 (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        #
        #         Returns:
        #
        #         Example:
        #
        #         ```python
        #         >>> from transformers import AutoTokenizer, LlamaForCausalLM
        #
        #         >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        #         >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        #
        #         >>> prompt = "Hey, are you conscious? Can you talk to me?"
        #         >>> inputs = tokenizer(prompt, return_tensors="pt")
        #
        #         >>> # Generate
        #         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        #         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        #         ```"""
        #
        #         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        #         output_hidden_states = (
        #             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        #         )
        #         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #
        #         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        #         outputs = self.model(
        #             input_ids=input_ids,
        #             attention_mask=attention_mask,
        #             position_ids=position_ids,
        #             past_key_values=past_key_values,
        #             inputs_embeds=inputs_embeds,
        #             use_cache=use_cache,
        #             output_attentions=output_attentions,
        #             output_hidden_states=output_hidden_states,
        #             return_dict=return_dict,
        #         )
        #
        #         hidden_states = outputs[0]
        #         if self.config.pretraining_tp > 1:
        #             lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        #             logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        #             logits = torch.cat(logits, dim=-1)
        #         else:
        #             logits = self.lm_head(hidden_states)
        #         logits = logits.float()
        #
        #         loss = None
        #         if labels is not None:
        #             # Shift so that tokens < n predict n
        #             shift_logits = logits[..., :-1, :].contiguous()
        #             shift_labels = labels[..., 1:].contiguous()
        #             # Flatten the tokens
        #             loss_fct = CrossEntropyLoss()
        #             shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #             shift_labels = shift_labels.view(-1)
        #             # Enable model parallelism
        #             shift_labels = shift_labels.to(shift_logits.device)
        #             loss = loss_fct(shift_logits, shift_labels)
        #             raise ValueError
        #
        #         if not return_dict:
        #             output = (logits,) + outputs[1:]
        #             return (loss,) + output if loss is not None else output
        #
        #         return CausalLMOutputWithPast(
        #             loss=loss,
        #             logits=logits,
        #             past_key_values=outputs.past_key_values,
        #             hidden_states=outputs.hidden_states,
        #             attentions=outputs.attentions,
        #         )

        # 以输入模型的第一个batch为例, 观察输入模型样本的组织方式及loss计算方法
        # =======================================================================================================
        # 调用 self.model(...) 前馈时的参数值:
        # input_ids.shape: [2, 512]
        # input_ids:
        # tensor([[30214, 33684, 34815, 32310, 33480, 30505, 44071, 30210, 35603, 33666,
        #          32579, 33698, 30267, 33298, 32188, 31072, 30909, 30275, 30630, 34453,
        #          33968, 40453, 30210, 35147, 32077, 30214, 32001, 32006, 38248, 30783,
        #          45679, 32137, 40747, 30503, 35399, 32400, 34716, 35260, 33383, 34303,
        #          34611, 30267, 32006, 32752, 40720, 45591, 30275, 31844, 30805, 31844,
        #          35229, 33065, 30214, 32614, 32483, 32316, 30210, 32342, 30214, 32857,
        #          37498, 30724, 30573, 34710, 32447, 34789, 38524, 34114, 30267, 49636,
        #          30214, 39610, 30505, 44071, 30210, 32366, 32832, 32354, 36792, 33688,
        #          40838, 30267,     2,     1, 29871, 43553, 33269, 30417, 32409, 35462,
        #          32157, 37792, 30267, 30417, 32409, 35462, 32157, 30210, 33269, 32072,
        #          32163, 36561, 30330, 47364, 31033, 31824, 40066, 30503, 31196, 35886,
        #          32926, 32522, 30267,     2,     1, 29871, 35868, 34590, 32739, 30210,
        #          32835, 41288, 31888, 41888, 30267, 34590, 30210, 32835, 41288, 31888,
        #          41888, 30383,     2,     1, 29871, 29896, 29889, 29871, 32232, 31666,
        #          35868, 30374, 32119, 30210, 48203,     2,     1, 29871, 29906, 29889,
        #          29871, 31267, 39132, 32592, 31026, 30437,     2,     1, 29871, 29941,
        #          29889, 29871, 31267, 30313, 31074, 32528, 32651, 30437, 30806, 30214,
        #          37934, 41212, 35567,     2,     1, 29871, 29946, 29889, 29871, 34536,
        #          30557, 39148, 30210, 33479, 33494,     2,     1, 29871, 29945, 29889,
        #          29871, 30573, 39621, 44742, 34536, 35659,     2,     1, 29871, 29953,
        #          29889, 29871, 32264, 30429, 30502, 38402, 30210, 32388, 32273,     2,
        #              1, 29871, 29955, 29889, 29871, 32586, 32016, 33169, 30210, 33194,
        #          32277,     2,     1, 29871, 30998, 33456, 29941, 29945, 36148, 30573,
        #          32661, 31466, 30354, 30545, 30267, 29941, 29889, 29945, 13105, 29871,
        #          29896, 29900, 29985, 29896,     2,     1, 29871, 30998, 31389, 32298,
        #          32813, 30573, 36401, 30330, 44176, 30883, 31391, 30275, 30883, 30267,
        #           1576,   379, 14287,   315,   440,   293, 44176, 30883, 30267,     2,
        #              1, 29871, 41540, 33269, 33098, 31074, 30909, 36665, 32913, 36523,
        #          32295, 30267, 33269, 33098, 31074, 30909, 36665, 32913, 36523, 32295,
        #          30392, 32109, 32333, 40295, 30330, 29941, 29945, 29900, 29889,   990,
        #          30503, 33830, 35190, 32295, 30267,     2,     1, 29871, 43553, 37407,
        #          37021, 29896, 29947, 33374, 35637, 32398, 30267, 29896, 29947, 33374,
        #          37933, 30904, 30631, 32406, 32398, 36927, 30383, 33099, 30257, 34298,
        #          30419, 29896, 29955, 29947, 29929, 29899, 29896, 29955, 29929, 29929,
        #          30470, 30409, 30214, 32048, 33003, 33395, 30419, 29896, 29955, 29955,
        #          29945, 29899, 29896, 29955, 29947, 29941, 30470, 30409, 30214, 32978,
        #          34298, 30419, 29896, 29955, 29953, 29900, 29899, 29896, 29947, 29906,
        #          29900, 30470, 30409, 30214, 39424, 33395, 30419, 29896, 29955, 29945,
        #          29953, 29899, 29896, 29955, 29953, 29941, 30470, 30409, 30503, 30581,
        #          30533, 34298, 30419, 29896, 29955, 29929, 29896, 29899, 29896, 29947,
        #          29900, 29946, 30470, 30409,     2,     1, 29871, 42149, 32252, 31267,
        #          29915,  4117, 29915, 36916, 36035, 30210, 37397, 30267, 31267, 29915,
        #           4117, 29915, 36916, 36035, 30210, 37397, 30417, 29901,  7548, 29892,
        #           3056, 29892, 17152, 29892,  3290, 29892,  1775, 29892,  2373, 29892,
        #           9950, 29892,   325,   271, 29892, 13563, 29892,   330,  8924, 29892,
        #            885,   271, 30503, 29887,  5450, 30267,     2,     1, 29871, 35031,
        #          35580, 32501, 32736, 45144, 30267, 29908, 30919, 32297, 32854, 32009,
        #          36254, 30584, 29908, 32305, 31076, 32602, 30267,     2,     1, 29871,
        #          31479, 32002, 32572, 32450, 31267, 30494, 40441, 33224, 33799, 30210,
        #          32534, 33211, 32480, 30267, 44439, 34494, 31267, 33188, 30494, 40441,
        #          30732, 39743, 30214, 32181, 32008, 30815, 33685, 32881, 30577, 31347,
        #          30267, 44439],
        #         [32545, 31605, 31855, 39955, 38239, 30267, 32078, 32735, 36191, 30743,
        #          31267, 34883, 45804, 37719, 35918, 32330, 32668, 30214, 33231, 30528,
        #          34585, 30503, 34291, 34755, 38793, 32316, 37880, 43064, 30214, 38797,
        #          33112, 30743, 34491, 33930, 33975, 37729, 37756, 30267, 31605, 31855,
        #          31994, 40632, 36758, 30330, 32430, 30486, 31605, 30503, 46750, 30214,
        #          30417, 31931, 30909, 34100, 32152, 32200, 38104, 37344, 33557, 33730,
        #          30805, 32452, 32330, 30267, 32232, 31994, 32083, 30214, 32605, 31605,
        #          31855, 32039, 32354, 38079, 39367, 32647, 32756, 30503, 38079, 34491,
        #          30685, 30883, 33072, 33887, 32243, 37756, 30267, 34972, 30214, 31605,
        #          31855, 47098, 32330, 34020, 31548, 30392, 41792, 37992, 30210, 30267,
        #              2,     1, 29871, 33976, 32817, 34289, 33148, 30267, 32138, 34799,
        #          48369, 51686, 30214, 33783, 42997, 30267, 30672, 37296, 32535, 32119,
        #          30214, 35728, 50082, 30214, 31666, 34188, 31016, 35973, 30267, 33442,
        #          38723, 37296, 32305, 31888, 42321, 32739, 30214, 31267, 32960, 30437,
        #          30806, 32807, 30743, 33566, 30780, 34315, 32277, 30214, 31666, 37296,
        #          36294, 30505, 36255, 30210, 35955, 34834, 30210, 31120, 34490, 30267,
        #          36220, 33488, 30544, 32037, 31475, 33820, 32096, 32185, 43093, 30214,
        #          34748, 30210, 49642, 32164, 32897, 32100, 30214, 32051, 34188, 31016,
        #          32296, 32305, 38729, 30267,     2,     1, 33110, 35552, 35556, 32401,
        #          30882, 32003, 31658, 32601, 32020, 30805, 35552, 35556, 32401, 30383,
        #          46652, 33434, 30210, 32592, 32195, 30882, 32084, 39362, 32592, 30210,
        #          34896, 30503, 36376, 30882, 32084, 34100, 32592, 48945, 30882, 46652,
        #          32592, 38693, 30882, 32454, 30505, 39146, 30503, 39146, 34788, 32595,
        #          32084, 30882, 32454, 32204, 35235, 34452, 32592, 32086, 32367, 30503,
        #          32407, 30882,     2,     1, 29871, 31522, 30544, 32379, 32740, 42037,
        #          31180, 30210, 37713, 30267, 32740, 42037, 31180, 30210, 39132, 37713,
        #          30392, 32545, 31669, 30716, 32093, 30214, 33231, 32147, 35044, 30716,
        #          38049, 30503, 40202, 37658, 30943, 30214, 32051, 31960, 42331, 43956,
        #          30267, 32725, 30214, 32117, 38429, 30503, 33985, 37053, 33792, 30716,
        #          32528, 32506, 32132, 30214, 46166, 32568, 32286, 30716, 32528, 32506,
        #          40392, 30267, 32219, 31994, 32117, 30573, 32059, 31669, 30716, 32093,
        #          32740, 42037, 31180, 38328, 34536, 38575, 33380, 30267,     2,     1,
        #          29871, 33530, 32002, 34538, 30406, 50375, 30267, 35871, 33018, 40895,
        #          31390, 30816, 30392, 37667, 30210, 30214, 44971, 50489, 50489, 30267,
        #          45580, 30505, 36734, 30275, 33877, 30392, 34863, 31619, 30214, 33409,
        #          30210, 35566, 31217, 30505, 31935, 32077, 30275, 37998, 34715, 38021,
        #          30267, 31594, 46172, 30214, 32003, 33835, 33708, 33747, 32346, 30210,
        #          35467, 32209, 30503, 50958, 53349, 30210, 35467, 32209, 30267, 32850,
        #          34533, 42745, 30214, 39271, 32085, 46172, 36708, 30710, 38324, 30275,
        #          33708, 36401, 42553, 33348, 30210, 32837, 33583, 31594, 39194, 30275,
        #          52650, 31331, 30672, 37887, 30267,     2,     1, 29871, 35170, 30275,
        #          32003, 32059, 32796, 34028, 30805, 40259, 34685, 30882, 35170, 30275,
        #          32003, 36702, 39588, 34028, 32163, 32059, 39588, 49744, 30214, 38823,
        #          32059, 32333, 30867, 30214, 32059, 36820, 47826, 36010, 30413, 32059,
        #          30594, 35369, 33409, 30214, 31295, 36702, 32897, 32693, 34265, 32351,
        #          33712, 31584, 30214, 32645, 40202, 37658, 32037, 42018, 30214, 31557,
        #          30505, 36199, 30594, 32059, 44005, 31391, 34546, 30214, 32051, 30406,
        #          43157, 43901, 31520, 30267,     2,     1, 29871, 32839, 32116, 32555,
        #          35383, 32277, 35460, 30267, 32555, 35383, 35460, 30214, 41676, 30417,
        #          31931, 30909, 38575, 30503, 33677, 33368, 32347, 30330, 39362, 33368,
        #          36619, 30330, 35416, 32005, 40284, 32276, 31666, 32826, 35552, 33368,
        #          34028, 30330]], device='cuda:0')
        #
        # attention_mask.shape: [2, 512]
        # attention_mask: 是一个全1张量
        #
        # labels: 和input_ids一模一样
        #
        # position_ids: None
        # past_key_values: None
        # inputs_embeds: None
        # use_cache: None
        # output_attentions: None(刚进入时) -> False(经过config赋值后)
        # output_hidden_states: None(刚进入时) -> False(经过config赋值后)
        # return_dict: None(刚进入时) -> True(经过config赋值后)

        # self.model(...) 调用后的各变量值:
        # len(outputs): 1
        # type(outputs): <class 'transformers.modeling_outputs.BaseModelOutputWithPast'>
        # hidden_states.shape: [2, 512, 5120]
        # logits.shape: [2, 512, 55296], 注意55296是词表大小
        # labels.shape: [2, 512]

        # loss = CrossEntropyLoss()(logits[..., :-1, :].view(-1, 55296), labels[..., 1:].view(-1))
        # loss.shape: [1022]
        # loss:
        # tensor([8.7310e+00, 1.1877e+01, 9.5181e+00, 1.0102e+01, 8.5739e+00, 1.2489e+01,
        #         5.1358e+00, 1.3528e+01, 1.2562e+01, 1.1216e+01, 1.1721e+01, 5.8467e+00,
        #         1.1031e+01, 8.7759e+00, 1.2573e+01, 7.7007e+00, 8.0372e+00, 7.5780e+00,
        #         8.2972e+00, 1.1174e+01, 1.1593e+01, 6.5241e+00, 1.0122e+01, 9.8864e+00,
        #         7.8430e+00, 1.1230e+01, 1.2343e+01, 1.1488e+01, 9.3370e+00, 1.0197e+01,
        #         1.0061e+01, 9.9297e+00, 7.5389e+00, 1.1386e+01, 1.0943e+01, 9.2240e+00,
        #         1.1330e+01, 9.6927e+00, 1.2446e+01, 1.0442e+01, 4.4784e+00, 1.2285e+01,
        #         1.2701e+01, 1.1849e+01, 9.9636e+00, 9.9599e+00, 1.0184e+01, 5.4365e+00,
        #         4.2909e+00, 8.3199e+00, 1.0497e+01, 3.5983e+00, 1.0565e+01, 1.0396e+01,
        #         1.0219e+01, 5.9113e+00, 1.1387e+01, 6.8020e+00, 1.1068e+01, 1.1432e+01,
        #         8.7317e+00, 7.8521e+00, 1.1855e+01, 1.0503e+01, 1.3422e+01, 1.0307e+01,
        #         1.1573e+01, 2.0926e+00, 1.1488e+01, 6.7613e+00, 1.1797e+01, 6.7606e+00,
        #         1.0192e+01, 2.0339e+00, 1.2630e+01, 1.0213e+01, 1.0476e+01, 1.2017e+01,
        #         1.3589e+01, 1.1106e+01, 5.1998e+00, 5.3683e+00, 1.7881e-06, 3.0088e+00,
        #         1.3890e+01, 1.2836e+01, 7.6888e+00, 1.1956e+01, 1.2130e+01, 1.1649e+01,
        #         1.2556e+01, 7.5581e+00, 7.4026e+00, 1.1747e+01, 1.2539e+01, 1.1278e+01,
        #         3.7582e+00, 1.2482e+01, 1.1693e+01, 1.0505e+01, 1.3642e+01, 5.0229e+00,
        #         1.0267e+01, 1.2634e+01, 1.0834e+01, 1.1323e+01, 2.5694e+00, 1.1913e+01,
        #         1.2945e+01, 1.2719e+01, 1.4356e+01, 1.8747e+00, 6.3297e+00, 1.7881e-06,
        #         3.3546e+00, 1.2020e+01, 1.0044e+01, 1.2722e+01, 5.7725e+00, 1.0388e+01,
        #         1.1926e+01, 9.5440e+00, 1.1301e+01, 3.7675e+00, 9.2268e+00, 7.0388e+00,
        #         1.0009e+01, 1.2641e+01, 3.0154e+00, 1.0524e+01, 4.0111e+00, 6.6671e+00,
        #         1.7881e-06, 2.3146e+00, 2.6746e+00, 2.4707e-01, 2.0421e+00, 1.0260e+01,
        #         1.1141e+01, 1.1425e+01, 9.8174e+00, 1.1770e+01, 5.2586e+00, 1.2564e+01,
        #         9.5716e+00, 1.7881e-06, 2.6326e+00, 2.4467e+00, 1.2596e-01, 9.6342e-01,
        #         7.5919e+00, 1.1762e+01, 1.2411e+01, 8.6416e+00, 7.2133e+00, 9.5466e+00,
        #         1.7881e-06, 3.0479e+00, 6.7878e-01, 4.6959e-03, 6.9506e-01, 4.2313e+00,
        #         8.7478e+00, 7.4333e+00, 1.1671e+01, 1.2732e+01, 8.7574e+00, 5.3792e+00,
        #         4.9542e+00, 1.2469e+01, 1.1466e+01, 1.0054e+01, 6.5172e+00, 1.7881e-06,
        #         2.2785e+00, 2.8392e-01, 1.6742e-03, 3.3566e-01, 1.1018e+01, 8.6935e+00,
        #         1.3621e+01, 4.3388e+00, 1.2827e+01, 1.1792e+01, 7.4383e+00, 1.7881e-06,
        #         2.3000e+00, 5.7975e-02, 2.7200e-04, 3.4709e-01, 6.1976e+00, 1.3106e+01,
        #         1.1529e+01, 1.0607e+01, 1.1532e+01, 7.1347e+00, 1.7881e-06, 2.3718e+00,
        #         9.7492e-02, 4.4396e-04, 3.0611e-01, 1.1396e+01, 8.7237e+00, 3.3703e+00,
        #         1.0660e+01, 5.2320e+00, 1.2270e+01, 1.2166e+01, 6.4381e+00, 1.7881e-06,
        #         2.2906e+00, 5.8447e-02, 7.5814e-05, 1.6117e-01, 1.1896e+01, 1.1612e+01,
        #         1.0133e+01, 5.5690e+00, 1.3076e+01, 1.0455e+01, 6.8158e+00, 1.7881e-06,
        #         2.1315e+00, 1.0852e+01, 1.3125e+01, 1.1971e+01, 5.7936e+00, 1.1420e+01,
        #         5.9025e+00, 9.1343e+00, 8.3598e+00, 2.7063e+00, 5.5269e+00, 2.8997e+00,
        #         8.3402e+00, 3.0424e+00, 1.8040e+00, 7.4035e+00, 4.8129e-02, 2.9749e-01,
        #         8.5925e-04, 1.0643e-01, 4.1665e+00, 8.9127e+00, 1.7881e-06, 2.6220e+00,
        #         5.9425e+00, 7.4380e+00, 1.1768e+01, 9.6838e+00, 2.8133e+00, 1.0721e+01,
        #         7.5970e+00, 1.2538e+01, 1.0679e+01, 7.1514e+00, 8.5633e+00, 5.8775e+00,
        #         1.6268e+00, 9.3296e+00, 6.2860e+00, 3.4584e+00, 3.3600e-01, 5.5559e-02,
        #         3.4571e-06, 1.3415e+01, 5.1793e+00, 3.7365e+00, 5.1078e+00, 1.7881e-06,
        #         2.8490e+00, 1.2334e+01, 1.1752e+01, 1.1710e+01, 1.0891e+01, 8.5734e+00,
        #         1.2625e+01, 1.1246e+01, 1.1750e+01, 1.3034e+01, 3.5690e+00, 1.1949e+01,
        #         1.2737e+01, 3.7047e+00, 1.8761e+00, 1.3003e+01, 1.1635e+01, 1.2386e+01,
        #         1.5095e+01, 7.3320e+00, 1.4702e+01, 1.1994e+01, 1.2654e+01, 8.2753e+00,
        #         8.0239e+00, 2.0011e+00, 2.1181e+00, 6.4463e+00, 8.4370e+00, 7.1132e+00,
        #         1.2290e+01, 1.0189e+01, 1.3123e+01, 1.2652e+00, 5.2811e+00, 1.7881e-06,
        #         2.6819e+00, 1.3481e+01, 1.0382e+01, 9.4057e+00, 9.6297e+00, 3.9852e+00,
        #         1.0998e+01, 9.4354e+00, 1.0446e+01, 2.6733e+00, 4.7094e+00, 1.5238e-01,
        #         1.0558e+01, 1.0260e+01, 1.2291e+01, 6.5212e+00, 1.1023e+01, 9.9171e+00,
        #         1.2514e+01, 7.6292e+00, 1.1928e+01, 8.0800e+00, 1.3551e+01, 6.4357e+00,
        #         2.3821e+00, 5.4269e+00, 3.4370e+00, 3.4901e+00, 5.7129e-01, 2.4367e-02,
        #         4.4505e+00, 1.3306e-03, 3.6081e-01, 5.6836e+00, 3.9618e-03, 1.0531e+00,
        #         1.2678e+01, 1.1626e+01, 1.2370e+01, 7.0802e-01, 2.9294e-03, 2.1732e+00,
        #         4.4093e+00, 1.9070e+00, 1.9855e-03, 1.4721e-03, 6.4005e-02, 1.9445e-01,
        #         3.3440e+00, 3.2987e-03, 1.1788e-03, 8.8207e-02, 1.3789e+01, 1.1512e+01,
        #         4.8635e-01, 1.0055e-03, 1.5085e-02, 5.1427e-01, 9.1414e-01, 1.8237e-04,
        #         5.6215e-04, 9.9297e+00, 2.6322e+00, 1.0911e-01, 1.9406e-03, 2.3272e-03,
        #         5.1196e-02, 1.2210e+01, 1.2642e+01, 1.3097e-01, 1.2026e-03, 1.9325e-02,
        #         2.0373e+00, 6.3347e+00, 4.5242e-04, 6.2589e-04, 6.7250e-02, 2.0349e-01,
        #         2.3848e+00, 9.2428e-04, 7.3728e-04, 6.4823e+00, 1.2129e+01, 4.8747e+00,
        #         1.0084e+01, 7.7091e-01, 4.6230e-03, 2.4276e-02, 4.8425e+00, 2.0846e+00,
        #         3.9971e-03, 7.5145e-04, 4.2464e-01, 2.0322e-01, 2.6099e-01, 6.4334e-03,
        #         1.8250e-03, 8.6060e+00, 1.7881e-06, 3.9157e+00, 1.0532e+01, 9.2852e+00,
        #         9.8925e+00, 1.1685e+01, 9.5486e+00, 4.8536e-02, 1.3240e+01, 1.3790e+01,
        #         6.4544e+00, 1.1958e+01, 2.2450e+00, 6.4635e+00, 6.3695e-01, 2.2201e+00,
        #         7.3998e-03, 1.1329e+01, 1.4478e+01, 3.5144e-01, 1.1755e+01, 7.8822e+00,
        #         1.1697e+01, 1.1463e+01, 1.2094e+00, 1.3437e+00, 1.1920e-02, 2.6618e+00,
        #         4.7717e-02, 1.8292e+00, 1.8440e-01, 1.1479e+00, 4.6597e-01, 8.0093e-01,
        #         8.1126e-01, 5.4037e+00, 9.6889e-01, 6.3472e+00, 3.0517e-03, 1.1446e+00,
        #         2.5612e+00, 5.1228e-01, 9.8006e+00, 3.0221e-01, 1.9670e-01, 9.3050e+00,
        #         5.6384e-05, 6.6710e+00, 7.2182e+00, 5.8561e+00, 2.9186e-01, 4.4811e+00,
        #         1.7881e-06, 2.6332e+00, 1.4191e+01, 1.0821e+01, 1.2335e+01, 1.0872e+01,
        #         1.1504e+01, 2.1303e+00, 8.4523e+00, 6.9656e+00, 1.1391e+01, 1.1937e+01,
        #         1.2564e+01, 1.1005e+01, 7.6306e+00, 1.8939e+00, 1.0710e+01, 9.1497e+00,
        #         1.2154e+01, 1.5782e+00, 4.2306e+00, 1.7881e-06, 2.9791e+00, 1.0591e+01,
        #         1.2585e+01, 1.2000e+01, 8.6872e+00, 8.5317e+00, 9.7817e+00, 1.5692e+01,
        #         1.2105e+01, 1.0018e+01, 4.8178e+00, 1.1482e+01, 9.3602e+00, 1.0857e+01,
        #         9.9543e-01, 1.1635e+01, 1.3677e+01, 7.3651e+00, 1.1452e+01, 4.7363e+00,
        #         1.3320e+01, 8.3934e+00, 1.1159e+01, 5.2033e+00, 1.2045e+01, 1.4143e+01,
        #         8.7168e+00, 9.7758e+00, 1.4203e+01, 7.1753e+00, 9.5194e+00, 5.8102e-01,
        #         1.1847e+01, 1.2450e+01, 1.3038e+01, 1.1576e+01, 1.1327e+01, 1.2526e+01,
        #         1.1231e+01, 9.0586e+00, 1.0023e+01, 1.1999e+01, 1.3063e+01, 1.3826e+01,
        #         9.9841e+00, 9.9803e+00, 1.2149e+01, 1.2582e+01, 1.2029e+01, 8.3942e+00,
        #         9.8731e+00, 9.8383e+00, 1.1590e+01, 1.0509e+01, 1.2538e+01, 9.8701e+00,
        #         1.2357e+01, 1.1790e+01, 1.0503e+01, 1.1684e+01, 7.1222e+00, 9.2176e+00,
        #         1.2776e+01, 8.5144e+00, 1.2353e+01, 1.0306e+01, 7.4619e+00, 1.2361e+01,
        #         1.1476e+01, 5.3815e+00, 1.4608e+01, 1.0972e+01, 1.1160e+01, 1.3417e+01,
        #         1.3516e+01, 7.9884e+00, 9.7698e+00, 1.2714e+01, 1.0511e+01, 7.8164e+00,
        #         1.1738e+01, 5.4496e+00, 8.2679e+00, 7.9096e+00, 3.0377e+00, 1.0951e+01,
        #         1.1725e+01, 9.7506e+00, 1.0155e+01, 1.1846e+01, 1.1992e+01, 1.1415e+01,
        #         1.0763e+01, 1.2657e+01, 1.2504e+01, 4.9954e+00, 1.0915e+01, 1.0779e+01,
        #         1.0100e+01, 5.0339e+00, 9.8213e+00, 1.1922e+01, 3.4105e+00, 9.6759e+00,
        #         1.0011e+01, 9.4982e+00, 1.0704e+01, 9.5519e+00, 9.8992e+00, 8.0653e+00,
        #         9.9302e+00, 1.2355e+01, 1.1217e+01, 1.2101e+01, 1.0116e+01, 1.2643e+01,
        #         1.0630e+01, 1.2005e+01, 3.4983e+00, 1.0341e+01, 6.7295e+00, 1.0186e+01,
        #         1.9531e+00, 1.1300e+01, 1.2934e+01, 1.1684e+01, 1.0772e+01, 7.7693e+00,
        #         1.2408e+01, 8.8689e+00, 4.4974e+00, 4.3979e+00, 4.1693e+00, 1.7881e-06,
        #         2.1122e+00, 1.0445e+01, 1.1800e+01, 1.1979e+01, 1.1807e+01, 9.3481e+00,
        #         1.1269e+01, 1.1960e+01, 1.0904e+01, 1.1241e+01, 3.8311e+00, 1.1093e+01,
        #         1.1857e+01, 3.4737e+00, 8.6816e+00, 1.1938e+01, 1.1096e+01, 1.2471e+01,
        #         1.3101e+00, 9.2715e+00, 1.2380e+01, 3.9307e+00, 6.8830e+00, 1.2053e+01,
        #         1.0953e+01, 1.2207e+01, 5.3457e+00, 1.0650e+01, 1.2701e+01, 1.1951e+01,
        #         1.0412e+01, 7.7834e+00, 1.0768e+01, 1.2661e+01, 1.1194e+00, 8.1500e+00,
        #         1.1857e+01, 9.2162e+00, 6.0957e+00, 9.5153e+00, 8.3243e+00, 1.2739e+01,
        #         1.0140e+01, 1.1970e+01, 1.0724e+01, 2.3821e+00, 2.2341e+00, 1.1347e+01,
        #         1.3299e+01, 5.6803e+00, 1.3421e+01, 7.3183e+00, 1.2028e+01, 1.1155e+01,
        #         6.0314e+00, 1.0083e+01, 1.4778e+01, 1.2860e+00, 9.9105e+00, 1.2590e+01,
        #         1.0874e+01, 1.1022e+01, 7.9303e+00, 1.1988e+01, 1.0151e+01, 1.1637e+01,
        #         1.0985e+01, 1.1375e+00, 1.1402e+01, 8.3351e+00, 1.2798e+01, 1.2359e+01,
        #         1.1565e+01, 1.0255e+01, 2.3766e+00, 1.2113e+01, 1.0495e+01, 1.0167e+01,
        #         1.3691e+01, 1.1174e+01, 9.9656e+00, 2.5143e+00, 6.3395e+00, 1.7881e-06,
        #         1.4644e+01, 9.8440e+00, 1.0351e+01, 1.3380e+01, 8.1003e+00, 1.2392e+01,
        #         8.9152e+00, 1.3138e+01, 1.0122e+01, 7.9714e+00, 8.8878e+00, 1.1746e+01,
        #         1.3079e+01, 6.7556e+00, 9.9264e+00, 1.1351e+01, 6.5961e+00, 1.2944e+01,
        #         1.2090e+01, 5.6443e+00, 1.0544e+01, 1.2774e+01, 1.2054e+01, 6.8715e+00,
        #         1.1163e+01, 7.5672e+00, 1.0535e+01, 3.4259e+00, 1.0613e+01, 1.0467e+01,
        #         1.2335e+01, 1.0485e+01, 2.8399e+00, 1.1062e+01, 1.1728e+01, 1.1171e+01,
        #         2.4057e+00, 9.7688e+00, 7.2799e+00, 1.2003e+01, 7.8417e+00, 1.2144e+01,
        #         9.2050e+00, 1.2652e+01, 1.1967e+01, 2.8378e+00, 9.5501e+00, 1.2133e+01,
        #         1.0412e+01, 1.1725e+01, 1.1792e+01, 1.2551e+01, 1.2278e+01, 5.8989e+00,
        #         1.3879e+01, 3.4058e+00, 5.6956e+00, 1.7881e-06, 1.9464e+00, 9.3405e+00,
        #         7.1564e+00, 1.0055e+01, 1.0113e+01, 1.1119e+01, 1.0121e+01, 4.5854e+00,
        #         1.0524e+01, 2.7161e+00, 1.1271e+01, 1.1323e+01, 5.5533e+00, 1.0426e+00,
        #         1.1579e+01, 1.0261e+01, 5.5447e+00, 1.2053e+01, 8.7428e+00, 1.0199e+01,
        #         1.1765e+01, 2.6414e+00, 1.1026e+01, 1.2279e+01, 1.0730e+01, 6.7644e+00,
        #         9.9330e+00, 6.3850e+00, 1.1087e+01, 1.1665e+01, 9.2247e+00, 3.5821e+00,
        #         1.2081e+01, 1.2935e+01, 1.3421e+01, 1.2207e+01, 3.7298e+00, 1.1844e+01,
        #         5.0425e+00, 9.6756e+00, 1.3874e+01, 4.9156e+00, 9.5371e+00, 1.1622e+01,
        #         9.3368e+00, 7.3592e+00, 9.8581e+00, 1.2330e+01, 1.0538e+01, 2.6341e+00,
        #         1.2319e+01, 1.1536e+01, 1.0909e+01, 4.6555e+00, 9.9843e+00, 1.2173e+01,
        #         1.2422e+01, 1.2355e+00, 1.1902e+01, 8.1566e+00, 1.0461e+01, 6.9431e+00,
        #         1.0611e+01, 7.2971e+00, 6.1983e-01, 1.1484e+01, 1.3165e+01, 1.1886e+01,
        #         6.2435e+00, 1.3018e+01, 1.0435e+01, 1.1880e+01, 1.1769e+01, 2.6525e+00,
        #         5.3243e+00, 1.7881e-06, 2.2702e+00, 1.3992e+01, 9.7616e+00, 1.1043e+01,
        #         6.9222e+00, 1.0318e+01, 4.5813e+00, 1.0695e+01, 1.3262e+01, 1.2424e+01,
        #         1.1556e+01, 7.0239e+00, 7.5631e+00, 9.0483e+00, 4.7571e+00, 2.5215e+00,
        #         1.0861e+01, 1.0286e+01, 9.6034e+00, 3.8141e+00, 1.2116e+01, 6.7331e+00,
        #         1.3435e+01, 5.5941e+00, 1.1653e+01, 5.9137e+00, 1.2179e+01, 9.2601e+00,
        #         4.6028e+00, 1.1872e+01, 5.1120e+00, 1.0458e+01, 1.2171e+01, 7.2796e+00,
        #         9.5259e+00, 1.1406e+01, 6.2990e+00, 1.3688e+01, 1.1401e+01, 1.3991e+01,
        #         1.5383e+00, 8.5538e+00, 1.2715e+01, 6.5946e+00, 1.1526e+01, 1.0336e+01,
        #         1.0355e+01, 1.0557e+01, 1.0923e+01, 5.0614e+00, 9.3130e+00, 1.1351e+01,
        #         6.5060e+00, 1.1823e+01, 1.2487e+01, 6.2528e+00, 9.6726e+00, 1.1423e+01,
        #         1.0087e+00, 1.3115e+01, 1.1392e+01, 1.2481e+01, 4.6256e+00, 1.1400e+01,
        #         9.9441e+00, 1.4485e+01, 1.1849e+01, 1.1933e+01, 7.2654e+00, 8.2065e+00,
        #         1.1040e+01, 1.1934e+01, 1.0759e+01, 1.0832e+01, 4.6522e+00, 1.2940e+01,
        #         1.2923e+01, 8.6101e+00, 9.4943e+00, 8.0713e+00, 1.1262e+01, 8.7940e+00,
        #         6.2610e+00, 1.0224e+01, 4.5438e+00, 5.2600e+00, 1.7881e-06, 4.2256e+00,
        #         1.1137e+01, 7.2450e+00, 1.1417e+01, 1.2288e+01, 1.2534e+01, 1.2488e+01,
        #         6.4123e+00, 1.2247e+01, 8.9401e+00, 5.2444e+00, 1.2031e+01, 6.6382e+00,
        #         1.1232e+01, 1.1617e+01, 1.3476e+01, 1.4031e+01, 1.1455e+01, 1.2014e+01,
        #         1.2935e+01, 1.2698e+01, 2.1338e+00, 1.0495e+01, 1.0896e+01, 1.2214e+01,
        #         1.0532e+01, 3.2093e+00, 1.0730e+01, 1.0782e+01, 1.3164e+01, 1.1599e+01,
        #         7.3473e+00, 1.1030e+01, 7.7824e+00, 1.2580e+01, 1.1980e+01, 5.5036e+00,
        #         9.4872e+00, 1.0329e+01, 1.0293e+01, 1.3941e+01, 7.9591e+00, 1.2327e+01,
        #         1.1118e+01, 8.8446e+00, 2.9983e+00, 1.1748e+01, 1.1478e+01, 1.2827e+01,
        #         1.1191e+01, 1.0360e+01, 2.9821e+00, 7.8367e+00, 6.1221e+00, 1.1891e+01,
        #         6.7065e+00, 1.0938e+01, 1.1097e+01, 7.5913e+00, 1.2565e+01, 5.0216e+00,
        #         1.2135e+01, 7.3330e+00, 1.0672e+01, 1.1752e+01, 8.9021e+00, 3.7462e+00,
        #         5.0032e+00, 1.7881e-06, 2.6562e+00, 9.6384e+00, 1.1143e+01, 9.0774e+00,
        #         1.1498e+01, 1.0492e+01, 1.1992e+01, 2.6583e+00, 1.0259e+01, 1.2450e+01,
        #         1.2178e+01, 3.8494e+00, 9.6980e+00, 7.1465e+00, 7.7739e+00, 2.1540e+00,
        #         1.1463e+01, 5.1413e+00, 1.2404e+01, 1.1484e+01, 1.3070e+01, 5.6854e+00,
        #         1.2226e+01, 1.1038e+01, 1.2262e+01, 3.1071e+00, 1.3052e+01, 1.0622e+01,
        #         1.3195e+01, 9.9100e+00, 7.3380e+00, 1.1379e+01, 9.7054e+00, 1.1018e+01,
        #         1.2851e+01, 3.1616e+00], device='cuda:0', grad_fn=<NllLossBackward0>)
        # =======================================================================================================

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
