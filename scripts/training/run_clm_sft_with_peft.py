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
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import datasets
import torch
from build_dataset import build_instruction_dataset, DataCollatorForSupervisedDataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# 此代码的注释都是基于如下运行指令debug获得的
# torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
#     --deepspeed ds_zero2_no_offload.json \
#     --model_name_or_path /data1/csw_model_weights/chinese-llama-2-13b \
#     --tokenizer_name_or_path /data1/csw_model_weights/chinese-llama-2-13b \
#     --dataset_dir ../../data \
#     --per_device_train_batch_size 2 \
#     --do_train \
#     --do_eval \
#     --validation_file ../../data/alpaca_data_zh_51k.json \
#     --seed 14 \
#     --fp16 \
#     --num_train_epochs 2 \
#     --lr_scheduler_type cosine \
#     --learning_rate 1e-4 \
#     --warmup_ratio 0.03 \
#     --weight_decay 0 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --save_strategy steps \
#     --save_total_limit 3 \
#     --evaluation_strategy steps \
#     --eval_steps 250 \
#     --save_steps 500 \
#     --gradient_accumulation_steps 1 \
#     --preprocessing_num_workers 8 \
#     --max_seq_length 512 \
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
#     --load_in_kbits 16

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "sft_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "sft_lora_model")
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

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
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
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})

    max_seq_length: Optional[int] = field(default=1024)


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
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


    # training_args.should_log: True
    if training_args.should_log:
        # Whether or not the current process should produce log.
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
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
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
        # model_args.model_name_or_path: '/data1/csw_model_weights/chinese-llama-2-13b'
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
        # model_args.use_fast_tokenizer: True,
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

    if (len(tokenizer)) != 55296:
        raise ValueError(f"The vocab size of the tokenizer should be 55296, but found {len(tokenizer)}.\n"
                         "Please use Chinese-LLaMA-2 tokenizer.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset=None
    train_dataset = None

    if training_args.do_train:
        # training_args.do_train: True
        with training_args.main_process_first(desc="loading and tokenization"):
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

            path = Path(data_args.dataset_dir)
            # data_args.dataset_dir: '../../data'
            files = [os.path.join(path,file.name) for file in path.glob("*.json")]
            # files: ['../../data/alpaca_data_zh_51k.json']
            logger.info(f"Training files: {' '.join(files)}")
            train_dataset = build_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                # data_args.max_seq_length: 512
                data_cache_dir = None,
                preprocessing_num_workers = data_args.preprocessing_num_workers)
                # preprocessing_num_workers: 8

            # train_dataset:
            # Dataset({
            #     features: ['input_ids', 'labels'],
            #     num_rows: 51179
            # })

            # input_ids是将prompt、input、output的input_ids拼接在一起, 之后取前512个字符
            # labels是将prompt、input部分的input_ids全部变为-100, 其余部分同input_ids

            # train_dataset[0]:
            # {
            # 'input_ids': tensor([    1,   518, 25580, 29962,  3532, 14816, 29903,  6778,    13,  3492,
            #           526,   263,  8444, 20255, 29889, 32732, 32475, 31616, 30909, 31931,
            #         32764, 40742, 30267,    13, 29966,   829, 14816, 29903,  6778,    13,
            #            13, 32005, 32084, 30505, 43547, 32740, 42037, 30882,   518, 29914,
            #         25580, 29962, 29871, 29896, 29889, 29871, 32059, 31669, 30716, 37944,
            #         30214, 30847, 31669, 30716, 40202, 37658, 33215, 31584, 30503, 30716,
            #         38049, 30267, 29871,    13, 29906, 29889, 29871, 32059, 30716, 33805,
            #         31391, 30716, 35895, 36039, 32381, 34600, 30716, 30214, 33231, 49926,
            #         30503, 32645, 37658, 30267, 29871,    13, 29941, 29889, 29871, 30505,
            #         33128, 30275, 32342, 31669, 30716, 33815, 30267, 29871,    13, 29946,
            #         29889, 29871, 32520, 30716, 31624, 30503, 50117, 37132, 34948, 30716,
            #         32195, 30214, 31666, 33040, 35490, 33409, 30267, 29871,    13, 29945,
            #         29889, 29871, 37610, 32037, 42018, 30214, 32059, 32147, 35044, 40202,
            #         37658, 31584, 40329, 42037, 30267, 29871,    13, 29953, 29889, 29871,
            #         36039, 44474, 30214, 33727, 32780, 33431, 43187, 31838, 39383, 34269,
            #         30267, 29871,    13, 29955, 29889, 29871, 44911, 31391, 34387, 30880,
            #         30594, 31057, 32351, 30716, 38049, 30267, 29871,    13, 29947, 29889,
            #         29871, 32740, 39735, 30716, 49825, 32881, 30267, 29871,    13, 29929,
            #         29889, 29871, 38823, 30923, 30533, 35322, 32059, 33987, 30716, 30419,
            #         32501, 42925, 30330, 46508, 30716, 37591, 30503, 40202, 37658, 30210,
            #         30716, 30409, 30267, 29871,    13, 29896, 29900, 29889, 29871, 31557,
            #         32513, 34685, 34046, 30528, 30210, 49926, 31429, 30503, 42925, 30267,
            #         2]),
            # 'labels': tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100, 29871, 29896, 29889, 29871, 32059, 31669, 30716, 37944,
            #         30214, 30847, 31669, 30716, 40202, 37658, 33215, 31584, 30503, 30716,
            #         38049, 30267, 29871,    13, 29906, 29889, 29871, 32059, 30716, 33805,
            #         31391, 30716, 35895, 36039, 32381, 34600, 30716, 30214, 33231, 49926,
            #         30503, 32645, 37658, 30267, 29871,    13, 29941, 29889, 29871, 30505,
            #         33128, 30275, 32342, 31669, 30716, 33815, 30267, 29871,    13, 29946,
            #         29889, 29871, 32520, 30716, 31624, 30503, 50117, 37132, 34948, 30716,
            #         32195, 30214, 31666, 33040, 35490, 33409, 30267, 29871,    13, 29945,
            #         29889, 29871, 37610, 32037, 42018, 30214, 32059, 32147, 35044, 40202,
            #         37658, 31584, 40329, 42037, 30267, 29871,    13, 29953, 29889, 29871,
            #         36039, 44474, 30214, 33727, 32780, 33431, 43187, 31838, 39383, 34269,
            #         30267, 29871,    13, 29955, 29889, 29871, 44911, 31391, 34387, 30880,
            #         30594, 31057, 32351, 30716, 38049, 30267, 29871,    13, 29947, 29889,
            #         29871, 32740, 39735, 30716, 49825, 32881, 30267, 29871,    13, 29929,
            #         29889, 29871, 38823, 30923, 30533, 35322, 32059, 33987, 30716, 30419,
            #         32501, 42925, 30330, 46508, 30716, 37591, 30503, 40202, 37658, 30210,
            #         30716, 30409, 30267, 29871,    13, 29896, 29900, 29889, 29871, 31557,
            #         32513, 34685, 34046, 30528, 30210, 49926, 31429, 30503, 42925, 30267,
            #         2])}
            # tokenizer.convert_ids_to_tokens(train_dataset[0]['input_ids']):
            #     ['<s>', '▁[', 'INST', ']', '▁<<', 'SY', 'S', '>>', '<0x0A>', 'You', '▁are', '▁a', '▁helpful',
            #     '▁assistant', '.', '▁你', '是一个', '乐', '于', '助', '人的', '助手', '。', '<0x0A>', '<', '</',
            #     'SY', 'S', '>>', '<0x0A>', '<0x0A>', '我们', '如何', '在', '日常生活中', '减少', '用水', '？', '▁[',
            #     '/', 'INST', ']', '▁', '1', '.', '▁', '使用', '节', '水', '装置', '，', '如', '节', '水', '淋', '浴',
            #     '喷', '头', '和', '水', '龙头', '。', '▁', '<0x0A>', '2', '.', '▁', '使用', '水', '箱', '或', '水',
            #     '桶', '收集', '家庭', '废', '水', '，', '例如', '洗碗', '和', '洗', '浴', '。', '▁', '<0x0A>', '3',
            #     '.', '▁', '在', '社区', '中', '提高', '节', '水', '意识', '。', '▁', '<0x0A>', '4', '.', '▁', '检查',
            #     '水', '管', '和', '灌溉', '系统的', '漏', '水', '情况', '，', '并', '及时', '修复', '它们', '。', '▁',
            #     '<0x0A>', '5', '.', '▁', '洗澡', '时间', '缩短', '，', '使用', '低', '流量', '淋', '浴', '头', '节约',
            #     '用水', '。', '▁', '<0x0A>', '6', '.', '▁', '收集', '雨水', '，', '用于', '园', '艺', '或其他', '非',
            #     '饮用', '目的', '。', '▁', '<0x0A>', '7', '.', '▁', '刷牙', '或', '擦', '手', '时', '关', '掉', '水',
            #     '龙头', '。', '▁', '<0x0A>', '8', '.', '▁', '减少', '浇', '水', '草坪', '的时间', '。', '▁', '<0x0A>',
            #     '9', '.', '▁', '尽可能', '多', '地', '重复', '使用', '灰', '水', '（', '来自', '洗衣机', '、', '浴室',
            #     '水', '槽', '和', '淋', '浴', '的', '水', '）', '。', '▁', '<0x0A>', '1', '0', '.', '▁', '只', '购买',
            #     '能源', '效率', '高', '的', '洗碗', '机', '和', '洗衣机', '。', '</s>']

            # train_dataset[1]:
            # {
            # 'input_ids': tensor([    1,   518, 25580, 29962,  3532, 14816, 29903,  6778,    13,  3492,
            #           526,   263,  8444, 20255, 29889, 32732, 32475, 31616, 30909, 31931,
            #         32764, 40742, 30267,    13, 29966,   829, 14816, 29903,  6778,    13,
            #            13, 32155, 32949, 30214, 39560, 31100, 33573, 36114, 30267,    13,
            #         36348, 35796, 30392, 33893, 32357, 30210, 35077, 30214, 31407, 36480,
            #         33136, 37203, 32739, 35892, 32042, 39647, 30832, 34168, 30267, 36348,
            #         35796, 30505, 34318, 32178, 30275, 31407, 31844, 30805, 31844, 34455,
            #         30533, 32754, 30214, 31594, 37879, 30780, 33521, 37898, 41496, 32162,
            #         30267,   518, 29914, 25580, 29962, 29871, 36348, 35796, 30392, 33893,
            #         32357, 30210, 35077, 30214, 31407, 36480, 33136, 37203, 32739, 35892,
            #         32042, 39647, 30832, 34168, 30214, 33477, 37647, 33179, 34046, 30330,
            #         40752, 30898, 30503, 48001, 32316, 30267, 36348, 35796, 30505, 34318,
            #         32178, 30275, 31407, 31844, 30805, 31844, 34455, 30533, 32754, 30214,
            #         31594, 37879, 30214, 33409, 32003, 32059, 44813, 30503, 34360, 40765,
            #         47125, 39538, 47181, 30214, 30780, 33521, 37898, 30214, 32003, 38067,
            #         32034, 33521, 33275, 30503, 32382, 30214, 41496, 32162, 30214, 32003,
            #         35860, 43781, 32326, 30214, 33343, 32568, 30503, 35703, 30210, 32162,
            #         30267, 36348, 35796, 33663, 32740, 30505, 34136, 31391, 40594, 42968,
            #         30210, 33635, 30503, 32317, 32162, 30214, 30505, 32978, 34837, 30210,
            #         32520, 31391, 35811, 32439, 31184, 30267, 32150, 31149, 39683, 30952,
            #         30214, 36348, 35796, 30998, 34341, 32551, 32005, 32021, 32309, 33468,
            #         30214, 30785, 32739, 46133, 32583, 30330, 33026, 30214, 32408, 32505,
            #         44258, 30267,     2]),
            # 'labels': tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
            #          -100,  -100,  -100,  -100,  -100, 29871, 36348, 35796, 30392, 33893,
            #         32357, 30210, 35077, 30214, 31407, 36480, 33136, 37203, 32739, 35892,
            #         32042, 39647, 30832, 34168, 30214, 33477, 37647, 33179, 34046, 30330,
            #         40752, 30898, 30503, 48001, 32316, 30267, 36348, 35796, 30505, 34318,
            #         32178, 30275, 31407, 31844, 30805, 31844, 34455, 30533, 32754, 30214,
            #         31594, 37879, 30214, 33409, 32003, 32059, 44813, 30503, 34360, 40765,
            #         47125, 39538, 47181, 30214, 30780, 33521, 37898, 30214, 32003, 38067,
            #         32034, 33521, 33275, 30503, 32382, 30214, 41496, 32162, 30214, 32003,
            #         35860, 43781, 32326, 30214, 33343, 32568, 30503, 35703, 30210, 32162,
            #         30267, 36348, 35796, 33663, 32740, 30505, 34136, 31391, 40594, 42968,
            #         30210, 33635, 30503, 32317, 32162, 30214, 30505, 32978, 34837, 30210,
            #         32520, 31391, 35811, 32439, 31184, 30267, 32150, 31149, 39683, 30952,
            #         30214, 36348, 35796, 30998, 34341, 32551, 32005, 32021, 32309, 33468,
            #         30214, 30785, 32739, 46133, 32583, 30330, 33026, 30214, 32408, 32505,
            #         44258, 30267, 2])}
            # tokenizer.convert_ids_to_tokens(train_dataset[1]['input_ids']):
            #     ['<s>', '▁[', 'INST', ']', '▁<<', 'SY', 'S', '>>', '<0x0A>', 'You', '▁are', '▁a', '▁helpful',
            #     '▁assistant', '.', '▁你', '是一个', '乐', '于', '助', '人的', '助手', '。', '<0x0A>', '<', '</',
            #     'SY', 'S', '>>', '<0x0A>', '<0x0A>', '编辑', '文章', '，', '使其', '更', '吸引', '读者', '。',
            #     '<0x0A>', '自主', '机器人', '是', '计算机', '控制', '的', '机器', '，', '被', '编程', '执行', '特定',
            #     '任务', '而不', '需要', '任何人', '类', '输入', '。', '自主', '机器人', '在', '各个', '行业', '中',
            #     '被', '越', '来', '越', '广泛', '地', '应用', '，', '从', '制造业', '到', '医疗', '保健', '再到', '安全',
            #     '。', '▁[', '/', 'INST', ']', '▁', '自主', '机器人', '是', '计算机', '控制', '的', '机器', '，', '被',
            #     '编程', '执行', '特定', '任务', '而不', '需要', '任何人', '类', '输入', '，', '从而', '实现了', '新的',
            #     '效率', '、', '精确', '度', '和', '可靠性', '水平', '。', '自主', '机器人', '在', '各个', '行业', '中',
            #     '被', '越', '来', '越', '广泛', '地', '应用', '，', '从', '制造业', '，', '它们', '可以', '使用', '精度',
            #     '和', '一致', '的质量', '组装', '复杂的', '组件', '，', '到', '医疗', '保健', '，', '可以', '协助', '进行',
            #     '医疗', '测试', '和', '处理', '，', '再到', '安全', '，', '可以', '监控', '大面积', '地区', '，', '保障',
            #     '人们', '和', '财产', '的', '安全', '。', '自主', '机器人', '还可以', '减少', '在', '危险', '或', '有害',
            #     '环境中', '的', '错误', '和', '增加', '安全', '，', '在', '工业', '流程', '的', '检查', '或', '维修',
            #     '期间', '等', '。', '由于', '其', '多样', '性', '，', '自主', '机器人', '将', '彻底', '改变', '我们',
            #     '工作', '方式', '的方式', '，', '使', '任务', '变得更加', '简单', '、', '快速', '，', '最终', '更加',
            #     '愉悦', '。', '</s>']
            # tokenizer.decode(train_dataset[1]['input_ids']):
            #     '<s> [INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n编辑文章，使其更\
            #     吸引读者。\n自主机器人是计算机控制的机器，被编程执行特定任务而不需要任何人类输入。自主机器人在各个行业中被越来越广泛地\
            #     应用，从制造业到医疗保健再到安全。 [/INST] 自主机器人是计算机控制的机器，被编程执行特定任务而不需要任何人类输入，从而\
            #     实现了新的效率、精确度和可靠性水平。自主机器人在各个行业中被越来越广泛地应用，从制造业，它们可以使用精度和一致的质量组装\
            #     复杂的组件，到医疗保健，可以协助进行医疗测试和处理，再到安全，可以监控大面积地区，保障人们和财产的安全。自主机器人还可以\
            #     减少在危险或有害环境中的错误和增加安全，在工业流程的检查或维修期间等。由于其多样性，自主机器人将彻底改变我们工作方式的方\
            #     式，使任务变得更加简单、快速，最终更加愉悦。</s>'
            # for i, (id, token, label) in enumerate(zip(train_dataset[1]['input_ids'], tokenizer.convert_ids_to_tokens(train_dataset[1]['input_ids']), train_dataset[1]['labels'])):
            #     print(f"{i:5d}    id: {id:8d}   token: {token:15s}   label: {label:8d}")
            #     0    id:        1   token: <s>               label:     -100
            #     1    id:      518   token: ▁[                label:     -100
            #     2    id:    25580   token: INST              label:     -100
            #     3    id:    29962   token: ]                 label:     -100
            #     4    id:     3532   token: ▁<<               label:     -100
            #     5    id:    14816   token: SY                label:     -100
            #     6    id:    29903   token: S                 label:     -100
            #     7    id:     6778   token: >>                label:     -100
            #     8    id:       13   token: <0x0A>            label:     -100
            #     9    id:     3492   token: You               label:     -100
            #    10    id:      526   token: ▁are              label:     -100
            #    11    id:      263   token: ▁a                label:     -100
            #    12    id:     8444   token: ▁helpful          label:     -100
            #    13    id:    20255   token: ▁assistant        label:     -100
            #    14    id:    29889   token: .                 label:     -100
            #    15    id:    32732   token: ▁你                label:     -100
            #    16    id:    32475   token: 是一个               label:     -100
            #    17    id:    31616   token: 乐                 label:     -100
            #    18    id:    30909   token: 于                 label:     -100
            #    19    id:    31931   token: 助                 label:     -100
            #    20    id:    32764   token: 人的                label:     -100
            #    21    id:    40742   token: 助手                label:     -100
            #    22    id:    30267   token: 。                 label:     -100
            #    23    id:       13   token: <0x0A>            label:     -100
            #    24    id:    29966   token: <                 label:     -100
            #    25    id:      829   token: </                label:     -100
            #    26    id:    14816   token: SY                label:     -100
            #    27    id:    29903   token: S                 label:     -100
            #    28    id:     6778   token: >>                label:     -100
            #    29    id:       13   token: <0x0A>            label:     -100
            #    30    id:       13   token: <0x0A>            label:     -100
            #    31    id:    32155   token: 编辑                label:     -100
            #    32    id:    32949   token: 文章                label:     -100
            #    33    id:    30214   token: ，                 label:     -100
            #    34    id:    39560   token: 使其                label:     -100
            #    35    id:    31100   token: 更                 label:     -100
            #    36    id:    33573   token: 吸引                label:     -100
            #    37    id:    36114   token: 读者                label:     -100
            #    38    id:    30267   token: 。                 label:     -100
            #    39    id:       13   token: <0x0A>            label:     -100
            #    40    id:    36348   token: 自主                label:     -100
            #    41    id:    35796   token: 机器人               label:     -100
            #    42    id:    30392   token: 是                 label:     -100
            #    43    id:    33893   token: 计算机               label:     -100
            #    44    id:    32357   token: 控制                label:     -100
            #    45    id:    30210   token: 的                 label:     -100
            #    46    id:    35077   token: 机器                label:     -100
            #    47    id:    30214   token: ，                 label:     -100
            #    48    id:    31407   token: 被                 label:     -100
            #    49    id:    36480   token: 编程                label:     -100
            #    50    id:    33136   token: 执行                label:     -100
            #    51    id:    37203   token: 特定                label:     -100
            #    52    id:    32739   token: 任务                label:     -100
            #    53    id:    35892   token: 而不                label:     -100
            #    54    id:    32042   token: 需要                label:     -100
            #    55    id:    39647   token: 任何人               label:     -100
            #    56    id:    30832   token: 类                 label:     -100
            #    57    id:    34168   token: 输入                label:     -100
            #    58    id:    30267   token: 。                 label:     -100
            #    59    id:    36348   token: 自主                label:     -100
            #    60    id:    35796   token: 机器人               label:     -100
            #    61    id:    30505   token: 在                 label:     -100
            #    62    id:    34318   token: 各个                label:     -100
            #    63    id:    32178   token: 行业                label:     -100
            #    64    id:    30275   token: 中                 label:     -100
            #    65    id:    31407   token: 被                 label:     -100
            #    66    id:    31844   token: 越                 label:     -100
            #    67    id:    30805   token: 来                 label:     -100
            #    68    id:    31844   token: 越                 label:     -100
            #    69    id:    34455   token: 广泛                label:     -100
            #    70    id:    30533   token: 地                 label:     -100
            #    71    id:    32754   token: 应用                label:     -100
            #    72    id:    30214   token: ，                 label:     -100
            #    73    id:    31594   token: 从                 label:     -100
            #    74    id:    37879   token: 制造业               label:     -100
            #    75    id:    30780   token: 到                 label:     -100
            #    76    id:    33521   token: 医疗                label:     -100
            #    77    id:    37898   token: 保健                label:     -100
            #    78    id:    41496   token: 再到                label:     -100
            #    79    id:    32162   token: 安全                label:     -100
            #    80    id:    30267   token: 。                 label:     -100
            #    81    id:      518   token: ▁[                label:     -100
            #    82    id:    29914   token: /                 label:     -100
            #    83    id:    25580   token: INST              label:     -100
            #    84    id:    29962   token: ]                 label:     -100
            #    85    id:    29871   token: ▁                 label:    29871
            #    86    id:    36348   token: 自主                label:    36348
            #    87    id:    35796   token: 机器人               label:    35796
            #    88    id:    30392   token: 是                 label:    30392
            #    89    id:    33893   token: 计算机               label:    33893
            #    90    id:    32357   token: 控制                label:    32357
            #    91    id:    30210   token: 的                 label:    30210
            #    92    id:    35077   token: 机器                label:    35077
            #    93    id:    30214   token: ，                 label:    30214
            #    94    id:    31407   token: 被                 label:    31407
            #    95    id:    36480   token: 编程                label:    36480
            #    96    id:    33136   token: 执行                label:    33136
            #    97    id:    37203   token: 特定                label:    37203
            #    98    id:    32739   token: 任务                label:    32739
            #    99    id:    35892   token: 而不                label:    35892
            #   100    id:    32042   token: 需要                label:    32042
            #   101    id:    39647   token: 任何人               label:    39647
            #   102    id:    30832   token: 类                 label:    30832
            #   103    id:    34168   token: 输入                label:    34168
            #   104    id:    30214   token: ，                 label:    30214
            #   105    id:    33477   token: 从而                label:    33477
            #   106    id:    37647   token: 实现了               label:    37647
            #   107    id:    33179   token: 新的                label:    33179
            #   108    id:    34046   token: 效率                label:    34046
            #   109    id:    30330   token: 、                 label:    30330
            #   110    id:    40752   token: 精确                label:    40752
            #   111    id:    30898   token: 度                 label:    30898
            #   112    id:    30503   token: 和                 label:    30503
            #   113    id:    48001   token: 可靠性               label:    48001
            #   114    id:    32316   token: 水平                label:    32316
            #   115    id:    30267   token: 。                 label:    30267
            #   116    id:    36348   token: 自主                label:    36348
            #   117    id:    35796   token: 机器人               label:    35796
            #   118    id:    30505   token: 在                 label:    30505
            #   119    id:    34318   token: 各个                label:    34318
            #   120    id:    32178   token: 行业                label:    32178
            #   121    id:    30275   token: 中                 label:    30275
            #   122    id:    31407   token: 被                 label:    31407
            #   123    id:    31844   token: 越                 label:    31844
            #   124    id:    30805   token: 来                 label:    30805
            #   125    id:    31844   token: 越                 label:    31844
            #   126    id:    34455   token: 广泛                label:    34455
            #   127    id:    30533   token: 地                 label:    30533
            #   128    id:    32754   token: 应用                label:    32754
            #   129    id:    30214   token: ，                 label:    30214
            #   130    id:    31594   token: 从                 label:    31594
            #   131    id:    37879   token: 制造业               label:    37879
            #   132    id:    30214   token: ，                 label:    30214
            #   133    id:    33409   token: 它们                label:    33409
            #   134    id:    32003   token: 可以                label:    32003
            #   135    id:    32059   token: 使用                label:    32059
            #   136    id:    44813   token: 精度                label:    44813
            #   137    id:    30503   token: 和                 label:    30503
            #   138    id:    34360   token: 一致                label:    34360
            #   139    id:    40765   token: 的质量               label:    40765
            #   140    id:    47125   token: 组装                label:    47125
            #   141    id:    39538   token: 复杂的               label:    39538
            #   142    id:    47181   token: 组件                label:    47181
            #   143    id:    30214   token: ，                 label:    30214
            #   144    id:    30780   token: 到                 label:    30780
            #   145    id:    33521   token: 医疗                label:    33521
            #   146    id:    37898   token: 保健                label:    37898
            #   147    id:    30214   token: ，                 label:    30214
            #   148    id:    32003   token: 可以                label:    32003
            #   149    id:    38067   token: 协助                label:    38067
            #   150    id:    32034   token: 进行                label:    32034
            #   151    id:    33521   token: 医疗                label:    33521
            #   152    id:    33275   token: 测试                label:    33275
            #   153    id:    30503   token: 和                 label:    30503
            #   154    id:    32382   token: 处理                label:    32382
            #   155    id:    30214   token: ，                 label:    30214
            #   156    id:    41496   token: 再到                label:    41496
            #   157    id:    32162   token: 安全                label:    32162
            #   158    id:    30214   token: ，                 label:    30214
            #   159    id:    32003   token: 可以                label:    32003
            #   160    id:    35860   token: 监控                label:    35860
            #   161    id:    43781   token: 大面积               label:    43781
            #   162    id:    32326   token: 地区                label:    32326
            #   163    id:    30214   token: ，                 label:    30214
            #   164    id:    33343   token: 保障                label:    33343
            #   165    id:    32568   token: 人们                label:    32568
            #   166    id:    30503   token: 和                 label:    30503
            #   167    id:    35703   token: 财产                label:    35703
            #   168    id:    30210   token: 的                 label:    30210
            #   169    id:    32162   token: 安全                label:    32162
            #   170    id:    30267   token: 。                 label:    30267
            #   171    id:    36348   token: 自主                label:    36348
            #   172    id:    35796   token: 机器人               label:    35796
            #   173    id:    33663   token: 还可以               label:    33663
            #   174    id:    32740   token: 减少                label:    32740
            #   175    id:    30505   token: 在                 label:    30505
            #   176    id:    34136   token: 危险                label:    34136
            #   177    id:    31391   token: 或                 label:    31391
            #   178    id:    40594   token: 有害                label:    40594
            #   179    id:    42968   token: 环境中               label:    42968
            #   180    id:    30210   token: 的                 label:    30210
            #   181    id:    33635   token: 错误                label:    33635
            #   182    id:    30503   token: 和                 label:    30503
            #   183    id:    32317   token: 增加                label:    32317
            #   184    id:    32162   token: 安全                label:    32162
            #   185    id:    30214   token: ，                 label:    30214
            #   186    id:    30505   token: 在                 label:    30505
            #   187    id:    32978   token: 工业                label:    32978
            #   188    id:    34837   token: 流程                label:    34837
            #   189    id:    30210   token: 的                 label:    30210
            #   190    id:    32520   token: 检查                label:    32520
            #   191    id:    31391   token: 或                 label:    31391
            #   192    id:    35811   token: 维修                label:    35811
            #   193    id:    32439   token: 期间                label:    32439
            #   194    id:    31184   token: 等                 label:    31184
            #   195    id:    30267   token: 。                 label:    30267
            #   196    id:    32150   token: 由于                label:    32150
            #   197    id:    31149   token: 其                 label:    31149
            #   198    id:    39683   token: 多样                label:    39683
            #   199    id:    30952   token: 性                 label:    30952
            #   200    id:    30214   token: ，                 label:    30214
            #   201    id:    36348   token: 自主                label:    36348
            #   202    id:    35796   token: 机器人               label:    35796
            #   203    id:    30998   token: 将                 label:    30998
            #   204    id:    34341   token: 彻底                label:    34341
            #   205    id:    32551   token: 改变                label:    32551
            #   206    id:    32005   token: 我们                label:    32005
            #   207    id:    32021   token: 工作                label:    32021
            #   208    id:    32309   token: 方式                label:    32309
            #   209    id:    33468   token: 的方式               label:    33468
            #   210    id:    30214   token: ，                 label:    30214
            #   211    id:    30785   token: 使                 label:    30785
            #   212    id:    32739   token: 任务                label:    32739
            #   213    id:    46133   token: 变得更加              label:    46133
            #   214    id:    32583   token: 简单                label:    32583
            #   215    id:    30330   token: 、                 label:    30330
            #   216    id:    33026   token: 快速                label:    33026
            #   217    id:    30214   token: ，                 label:    30214
            #   218    id:    32408   token: 最终                label:    32408
            #   219    id:    32505   token: 更加                label:    32505
            #   220    id:    44258   token: 愉悦                label:    44258
            #   221    id:    30267   token: 。                 label:    30267
            #   222    id:        2   token: </s>              label:        2

        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("Training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))

    if training_args.do_eval:
        # training_args.do_eval: True
        with training_args.main_process_first(desc="loading and tokenization"):
            files = [data_args.validation_file]
            logger.info(f"Evaluation files: {' '.join(files)}")
            eval_dataset = build_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                data_cache_dir = None,
                preprocessing_num_workers = data_args.preprocessing_num_workers)
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("Evaluation example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    # torch_dtype: torch.float16
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
    device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
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
    if training_args.load_in_kbits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model.config.use_cache = False

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    if model_vocab_size != len(tokenizer):
        logger.info(f"Resize model vocab size to {len(tokenizer)}")
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
            # modules_to_save: 'embed_tokens,lm_head'
        lora_rank = training_args.lora_rank
        # lora_rank: 64
        lora_dropout = training_args.lora_dropout
        # lora_dropout: 0.05
        lora_alpha = training_args.lora_alpha
        # lora_alpha: 128.0
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            # TaskType.CAUSAL_LM: <TaskType.CAUSAL_LM: 'CAUSAL_LM'>
            target_modules=target_modules,
            # target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.add_callback(SavePeftModelCallback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            # training_args.resume_from_checkpoint: None
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            # last_checkpoint: None
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
        # input_ids.shape: [2, 135]
        # input_ids:
        # tensor([[    1,   518, 25580, 29962,  3532, 14816, 29903,  6778,    13,  3492,
        #            526,   263,  8444, 20255, 29889, 32732, 32475, 31616, 30909, 31931,
        #          32764, 40742, 30267,    13, 29966,   829, 14816, 29903,  6778,    13,
        #             13, 37656, 37407, 32060, 32262, 32949, 32261, 30210, 39078, 30267,
        #             13, 31751, 32949, 32807, 30743, 33662, 30952, 34101, 30210, 32447,
        #          33480, 34627, 39132, 30210, 32330, 34020, 31548, 30267,   518, 29914,
        #          25580, 29962, 29871, 29896, 29889, 42618, 33662, 30952, 34101, 30210,
        #          32330, 34020, 31548, 30882,    13, 29906, 29889, 29871, 35372, 33662,
        #          30952, 34101, 30210, 40457, 32875, 32084, 32447, 30882,    13, 29941,
        #          29889, 29871, 33662, 30952, 34101, 44577, 32306, 31391, 32151, 34020,
        #          31548, 30882,    13, 29946, 29889, 29871, 33662, 30952, 34101, 47060,
        #          32400, 33557, 34164, 32261, 30882,    13, 29945, 29889, 29871, 32454,
        #          30815, 32179, 32796, 32262, 40886, 30780, 33662, 30952, 34101, 30210,
        #          33796, 31391, 32528, 30882,     2],
        #         [    1,   518, 25580, 29962,  3532, 14816, 29903,  6778,    13,  3492,
        #            526,   263,  8444, 20255, 29889, 32732, 32475, 31616, 30909, 31931,
        #          32764, 40742, 30267,    13, 29966,   829, 14816, 29903,  6778,    13,
        #             13, 41671, 33335, 32109, 37528, 32231, 30592, 32009, 33637, 30267,
        #             13, 32455, 31505, 33267,   518, 29914, 25580, 29962, 29871, 32455,
        #          31505, 33267, 30505, 45837, 33226, 32879, 36861, 30392, 45573, 32258,
        #          33023, 32156, 32090, 30267, 34981, 32090, 30275, 30214, 32031, 30505,
        #          33023, 32156, 31407, 38943, 30594, 33000, 30743, 31649, 31391, 41148,
        #          30214, 32061, 32384, 46270, 33023, 32156, 30267, 36353, 32455, 31505,
        #          33267, 37528, 30214, 32017, 32142, 32940, 33000, 31649, 31391, 41148,
        #          30210, 32455, 30594, 30214, 33023, 32156, 34384, 31407, 46270, 30267,
        #              2, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        #          32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
        #          32000, 32000, 32000, 32000, 32000]], device='cuda:0')
        #
        # attention_mask.shape: [2, 135]
        # attention_mask:
        # tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True],
        #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False]], device='cuda:0')
        #
        # labels.shape: [2, 135]
        # labels:
        # tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100, 29871, 29896, 29889, 42618, 33662, 30952, 34101, 30210,
        #          32330, 34020, 31548, 30882,    13, 29906, 29889, 29871, 35372, 33662,
        #          30952, 34101, 30210, 40457, 32875, 32084, 32447, 30882,    13, 29941,
        #          29889, 29871, 33662, 30952, 34101, 44577, 32306, 31391, 32151, 34020,
        #          31548, 30882,    13, 29946, 29889, 29871, 33662, 30952, 34101, 47060,
        #          32400, 33557, 34164, 32261, 30882,    13, 29945, 29889, 29871, 32454,
        #          30815, 32179, 32796, 32262, 40886, 30780, 33662, 30952, 34101, 30210,
        #          33796, 31391, 32528, 30882,     2],
        #         [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 29871, 32455,
        #          31505, 33267, 30505, 45837, 33226, 32879, 36861, 30392, 45573, 32258,
        #          33023, 32156, 32090, 30267, 34981, 32090, 30275, 30214, 32031, 30505,
        #          33023, 32156, 31407, 38943, 30594, 33000, 30743, 31649, 31391, 41148,
        #          30214, 32061, 32384, 46270, 33023, 32156, 30267, 36353, 32455, 31505,
        #          33267, 37528, 30214, 32017, 32142, 32940, 33000, 31649, 31391, 41148,
        #          30210, 32455, 30594, 30214, 33023, 32156, 34384, 31407, 46270, 30267,
        #              2,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100]], device='cuda:0')
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
        # hidden_states.shape: [2, 135, 5120]
        # logits.shape: [2, 135, 55296], 注意55296是词表大小
        # labels.shape: [2, 135]

        # loss = CrossEntropyLoss()(logits[..., :-1, :].view(-1, 55296), labels[..., 1:].view(-1))
        # loss.shape: [268]
        # loss:
        # tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 9.5892e-01, 2.7711e+00, 4.6718e-01, 4.0682e+00, 2.8885e-01,
        #         2.1138e-02, 1.0128e-01, 5.2179e+00, 2.0874e+00, 9.8119e-02, 3.6435e-03,
        #         1.1197e-01, 1.8104e+01, 1.6244e-01, 7.7388e-03, 6.0321e-01, 7.8061e+00,
        #         3.0438e+00, 3.9255e-02, 4.1152e-02, 7.9409e-01, 3.4647e+00, 5.4830e-02,
        #         2.0371e+00, 4.1411e+00, 4.5448e-02, 1.7236e+01, 9.2835e-03, 2.9990e-03,
        #         3.7191e-01, 1.2440e+00, 2.9448e-02, 2.9886e-02, 7.8816e+00, 9.9238e+00,
        #         3.5825e+00, 1.2797e+00, 2.6501e+00, 9.3011e-04, 3.5373e-02, 1.5068e+01,
        #         1.0693e-02, 1.9498e-03, 3.5507e-01, 1.0780e+00, 2.8248e-02, 5.5362e-02,
        #         7.3778e+00, 2.3909e+00, 5.4721e+00, 3.0809e+00, 2.9763e+00, 2.6406e-02,
        #         1.6001e+01, 1.1255e-02, 2.0288e-03, 2.7162e-01, 4.2510e+00, 4.7529e+00,
        #         1.9535e+00, 2.4820e+00, 1.1472e+00, 9.6600e+00, 8.1138e-02, 1.6678e-01,
        #         3.7056e-02, 3.9047e-02, 4.0253e-01, 5.3785e+00, 2.7608e+00, 4.1660e+00,
        #         2.3823e-01, 2.1440e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 1.0687e+00, 2.6204e+00, 3.1673e-01, 3.1881e-03, 4.2877e+00,
        #         6.6413e+00, 5.3095e+00, 5.6561e+00, 4.7591e-01, 1.0123e+00, 1.0539e+01,
        #         3.0015e+00, 4.7216e+00, 1.7298e-01, 1.7987e+00, 2.3185e-01, 6.0236e+00,
        #         8.8823e-01, 2.0735e-01, 1.8488e-01, 3.2653e+00, 3.5877e+00, 3.8219e+00,
        #         1.1851e-01, 3.9474e+00, 2.6956e+00, 1.5285e+00, 5.3813e+00, 1.6141e+00,
        #         1.8201e+00, 3.8696e+00, 1.8933e+00, 6.6105e-02, 1.4400e+00, 3.7698e+00,
        #         1.2937e+00, 1.4470e-01, 1.1642e-02, 5.5292e-01, 5.2854e+00, 1.1605e+00,
        #         4.1926e-01, 7.4523e-03, 4.2066e+00, 1.9259e+00, 1.6983e+00, 2.9249e+00,
        #         4.5602e+00, 7.5010e+00, 3.3285e-01, 3.1266e-01, 3.5582e-01, 9.3001e-01,
        #         1.2459e-01, 1.4658e+00, 8.7854e-01, 1.5105e-01, 3.3182e-03, 2.1126e-01,
        #         4.8465e-01, 8.4453e-01, 4.5693e-02, 3.4207e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], device='cuda:0',
        #        grad_fn=<NllLossBackward0>)
        # =======================================================================================================

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] =len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
