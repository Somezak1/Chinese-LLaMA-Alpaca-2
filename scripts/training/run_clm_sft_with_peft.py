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

# 该代码的运行参数为
# torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
#     --deepspeed ds_zero2_no_offload.json \
#     --model_name_or_path /data1/csw_model_weights/chinese-llama-2-13b \
#     --tokenizer_name_or_path /data1/csw_model_weights/chinese-llama-2-13b \
#     --dataset_dir ../../data \
#     --per_device_train_batch_size 2 \
#     --do_train \
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
#     --load_in_kbits

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
        # model_args.tokenizer_name_or_path: '/data/model_weights/chinese-llama-plus-7b-official'
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

        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("Training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))

    if training_args.do_eval:
        # training_args.do_eval: False
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
        # lora_rank: 8
        lora_dropout = training_args.lora_dropout
        # lora_dropout: 0.05
        lora_alpha = training_args.lora_alpha
        # lora_alpha: 32.0
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

        # batch_size=2
        # 训练的第一个step, 其inputs由2个样本的input_ids, labels, attention_mask组成, 三个元素的形状都为(2, 97)
        # inputs['attention_mask']:
        # tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False],
        #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True]])
        # inputs['labels']:
        # tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100, 29871, 40722, 31305, 30210, 32719, 30573,
        #          32029, 42062, 34201, 30267,     2,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100],
        #         [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100, 29871, 30392, 30210, 30214, 29954,
        #           7982, 35894, 35587, 37145, 36977, 34886, 38547, 34886, 36977, 32204,
        #          37559, 30267, 32269, 32039, 44049, 36977, 32326, 30893, 32019, 32968,
        #          30210, 38547, 32333, 30210, 30267, 32070, 38547, 30785, 29954,  7982,
        #          35894, 35587, 32244, 34886, 36977, 32204, 36444, 33690, 30214, 30847,
        #          33155, 30214, 32772, 31391, 32417, 30267,     2]])
        # inputs['input_ids']:
        # tensor([[    1, 13866,   338,   385, 15278,   393, 16612,   263,  3414, 29889,
        #          14350,   263,  2933,   393,  7128,  2486,  1614,  2167,   278,  2009,
        #          29889,    13,    13,  2277, 29937,  2799,  4080, 29901,    13, 33347,
        #          31999, 30495, 37121, 30503, 43423, 30210, 40722, 31305, 30210, 32719,
        #          30267,    13, 29941, 34201, 30503, 29946, 34201,    13,    13,  2277,
        #          29937, 13291, 29901, 29871, 29871, 40722, 31305, 30210, 32719, 30573,
        #          32029, 42062, 34201, 30267,     2, 49953, 49953, 49953, 49953, 49953,
        #          49953, 49953, 49953, 49953, 49953, 49953, 49953, 49953, 49953, 49953,
        #          49953, 49953, 49953, 49953, 49953, 49953, 49953, 49953, 49953, 49953,
        #          49953, 49953, 49953, 49953, 49953, 49953, 49953],
        #         [    1, 13866,   338,   385, 15278,   393, 16612,   263,  3414, 29889,
        #          14350,   263,  2933,   393,  7128,  2486,  1614,  2167,   278,  2009,
        #          29889,    13,    13,  2277, 29937,  2799,  4080, 29901,    13, 29954,
        #           7982, 35894, 35587, 30815, 34886, 37559, 32027, 30882,    13,    13,
        #           2277, 29937, 13291, 29901, 29871, 29871, 30392, 30210, 30214, 29954,
        #           7982, 35894, 35587, 37145, 36977, 34886, 38547, 34886, 36977, 32204,
        #          37559, 30267, 32269, 32039, 44049, 36977, 32326, 30893, 32019, 32968,
        #          30210, 38547, 32333, 30210, 30267, 32070, 38547, 30785, 29954,  7982,
        #          35894, 35587, 32244, 34886, 36977, 32204, 36444, 33690, 30214, 30847,
        #          33155, 30214, 32772, 31391, 32417, 30267,     2]])

        # tokenizer.decode(inputs['input_ids'][0]):
        # '<s> Below is an instruction that describes a task. Write a response that appropriately completes the
        # request.\n\n### Instruction:\n计算给定长度和宽度的矩形的面积。\n3厘米和4厘米\n\n### Response:  矩形的面积为12平
        # 方厘米。</s> [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
        # [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

        # tokenizer.decode(inputs['input_ids'][1]):
        # '<s> Below is an instruction that describes a task. Write a response that appropriately completes the
        # request.\n\n### Instruction:\nGPT-3模型能识别物体吗？\n\n### Response:  是的，GPT-3模型可以通过图像识别算法识
        # 别图像中的物体。这是通过标记图像数据集进行训练的算法实现的。这些算法使GPT-3模型能够识别图像中的特定对象，如猫，狗或汽车。</s>'

        # 训练的第二个step, 其inputs由2个样本的input_ids, labels, attention_mask组成, 三个元素的形状都为(2, 84)
        # 可以看出, 由于collator的存在, 每个batch中的样本总是按当前batch中最长的那个样本长度来padding

        # class LlamaForCausalLM(LlamaPreTrainedModel):
        #     def __init__(self, config):
        #         super().__init__(config)
        #         self.model = LlamaModel(config)
        #
        #         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #
        #         # Initialize weights and apply final processing
        #         self.post_init()
        #
        #    ......
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
        #         >>> prompt = "Hey, are you consciours? Can you talk to me?"
        #         >>> inputs = tokenizer(prompt, return_tensors="pt")
        #
        #         >>> # Generate
        #         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        #         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #         "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
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
        #             input_ids=input_ids,  # 第一个step的size为(2, 97)
        #             attention_mask=attention_mask,  # 第一个step的size为(2, 97)
        #             position_ids=position_ids,  # None
        #             past_key_values=past_key_values,  # None
        #             inputs_embeds=inputs_embeds,  # None
        #             use_cache=use_cache,  # None
        #             output_attentions=output_attentions,  # False
        #             output_hidden_states=output_hidden_states,  # False
        #             return_dict=return_dict,  # True
        #         )
        #
        #         hidden_states = outputs[0]  # hidden_states.size(): (2, 97, 4096)
        #         logits = self.lm_head(hidden_states)  # logits.size(): (2, 97, 49954)
        #
        #         loss = None
        #         if labels is not None:
        #             # Shift so that tokens < n predict n
        #             shift_logits = logits[..., :-1, :].contiguous()  # shift_logits.size(): (2, 96, 49954)
        #             shift_labels = labels[..., 1:].contiguous()  # shift_labels.size(): (2, 96)
        #             # Flatten the tokens
        #             loss_fct = CrossEntropyLoss()
        #             shift_logits = shift_logits.view(-1, self.config.vocab_size)  # shift_logits.size(): (192, 49954)
        #             shift_labels = shift_labels.view(-1)  # shift_labels.size(): (192, )
        #             # Enable model parallelism
        #             shift_labels = shift_labels.to(shift_logits.device)
        #             loss = loss_fct(shift_logits, shift_labels)  # labels中-100的地方损失为0
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

        # 上面如果CrossEntropyLoss()定义时reduce=False, 则对第一个step数据计算得到的loss.view(2, 97)得到如下损失矩阵
        # tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.4434e+00,
        #          2.9238e+00, 1.2901e-02, 7.6855e-01, 2.3486e-01, 1.7949e+00, 5.7812e+00,
        #          1.2822e+00, 6.5613e-02, 8.8770e-01, 1.2094e+01, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        #         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #          0.0000e+00, 0.0000e+00, 5.5703e+00, 2.8789e+00, 1.1992e+00, 1.5488e+00,
        #          1.6680e+00, 2.5406e-02, 3.9948e-02, 1.5654e+00, 6.5898e+00, 2.4414e+00,
        #          4.7021e-01, 7.8086e+00, 9.1992e-01, 3.9629e+00, 6.0010e-01, 3.5547e-01,
        #          2.3682e-01, 6.0547e+00, 3.7734e+00, 7.0859e+00, 2.0684e+00, 6.3672e+00,
        #          3.8477e+00, 3.8633e+00, 1.3262e+00, 2.0488e+00, 3.6816e+00, 5.0781e+00,
        #          4.4531e-01, 1.9812e-01, 5.6523e+00, 2.4805e+00, 6.2734e+00, 1.6045e+00,
        #          2.5620e-02, 3.6041e-02, 1.4746e+00, 1.2236e+00, 1.7871e+00, 1.5029e+00,
        #          2.1509e-01, 3.8438e+00, 2.4141e+00, 1.2256e+00, 2.1367e+00, 3.5195e+00,
        #          1.9746e+00, 8.9893e-01, 3.6602e+00, 2.0605e+00, 4.4189e-01, 1.3930e+01]],
        #        device='cuda:0', dtype=torch.float16, grad_fn= < ViewBackward0 >)

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
