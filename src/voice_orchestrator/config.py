"""Configuration management for unified master configuration (finetuning + inference)."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

from voice_orchestrator.constants import Misc


class TRLConfig(BaseModel):
    """Configuration for fine-tuning with TRL."""

    beta: float = Field(
        0.001,
        description="RL beta hyperparameter",
    )
    max_completion_length: int = Field(
        2048,
        description="Maximum token length of completions during TRL"
    )
    use_vllm: bool = Field(False, description="Whether to use vLLM during training")
    num_generations: int = Field(1, description="Number of generations to sample")
    reward_funcs: list[str] = Field(..., description="List of stylometric rewards to use")
    reward_weights: list[float] = Field(
        ...,
        description="List of weights for stylometric rewards"
    )

    temperature: float = Field(
        0.7,
        description="Temperature for GRPO sampling when calculating advantages"
    )

    log_completions: bool | None = Field(
        False,
        description="Whether to log completions during training"
    )
    num_completions_to_print: int | None = Field(
        None,
        description="Number of completions to print during training"
    )



class FinetuneConfig(BaseModel):
    """Configuration for LoRA/QLoRA finetuning."""

    base_model: Optional[str] = Field(None, description="Name of the model to use")
    seed: int = Field(42, description="Random seed")
    output_dir: str = Field(..., description="Directory to save checkpoints and outputs")
    device_map: str = Field("auto", description="Device map for model loading")

    gpu_type: str = Field("NVIDIA A40", description="GPU type (will be routed upward)")
    gpus: int = Field(1, description="Number of GPUs to use (will be routed upward)")
    volume_in_gb: int = Field(50, description="Volume size in GB (will be routed upward)")
    container_disk_in_gb: int = Field(
        50,
        description="Container size in GB (will be routed upward)"
    )

    adapter: str = Field(..., description="Name of the adapter model to use")
    load_in_8bit: bool = Field(False, description="Load the model from 8bit")
    load_in_4bit: bool = Field(False, description="Load the model from 4bit")
    bf16: bool = Field(False, description="Load the model from BF16")
    fp16: bool = Field(True, description="Load the model from FP16")
    optimizer: str = Field("paged_adamw_32bit", description="Optimizer to use")
    num_epochs: int = Field(3, description="Number of training epochs")
    learning_rate: float = Field(2e-4, description="Learning rate")
    micro_batch_size: int = Field(2, description="Batch size per device")
    sequence_len: int = Field(1024, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(4, description="No. of accumulation steps")
    gradient_checkpointing: bool = Field(False, description="Use gradient checkpointing")
    flash_attention: bool = Field(False, description="Use flash attention if available")

    lora_r: int = Field(8, description="LoRA rank")
    lora_alpha: int = Field(16, description="LoRA alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout")
    lora_target_modules: list[str] |  None = Field(
        None,
        description="List of target modules for LoRA",
    )

    rl: Optional[str] = Field(None, description="Name of RL model to use (e.g. GRPO)")
    trl: Optional[TRLConfig] = Field(
        None,
        description="Optional configuration for TRL"
    )

    tokenizer_config: str | None = Field(None, description="Tokenizer config")
    special_tokens: dict[str, str] | None = Field(None, description="Special tokens dict")

    save_steps: int | float | None = Field(
        0,
        description="When to save model checkpoints",
    )
    save_strategy: str | None = Field("no", description="Saving strategy")
    save_total_limit: int = Field(
        0,
        description="Maximum number of checkpoints to save at one point"
    )
    save_only_model: bool = Field(
        True,
        description="Whether to save only the model",
    )

    datasets: list[dict[str, str]] = Field(
        ...,
        description="Datasets to use"
    )
    test_datasets: list[dict[str, str]] = Field(
        ...,
        description="Validation datasets to use"
    )
    eval_steps: int | float | None = Field(
        None,
        description="How often to run validation, in steps"
    )

    use_wandb: bool = Field(True, description="Whether to use wandb")
    wandb_project: str = Field(
        os.getenv("WANDB_PROJECT"),
        description="wandb project name",
    )
    wandb_entity: str = Field(
        os.getenv("WANDB_ENTITY"),
        description="wandb entity name",
    )
    wandb_watch: str = Field(
        "checkpoint",
        description="When to log model artifact"
    )
    wandb_log_model: str = Field(
        "checkpoint",
        description="When to log model artifact"
    )
    hub_model_id: str = Field(
        ...,
        description="Where to push checkpoints to on HF hub"
    )
    hub_strategy: str = Field(
        "end",
        description="How to push checkpoints to HF hub"
    )

    @field_validator("output_dir")
    def create_output_path(cls, v: str) -> str:
        """Ensure output directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class InferenceConfig(BaseModel):
    """Configuration for performing inference with the finetuned model."""

    model: Optional[str] = Field(None, description="Model name or path")
    test_data: Optional[str] = Field(
        None,
        description="Path to dataset or HF dataset name"
    )

    split: str = Field("test", description="Dataset split to use for inference")
    gpu_type: str = Field("NVIDIA A40", description="GPU type (will be routed upward)")
    gpus: int = Field(1, description="Number of GPUs to use")
    volume_in_gb: int = Field(50, description="Volume size in GB (will be routed upward)")
    container_disk_in_gb: int = Field(
        50,
        description="Container size in GB (will be routed upward)"
    )
    quantization: Optional[str] = Field(
        None, description="Quantization method (e.g. 4bit or 8bit)"
    )
    max_tokens: int = Field(2048, description="Maximum tokens to generate")
    temperature: float = Field(
        0.7,
        description="Temperature for sampling during inference"
    )
    output_file: str = Field(..., description="File path to save inference output")

    @field_validator("output_file")
    def ensure_output_dir_exists(cls, v: str) -> str:
        """Ensure the output directory for inference exists."""
        output_path = Path(v)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return v

class LegomenaConfig(BaseModel):
    """Configuration for legomena calculations with style bench."""

    hapax: bool = True
    dislegomena: bool = True
    trilegomina: bool = True


class RichnessConfig(BaseModel):
    """Configuration for lexical richness calculations with style bench."""

    mattr: bool = True
    ttr: bool = True


class SentimentConfig(BaseModel):
    """Configuration for sentiment analysis with style bench."""

    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    classes: int = 7
    batch_size: int = 64


class LexicalConfig(BaseModel):
    """Configuration for lexical analysis with style bench."""

    richness: RichnessConfig = RichnessConfig()
    word_length: bool = True
    function_words: bool = True
    density: bool = True
    legomena: LegomenaConfig = LegomenaConfig()
    sentiment: SentimentConfig = SentimentConfig()


class ComparisonConfig(BaseModel):
    """Configuration for response comparison analysis with style bench."""

    run_comparison: Optional[bool] = True
    reference_key: Optional[str] = "reference_response"


class DataConfig(BaseModel):
    """Configuration for data analysis with style bench."""

    data_path: Optional[str] = "data/results.json"
    target_key: Optional[str] = "generated_response"
    output_path: Optional[str] = "output/"

    @field_validator("output_path")
    def create_output_path(cls, v: str) -> str:
        """Ensure the output directory for analysis exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class AnalyzeConfig(BaseModel):
    """Configuration for analyzing model outputs with style bench."""

    lexical: LexicalConfig = LexicalConfig()
    data: Optional[DataConfig] = DataConfig()
    compare: Optional[ComparisonConfig] = ComparisonConfig()
    # Can be defined by the user
    experiment_name: Optional[str] = None
    description: Optional[str] = None

class MasterConfig(BaseModel):
    """Master configuration combining finetuning and inference settings."""

    base_model: str = Field(..., description="Base HF model to use")
    name: str = Field(..., description="Resulting adapter name or output dir")

    finetune_gpus: Optional[int] = Field(
        None,
        description="Number of GPUs to use fo finetuning"
    )
    gpu_type_finetune: Optional[str] = Field(None, description="GPU type for finetuning")
    gpu_type_inference: Optional[str] = Field(None, description="GPU type for inference")

    volume_in_gb_finetune: int = Field(50, description="Volume for finetuning")
    volume_in_gb_inference: int = Field(50, description="Volume for inference")

    container_disk_in_gb_finetune: int = Field(
        50,
        description="Container disk for finetuning"
    )
    container_disk_in_gb_inference: int = Field(
        50,
        description="Container disk for inference"
    )

    finetune: FinetuneConfig
    inference: InferenceConfig
    analyze: AnalyzeConfig

    @model_validator(mode="after")
    def route_shared_fields(cls, values: "MasterConfig") -> "MasterConfig":
        """Route shared fields between finetuning and inference configurations."""
        load_dotenv()
        hf_org = os.getenv("HF_ORG")

        base_model = values.base_model
        name = values.name
        merged_name = f"{hf_org}/" + name + "-Merged"

        finetune = values.finetune
        inference = values.inference
        analyze = values.analyze

        if getattr(finetune, "gpu_type", None):
            values.gpu_type_finetune = finetune.gpu_type
            delattr(finetune, "gpu_type")

        if getattr(inference, "gpu_type", None):
            values.gpu_type_inference = inference.gpu_type
            delattr(inference, "gpu_type")

        if getattr(finetune, "volume_in_gb", None):
            values.volume_in_gb_finetune = finetune.volume_in_gb
            delattr(finetune, "volume_in_gb")

        if getattr(inference, "volume_in_gb", None):
            values.volume_in_gb_inference = inference.volume_in_gb
            delattr(inference, "volume_in_gb")

        if getattr(finetune, "gpus", None):
            values.finetune_gpus = finetune.gpus
            delattr(finetune, "gpus")

        if getattr(finetune, "container_disk_in_gb", None):
            values.container_disk_in_gb_finetune = finetune.container_disk_in_gb
            delattr(finetune, "container_disk_in_gb")

        if getattr(inference, "container_disk_in_gb", None):
            values.container_disk_in_gb_inference = inference.container_disk_in_gb
            delattr(inference, "container_disk_in_gb")

        finetune.base_model = base_model
        finetune.output_dir = name

        inference.model = merged_name
        inference.test_data = finetune.datasets[0]["path"]

        analyze.experiment_name = name

        if finetune.load_in_4bit:
            inference.quantization = "4bit"
        elif finetune.load_in_8bit:
            inference.quantization = "8bit"

        return values


def load_master_config(
        config_path: str,
        label: str | None = None
) -> MasterConfig:
    """
    Load master configuration from a YAML file.

    :param config_path: Path to YAML configuration file
    :param label: Optional label to use (for multiple run orchestration)
    :return: master config object
    """
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    config_dict = yaml.safe_load("".join(lines))
    if label:
        config_dict["name"] = f"{config_dict["name"]}-{label}"
    return MasterConfig(**config_dict)

def load_wandb_config(config_path: str) -> dict[str, Any]:
    """
    Load the YAML file at config_path and returns its contents as a dictionary.

    Creates nicer visualisation in the wandb dashboard.

    :param config_path: Path to YAML file.
    :return: Dictionary of the YAML file contents.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    if "name" in config_dict:
        config_dict.pop("name")

    # Renumber keys for clean ordering
    return {
        Misc.NUMBERED_KEYS.get(key,key): val
        for key, val in config_dict.items()
    }
