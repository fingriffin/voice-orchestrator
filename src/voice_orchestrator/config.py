"""Configuration management for unified master configuration (finetuning + inference)."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class FinetuneConfig(BaseModel):
    """Configuration for finetuning a model with LoRA/QLoRA adapters."""

    model_name: Optional[str] = Field(None, description="Name of the model to use")
    train_data_path: Optional[str] = Field(None, description="Path to training data")
    output_dir: Optional[str] = Field(None, description="Directory to save model outputs")

    adapter: str = Field(..., description="Name of the adapter to use (e.g. lora)")

    load_in_8bit: bool = Field(False, description="Load model from 8bit")
    load_in_4bit: bool = Field(False, description="Load model from 4bit")
    bf16: bool = Field(False, description="Use BF16 precision")
    fp16: bool = Field(True, description="Use FP16 precision")
    gradient_checkpointing: bool = Field(True, description="Use gradient checkpointing")

    optimizer: str = Field("paged_adamw_32bit", description="Optimizer to use")
    gpu_type: str = Field("NVIDIA A40", description="GPU type (will be routed upward)")
    gpus: int = Field(1, description="Number of GPUs to use")

    epochs: int = Field(3, description="Number of training epochs")
    micro_batch_size: int = Field(2, description="Micro batch size")
    gradient_accumulation_steps: int = Field(4, description="Gradient accumulation steps")
    learning_rate: float = Field(2e-4, description="Learning rate")

    lora_r: int = Field(8, description="LoRA rank")
    lora_alpha: int = Field(16, description="LoRA alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout")

    sequence_len: int = Field(1024, description="Max sequence length")
    device_map: str = Field("auto", description="Device map for model loading")
    flash_attention: bool = Field(False, description="Use flash attention")

    seed: int = Field(42, description="Random seed")
    checkpointing: bool = Field(False, description="Save checkpoints during training")
    push_to_hub: bool = Field(True, description="Push model and checkpoints to HF Hub")
    do_validation: bool = Field(False, description="Run validation during training")
    do_merge: bool = Field(True, description="Merge LoRA weights after training")
    adapter_subfolder: Optional[str] = Field(
        None, description="Adapter subfolder inside adapter repo"
    )

    @field_validator("output_dir")
    def ensure_output_dir_exists(cls, v: Optional[str]) -> Optional[str]:
        """Ensure the output directory for finetuning exists."""
        if v is not None:
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
    quantization: Optional[str] = Field(
        None, description="Quantization method (e.g. 4bit or 8bit)"
    )
    max_tokens: int = Field(2048, description="Maximum tokens to generate")
    output_file: str = Field(..., description="File path to save inference output")

    @field_validator("output_file")
    def ensure_output_dir_exists(cls, v: str) -> str:
        """Ensure the output directory for inference exists."""
        output_path = Path(v)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return v


class MasterConfig(BaseModel):
    """Master configuration combining finetuning and inference settings."""

    base_model: str = Field(..., description="Base HF model to use")
    data_path: str = Field(..., description="Training data path or dataset name")
    name: str = Field(..., description="Resulting adapter name or output dir")

    gpu_type_finetune: Optional[str] = Field(None, description="GPU type for finetuning")
    gpu_type_inference: Optional[str] = Field(None, description="GPU type for inference")

    finetune: FinetuneConfig
    inference: InferenceConfig

    @model_validator(mode="after")
    def route_shared_fields(cls, values: "MasterConfig") -> "MasterConfig":
        """Route shared fields between finetuning and inference configurations."""
        base_model = values.base_model
        data_path = values.data_path
        name = values.name
        merged_name = name + "-Merged"

        finetune = values.finetune
        inference = values.inference

        if getattr(finetune, "gpu_type", None):
            values.gpu_type_finetune = finetune.gpu_type
            delattr(finetune, "gpu_type")

        if getattr(inference, "gpu_type", None):
            values.gpu_type_inference = inference.gpu_type
            delattr(inference, "gpu_type")

        finetune.model_name = base_model
        finetune.train_data_path = data_path
        finetune.output_dir = name

        inference.model = merged_name
        inference.test_data = data_path

        if finetune.load_in_4bit:
            inference.quantization = "4bit"
        elif finetune.load_in_8bit:
            inference.quantization = "8bit"

        return values


def load_master_config(config_path: str) -> MasterConfig:
    """Load master configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    config_dict = yaml.safe_load("".join(lines))
    return MasterConfig(**config_dict)
