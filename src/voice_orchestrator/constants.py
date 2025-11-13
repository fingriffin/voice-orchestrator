"""Constants for the VOICE orchestration."""

class TemplateIds:
    """RunPod template IDs."""

    FINETUNE = "eziymt38z4"
    INFERENCE = "lwox0565zs"

class ImageNames:
    """RunPod image names."""

    FINETUNE = "ghcr.io/fingriffin/voice-finetune:latest" # Private

class ConfigTypes:
    """Config types for experiment tracking."""

    MASTER_CONFIG = "MasterConfig"
    SUB_CONFIGS = {
        "finetune": "FinetuneConfig",
        "inference": "InferenceConfig",
    }
