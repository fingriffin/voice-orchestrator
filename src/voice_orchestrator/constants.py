"""Constants for the VOICE orchestration."""

class BashCommands:
    """Bash commands used in VOICE orchestration."""

    GO_TO_APP = "cd .. && cd app"
    ACTIVATE = "source .venv/bin/activate"
    FINETUNE = "finetune"
    INFERENCE = "infer"
    ANALYZE = "analyze"

class TemplateIds:
    """RunPod template IDs."""

    FINETUNE = "eziymt38z4"
    INFERENCE = "lwox0565zs"
    ANALYZE = "ufld0ha15b"

class ImageNames:
    """RunPod image names."""

    FINETUNE = "ghcr.io/fingriffin/voice-finetune:latest" # Private
    INFERENCE = "ghcr.io/fingriffin/voice-inference:latest" # Private
    ANALYZE = "ghcr.io/fingriffin/style-bench:latest" # Private

class ConfigTypes:
    """Config types for experiment tracking."""

    MASTER_CONFIG = "MasterConfig"
    SUB_CONFIGS = {
        "finetune": "FinetuneConfig",
        "inference": "InferenceConfig",
        "analyze": "AnalyzeConfig",
    }

class Misc:
    """Miscellaneous constants."""

    SSH_TCP_PORT = "22/tcp" # Runpod SSH port with TCP protocol
