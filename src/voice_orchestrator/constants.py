"""Constants for the VOICE orchestration."""

class ShellCommands:
    """Shell commands used on RunPod instances."""

    ZENML_HOST_STARTUP = (
        'export PATH="/runpod-volume/.local/bin:$PATH" && '
        'export ZENML_CONFIG_PATH="/runpod-volume/.config/zenml" &&'
        'cd /runpod-volume && '
        'source .venv/bin/activate && '
        'zenml up'
    )

class TemplateIds:
    """RunPod template IDs."""

    FINETUNE = "eziymt38z4"
    INFERENCE = "lwox0565zs"

class ImageNames:
    """RunPod image names."""

    CPU = "runpod/base:0.7.0-ubuntu2404"
    FINETUNE = "ghcr.io/fingriffin/voice-finetune:latest" # Private
