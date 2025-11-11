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
