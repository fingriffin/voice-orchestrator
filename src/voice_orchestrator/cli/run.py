"""CLI to run VOICE orchestrator with a specified configuration."""

import click
from loguru import logger

from voice_orchestrator.config import load_master_config
from voice_orchestrator.logging import setup_logging
from voice_orchestrator.runpod import FinetuningPod


@click.command()
@click.argument("config_path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
def main(
    config_path: str,
    log_level: str,
    log_file: str | None = None,
) -> None:
    """
    Run VOICE orchestrator with the specified configuration.

    :param config_path: Path to the configuration file
    :param log_level: Logging level
    :param log_file: Log file path
    :return: None
    """
    # Setup logging
    setup_logging(level=log_level, log_file=log_file)

    # Load config
    try:
        logger.info("Loading config from {}", config_path)
        config = load_master_config(config_path)
        logger.success("Config loaded successfully!")
        print("Current configuration:")
        print(config.model_dump_json(indent=2))
        print("")
    except Exception as e:
        logger.error("Failed to load config: {}", e)
        raise

    # Spin up finetuning pod
    finetuning_pod = FinetuningPod(gpu_type_id=config.gpu_type_finetune) # type: ignore[arg-type]

    __ = finetuning_pod # Placeholder for linter
