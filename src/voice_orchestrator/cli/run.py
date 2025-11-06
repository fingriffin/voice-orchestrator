"""CLI to run VOICE orchestrator with a specified configuration."""

import click
from loguru import logger

from voice_orchestrator.config import load_master_config
from voice_orchestrator.logging import setup_logging


@click.command()
@click.argument("config_path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
def main(
    config_path: str,
    log_level: str,
    log_file: str | None = None,
) -> None:
    """Run VOICE orchestrator with the specified configuration."""
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
