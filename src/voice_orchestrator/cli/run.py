"""CLI to run VOICE orchestrator with a specified configuration."""

import os
from datetime import datetime

import click
import wandb
from dotenv import load_dotenv
from loguru import logger

from voice_orchestrator.config import load_master_config, load_wandb_config
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
    """
    Run VOICE orchestrator with the specified configuration.

    :param config_path: Path to the configuration file
    :param log_level: Logging level
    :param log_file: Log file path
    :return: None
    """
    # Setup logging
    load_dotenv()
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

    # Prepare wandb run
    os.environ["WANDB_DIR"] = "/tmp" # Avoid flooding repo with wandb runs
    timestamp = datetime.now().strftime("%H-%M-%d-%m-%y")
    name = config.name + "-" + timestamp
    wandb_config = load_wandb_config(config_path)

    # Start wandb run
    wandb_run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        name=name,
        config=wandb_config,
    )

    # Log config file as artifact for reproducibility
    cfg_artifact = wandb.Artifact(
        name=f"{name}_config",
        type="MasterConfig",
        description="YAML config used for this run"
    )
    cfg_artifact.add_file(config_path)
    wandb_run.log_artifact(cfg_artifact).wait()

    wandb_run.finish()
