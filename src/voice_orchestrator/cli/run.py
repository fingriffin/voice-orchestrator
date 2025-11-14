"""CLI to run VOICE orchestrator with a specified configuration."""


import click
from dotenv import load_dotenv
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

    #
    # # Prepare wandb run
    # run = WandbRun(config=config, config_path=config_path)
    #
    # # Log config artifacts (including sub-configs) to wandb
    # run.log_config_artifacts()
    #
    # # Spin up finetuning pod
    # finetune_pod = FinetunePod(
    #     gpu_type_id=config.gpu_type_finetune, # type: ignore[arg-type]
    #     gpu_count=config.finetune.gpus,
    # )
    #
    # # Run finetuning job with saved finetune config artifact
    # finetune_config_uri = run.get_config_uri(
    #     config_type=ConfigTypes.SUB_CONFIGS["finetune"]
    # )
    # finetune_pod.finetune(finetune_config_uri)
    #
    # finetune_pod.kill()
    #
    # # Spin up inference pod
    # inference_pod = InferencePod(
    #     gpu_type_id=config.gpu_type_inference, # type: ignore[arg-type]
    #     gpu_count=config.inference.gpus,
    # )
    #
    # # Run inference job with saved inference config artifact
    # inference_config_uri = run.get_config_uri(
    #     config_type=ConfigTypes.SUB_CONFIGS["inference"]
    # )
    # inference_pod.infer(inference_config_uri)
    #
    # inference_pod.kill()
    #
    # run.finish()
