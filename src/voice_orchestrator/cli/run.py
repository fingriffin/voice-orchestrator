"""CLI to run VOICE orchestrator with a specified configuration."""


import click
from dotenv import load_dotenv
from loguru import logger

from voice_orchestrator.config import load_master_config
from voice_orchestrator.constants import ASCII_LOGO, ConfigTypes
from voice_orchestrator.errors import PodCommandError, PodInterrupted
from voice_orchestrator.logging import setup_logging
from voice_orchestrator.runpod import AnalyzePod, FinetunePod, InferencePod, Pod
from voice_orchestrator.wandb import WandbRun


@click.command()
@click.argument("config_path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.option("--label", help="Label for orchestration, for running parallel experiments")
def main(
    config_path: str,
    log_level: str,
    log_file: str | None = None,
    label: str | None = None,
) -> None:
    """
    Run VOICE orchestrator with the specified configuration.

    :param config_path: Path to the configuration file
    :param log_level: Logging level
    :param log_file: Log file path
    :param label: Label for orchestration, for running parallel experiments
    :return: None
    """
    # Setup logging
    load_dotenv()
    setup_logging(level=log_level, log_file=log_file)

    # Print ASCII logo
    print(ASCII_LOGO)

    # Get user SSH
    Pod._get_user_ssh()

    # Load config
    try:
        logger.info("Loading config from {}", config_path)
        config = load_master_config(config_path, label)
        logger.success("Config loaded successfully!")
        print("Current configuration:")
        print(config.model_dump_json(indent=2))
        print("")
    except Exception as e:
        logger.error("Failed to load config: {}", e)
        raise


    # Prepare wandb run
    run = WandbRun(config=config, config_path=config_path)

    # Log config artifacts (including sub-configs) to wandb
    run.log_config_artifacts()

    # End run to be continued via finetuning and inference pods
    run.finish()

    # Spin up finetuning pod
    finetune_pod = FinetunePod(
        gpu_type_id=config.gpu_type_finetune, # type: ignore[arg-type]
        gpu_count=config.finetune.gpus,
        volume_in_gb=config.volume_in_gb_finetune,
        container_disk_in_gb=config.container_disk_in_gb_finetune,
        label=label,
    )

    # Run finetuning job with saved finetune config artifact
    finetune_config_uri = run.get_config_uri(
        config_type=ConfigTypes.SUB_CONFIGS["finetune"]
    )
    try:
        finetune_pod.finetune(
            config_path=finetune_config_uri,
            wandb_run_id=run.id,
        )
    except (PodCommandError, PodInterrupted):
        finetune_pod.kill()
        raise

    finetune_pod.kill()

    # Spin up inference pod
    inference_pod = InferencePod(
        gpu_type_id=config.gpu_type_inference, # type: ignore[arg-type]
        gpu_count=config.inference.gpus,
        volume_in_gb=config.volume_in_gb_inference,
        container_disk_in_gb=config.container_disk_in_gb_inference,
        label=label,
    )

    # Run inference job with saved inference config artifact
    inference_config_uri = run.get_config_uri(
        config_type=ConfigTypes.SUB_CONFIGS["inference"]
    )
    try:
        inference_pod.infer(
            config_path=inference_config_uri,
            wandb_run_id=run.id,
        )
        inference_pod.infer(
            config_path=inference_config_uri,
            wandb_run_id=run.id,
            base_model=config.finetune.model_name,
        )
    except (PodCommandError, PodInterrupted):
        inference_pod.kill()
        raise

    inference_pod.kill()

    # Spin up analyze pod
    analyze_pod = AnalyzePod(label=label)

    # Run analysis job with saved analyze config artifact
    analyze_config_uri = run.get_config_uri(
        config_type=ConfigTypes.SUB_CONFIGS["analyze"]
    )
    try:
        analyze_pod.analyze(
            config_path=analyze_config_uri,
            wandb_run_id=run.id,
        )
    except (PodCommandError, PodInterrupted):
        analyze_pod.kill()
        raise

    analyze_pod.kill()
