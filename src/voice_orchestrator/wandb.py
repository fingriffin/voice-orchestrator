"""Utilities for wandb integration."""
import os
import tempfile
from datetime import datetime

import wandb
import yaml
from dotenv import load_dotenv
from wandb import Run

from voice_orchestrator.config import MasterConfig, load_wandb_config
from voice_orchestrator.constants import ConfigTypes


class WandbRun:
    """Class to manage a wandb run for VOICE experiment tracking."""

    def __init__(self, *, config: MasterConfig, config_path: str):
        """
        Initialise a wandb run.

        :param config: pydantic validated master config object
        :param config_path: path to master config file
        """
        self.config = config
        self.config_path = config_path
        self.name: str = ""
        self.run: Run

        self._prepare_run()

    def _prepare_run(self) -> None:
        """
        Prepare a wandb run with the given name + timestamp.

        :return: None
        """
        # Direct wandb to use /tmp directory to avoid flooding repo
        load_dotenv()
        os.environ["WANDB_DIR"] = "/tmp"

        # Make wandb run name unique by appending timestamp
        timestamp = datetime.now().strftime("%H-%M-%d-%m-%y")
        self.name = self.config.name + "-" + timestamp

        # Start run
        wandb_config = load_wandb_config(self.config_path)
        self.run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=self.name,
            config=wandb_config,
        )

    def log_config_artifacts(self) -> None:
        """
        Log master and sub-configs as artifacts to wandb.

        Sub-configs are not saved locally so temp .yaml files are created.

        :return: None
        """
        # Log master config as artifact
        self._log_artifact(
            path=self.config_path,
            config_type=ConfigTypes.MASTER_CONFIG
        )
        # Log sub-configs as artifacts
        for sub in ConfigTypes.SUB_CONFIGS.keys():
            config_dict = getattr(self.config, sub).model_dump()

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fp:
                yaml.dump(config_dict, fp)
                tmp_path = fp.name

            self._log_artifact(path=tmp_path, config_type=ConfigTypes.SUB_CONFIGS[sub])

            os.remove(tmp_path)

    def _log_artifact(self, *, path: str, config_type: str) -> None:
        """
        Log a config file as a wandb artifact.

        :param path: path to config file
        :param config_type: config type stylized e.g. "InferenceConfig"
        :return: None
        """
        artifact_name = self.name + "-" + config_type
        artifact = wandb.Artifact(
            name=artifact_name,
            type=config_type,
        )
        artifact.add_file(path, name=artifact_name + ".yaml")
        self.run.log_artifact(artifact).wait()

    def get_config_uri(self, config_type: str) -> str:
        """
        Get the wandb URI for a logged config artifact.

        :param config_type: config type stylized e.g. "InferenceConfig"
        :return: wandb URI string
        """
        return "/".join(
            [
                os.getenv("WANDB_ENTITY"), # type: ignore[list-item]
                os.getenv("WANDB_PROJECT"), # type: ignore[list-item]
                f"{self.name}-{config_type}:v0",
            ]
        )

    def finish(self) -> None:
        """
        End the wandb run.

        :return: None
        """
        self.run.finish()
