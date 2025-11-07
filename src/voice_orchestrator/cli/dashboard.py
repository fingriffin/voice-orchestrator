"""CLI command to open ZenML dashboard via SSH tunnel."""

import click

from voice_orchestrator.logging import setup_logging
from voice_orchestrator.zenml import connect_to_zenml_server


@click.command()
def main() -> None:
    """
    Spin up ZenML host pod and open dashboard via SSH tunnel.

    :return: None
    """
    # Setup logging and get environment variables
    setup_logging()

    # Start zenml server
    connect_to_zenml_server(launch=True)
