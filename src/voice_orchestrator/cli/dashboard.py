"""CLI command to open ZenML dashboard via SSH tunnel."""

import os
import time
import webbrowser
from getpass import getpass

import click
from dotenv import load_dotenv
from loguru import logger
from sshtunnel import SSHTunnelForwarder

from voice_orchestrator.logging import setup_logging


@click.command()
def main() -> None:
    """
    Open ZenML dashboard via SSH tunnel.

    :return: None
    """
    # Setup logging and get environment variables
    setup_logging()

    load_dotenv()
    ssh_host = os.getenv("RUNPOD_SSH_HOST")
    ssh_port = int(os.getenv("RUNPOD_SSH_PORT", "22"))
    ssh_user = os.getenv("RUNPOD_SSH_USER", "root")
    ssh_key = os.getenv("RUNPOD_SSH_KEY_PATH")
    zenml_port = int(os.getenv("ZENML_PORT", "8237"))
    if not all([ssh_host, ssh_key]):
        raise click.ClickException("Missing SSH details in .env")

    # Create SSH tunnel
    logger.info("Creating SSH tunnel to {}", ssh_host)
    tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_private_key=ssh_key,
        ssh_private_key_password=getpass(f"Enter passphrase for key {ssh_key}: "),
        remote_bind_address=("127.0.0.1", zenml_port),
        local_bind_address=("127.0.0.1", zenml_port),
    )

    tunnel.start()
    logger.success(f"Tunnel open: http://127.0.0.1:{zenml_port}")
    webbrowser.open(f"http://127.0.0.1:{zenml_port}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tunnel.stop()
