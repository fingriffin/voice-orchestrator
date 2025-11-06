"""CLI command to open ZenML dashboard via SSH tunnel."""

import os
import time
import webbrowser

import click
from dotenv import load_dotenv
from loguru import logger
from sshtunnel import SSHTunnelForwarder

from voice_orchestrator.logging import setup_logging
from voice_orchestrator.runpod import ZenMLHostPod


@click.command()
def main() -> None:
    """
    Spin up ZenML host pod and open dashboard via SSH tunnel.

    :return: None
    """
    # Setup logging and get environment variables
    setup_logging()

    # Spin up zenml host pod (and start server)
    zenml_host = ZenMLHostPod()

    load_dotenv()
    ssh_host = zenml_host.public_ip
    zenml_host_port = zenml_host.port
    zenml_port = int(os.getenv("ZENML_PORT")) # type: ignore[arg-type]
    if not all([ssh_host, zenml_host.ssh_key_path]):
        raise click.ClickException("Missing SSH details in .env")

    # Create SSH tunnel
    logger.info("Creating SSH tunnel to {}", ssh_host)
    tunnel = SSHTunnelForwarder(
        (ssh_host, zenml_host_port),
        ssh_username=zenml_host.ssh_user,
        ssh_private_key=zenml_host.ssh_key_path,
        ssh_private_key_password=zenml_host.passphrase,
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
