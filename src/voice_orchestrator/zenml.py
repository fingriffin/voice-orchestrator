"""Module for ZenML functionality in the VOICE orchestrator."""

import os
import time
import webbrowser

from dotenv import load_dotenv
from loguru import logger
from sshtunnel import SSHTunnelForwarder
from zenml.client import Client
from zenml.config.global_config import GlobalConfiguration
from zenml.zen_stores.rest_zen_store import RestZenStoreConfiguration

from voice_orchestrator.runpod import ZenMLHostPod

LOCAL_HOST = "127.0.0.1"

def connect_to_zenml_server(launch: bool = False) -> Client:
    """
    Connect to the remote ZenML server.

    - Spins up zenml host pod (if not already running)
    - Starts zenml server on host pod using SSH
    - Creates SSH tunnel to the zenml server

    :param launch: whether to launch ZenML in a browser, if not returns the client
    :return: ZenML client connected to the remote server
    """
    # Spin up zenml host pod (and start server)
    zenml_host = ZenMLHostPod()

    load_dotenv()
    ssh_host = zenml_host.public_ip
    zenml_host_port = zenml_host.port
    zenml_port = int(os.getenv("ZENML_PORT"))  # type: ignore[arg-type]

    # Create SSH tunnel
    logger.info("Creating SSH tunnel to {}", ssh_host)
    tunnel = SSHTunnelForwarder(
        (ssh_host, zenml_host_port),
        ssh_username=zenml_host.ssh_user,
        ssh_private_key=zenml_host.ssh_key_path,
        ssh_private_key_password=zenml_host.passphrase,
        remote_bind_address=(LOCAL_HOST, zenml_port),
        local_bind_address=(LOCAL_HOST, zenml_port),
    )

    server_url = f"http://{LOCAL_HOST}:{zenml_port}"

    tunnel.start()
    logger.success("Tunnel open: {}", server_url)

    if launch:
        webbrowser.open(server_url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            tunnel.stop()
    else:
        # Create zenml client
        cfg = RestZenStoreConfiguration(url=server_url, verify_ssl=False)

        gc = GlobalConfiguration()
        gc.set_store(config=cfg)

        return Client()
