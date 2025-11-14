"""Wrapper for runpod-python to manage pods and execute commands via SSH."""

import contextlib
import getpass
import io
import os
import time

import paramiko
import requests
import runpod
from dotenv import load_dotenv
from loguru import logger

from voice_orchestrator.constants import BashCommands, ImageNames, TemplateIds
from voice_orchestrator.logging import setup_logging


class Pod:
    """Pod class to execute commands via SSH."""

    # Class variables
    _ssh_user: str | None = None
    _ssh_key_path: str | None = None
    _ssh_passphrase: str | None = None

    def __init__(
            self,
            name: str,
            template_id: str | None = None,
            image_name: str = "runpod/base:0.7.0-ubuntu2404",
            volume_in_gb: int = 50,
            gpu_type_id: str | None = None,
            gpu_count: int | None = None,
            network_volume_id: str | None = None,
    ):
        """
        Initialise Pod class.

        - Spins up pod, if it doesn't exist yet (unique by name)
        - Waits for pod to be SSH ready
        - Gets SSH credentials from environment variables

        :param name: name of the pod (unique identifier)
        :param image_name: name of the docker image to use
        :param gpu_type_id: name of the gpu to use
        :param gpu_count: number of gpus to use
        :param network_volume_id: network volume id to mount to the pod
        """
        setup_logging()
        load_dotenv()

        self.name = name
        self.template_id = template_id
        self.image_name = image_name
        self.volume_in_gb = volume_in_gb
        self.support_public_ip = True
        self.start_ssh = True
        self.gpu_type_id = gpu_type_id
        self.instance_id = "cpu3c-2-4" if gpu_type_id is None else None
        self.gpu_count = gpu_count
        self.network_volume_id = network_volume_id

        self.api_key = os.getenv("RUNPOD_API_KEY")
        runpod.api_key = self.api_key

        self._get_user_ssh()

        # Check if pod exists (unique by name)
        self.pods: list[dict] = []
        if self._pod_exists():
            logger.success("Pod {} found, using existing pod.", self.name)
            self.pod = next(
                (
                pod for pod in self.pods if pod.get('name') == self.name),
                None
            )
            self.id = self.pod["id"] # type: ignore[index]

            self._wait_for_pod()
        else:
            # Otherwise, spin up pod
            logger.info("Spinning up pod: {}", self.name)
            silent = io.StringIO()
            with contextlib.redirect_stdout(silent):
                self.pod = runpod.create_pod(
                    name=self.name,
                    template_id=self.template_id,
                    image_name=self.image_name,
                    volume_in_gb=self.volume_in_gb,
                    support_public_ip=self.support_public_ip,
                    start_ssh=self.start_ssh,
                    gpu_type_id=self.gpu_type_id,
                    instance_id=self.instance_id,
                    cloud_type="SECURE",
                    gpu_count=self.gpu_count,
                    network_volume_id=self.network_volume_id,
                )
                self.id = self.pod["id"] # type: ignore[index]

            self._wait_for_pod()

    @classmethod
    def _get_user_ssh(cls) -> None:
        """
        Retrieve user SSH info or get from class cache.

        :return: None
        """
        if (
                Pod._ssh_user
                and Pod._ssh_key_path
                and Pod._ssh_passphrase
        ):
            return

        # Load defaults
        ssh_key_path = os.getenv("RUNPOD_SSH_KEY_PATH", "~/.ssh/id_ed25519")
        ssh_key_path = os.path.expanduser(ssh_key_path)
        ssh_user = os.getenv("RUNPOD_SSH_USER", "root")

        # Prompt user
        msg = f"Enter passphrase for key {ssh_key_path}: "
        passphrase = getpass.getpass(msg)

        # Store on Pod class (not cls) to guarantee shared state
        Pod._ssh_user = ssh_user
        Pod._ssh_key_path = ssh_key_path
        Pod._ssh_passphrase = passphrase

    def _pod_exists(self) -> bool:
        """Check if a pod with the given name already exists."""
        self.pods = runpod.get_pods()
        if not self.pods:
            return False
        else:
            return any(pod.get("name") == self.name for pod in self.pods)

    def _wait_for_pod(self, timeout: int = 900, interval: int = 1) -> None:
        """
        Wait until the pod is SSH ready.

        :param timeout: maximum time to wait in seconds
        :param interval: time between checks in seconds
        """
        elapsed = 0
        while elapsed < timeout:
            self.public_ip, self.port = self._get_tcp_port()

            if self.public_ip and self.port:
                msg = f"Pod {self.name} available at: {self.public_ip}:{self.port}"
                logger.success(msg)
                return

            time.sleep(interval)
            elapsed += interval

        logger.error(f"Timed out waiting for pod {self.name} to launch.")

    def _get_tcp_port(self) -> tuple[str | None, int | None]:
        """
        Get the public IP and SSH port of the pod.

        :return: server IP and port
        """
        headers = {"Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}"}
        url = f"https://rest.runpod.io/v1/pods/{self.id}"
        resp = requests.get(url, headers=headers)
        data = resp.json()
        ip = data["publicIp"]
        if ip:
            port = data["portMappings"]["22"]
            # For GPU pods, the above is overwritten to udp port
            if self.gpu_count:
                # Runpod assigns tcp port as udp port -1
                port -= 1
        else:
            port = None
        return ip, port

    def _connect_ssh(self) -> paramiko.SSHClient:
        """Create and return a connected Paramiko SSH client."""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=self.public_ip,  # type: ignore[arg-type]
            port=self.port,  # type: ignore[arg-type]
            username=Pod._ssh_user,
            key_filename=Pod._ssh_key_path,
            passphrase=Pod._ssh_passphrase,
        )
        return ssh

    def _write_dotenv(self) -> None:
        """
        Write the exact contents of the local .env file into pod.

        :return: None
        """
        local_env_path = ".env"
        if not os.path.exists(local_env_path):
            logger.error("Local .env file not found.")
            return

        # Read the local .env raw text
        with open(local_env_path, "r") as f:
            env_text = f.read()

        safe_text = env_text.replace("'", "'\"'\"'")

        # Build the command to write .env on the pod
        cmd = (
            "mkdir -p /app && "
            f"printf '%s' '{safe_text}' > /app/.env && "
            "chmod 600 /app/.env"
        )

        self.execute(cmd)

    def execute(self, command: str, stream: bool = False) -> str | None:
        """
        Execute a command on the pod over SSH.

        :param command: command to execute
        :param stream: stream the output of the command to terminal
        :return: output of the command or None if error occurs
        """
        ssh = self._connect_ssh()

        stdin, stdout, stderr = ssh.exec_command(command, get_pty=stream)

        if stream:
            try:
                # Stream stdout
                while not stdout.channel.exit_status_ready():
                    if stdout.channel.recv_ready():
                        chunk = stdout.channel.recv(1024).decode()
                        print(chunk, end="")  # stream to local terminal

                # Read any remaining output
                remainder = stdout.read().decode()
                if remainder:
                    print(remainder, end="")

                # Check for errors
                err = stderr.read().decode()
                if err:
                    print(err)
                    logger.error(f"Command error: {err.strip()}")
                    return None

                return None

            finally:
                ssh.close()

        output = stdout.read().decode()
        error = stderr.read().decode()

        ssh.close()

        if error:
            logger.error(f"Command error: {error.strip()}")
            return None

        return output.strip()

    def kill(self) -> None:
        """
        Kill the pod.

        :return: None
        """
        runpod.terminate_pod(self.id)
        logger.info(f"Pod {self.name} killed.")

class FinetunePod(Pod):
    """Pod class to manage finetuning GPU pod."""

    def __init__(
            self,
            gpu_type_id: str,
            name: str = "voice-finetune",
            template_id: str = TemplateIds.FINETUNE,
            image_name: str = ImageNames.FINETUNE,
            volume_in_gb: int = 50,
            gpu_count: int = 1,
    ):
        """
        Initialise finetuning pod.

        :param gpu_type_id: id of the gpu to use
        :param name: name of the pod
        :param template_id: id of the template to use
        :param gpu_count: number of gpus to use
        """
        super().__init__(
            name=name,
            template_id=template_id,
            image_name=image_name,
            volume_in_gb=volume_in_gb,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
        )

        self._write_dotenv()

    def finetune(self, config_path: str) -> None:
        """
        Excecute finetuning command on the pod.

        :param config_path: path to finetune config file
        :return: None
        """
        cmd = "&&".join(
            [
                BashCommands.GO_TO_APP,
                BashCommands.ACTIVATE,
                BashCommands.FINETUNE + f" {config_path}",
            ]
        )
        self.execute(cmd, stream=True)

class InferencePod(Pod):
    """Pod class to manage inference GPU pod."""

    def __init__(
            self,
            gpu_type_id: str,
            name: str = "voice-inference",
            template_id: str = TemplateIds.INFERENCE,
            image_name: str = ImageNames.INFERENCE,
            volume_in_gb: int = 50,
            gpu_count: int = 1,
    ):
        """
        Initialise inference pod.

        :param gpu_type_id: id of the gpu to use
        :param name: name of the pod
        :param template_id: id of the template to use
        :param gpu_count: number of gpus to use
        """
        super().__init__(
            name=name,
            template_id=template_id,
            image_name=image_name,
            volume_in_gb=volume_in_gb,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
        )

        self._write_dotenv()

    def infer(self, config_path: str) -> None:
        """
        Execute inference command on the pod.

        :param config_path: path to finetune config file
        :return: None
        """
        cmd = "&&".join(
            [
                BashCommands.GO_TO_APP,
                BashCommands.ACTIVATE,
                BashCommands.INFERENCE + f" {config_path}",
            ]
        )
        self.execute(cmd, stream=True)
