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

from voice_orchestrator.constants import ImageNames, ShellCommands, TemplateIds
from voice_orchestrator.logging import setup_logging


class Pod:
    """Pod class to execute commands via SSH."""

    def __init__(
            self,
            name: str,
            template_id: str | None = None,
            image_name: str = "runpod/base:0.7.0-ubuntu2404",
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
        self.support_public_ip = True
        self.start_ssh = True
        self.gpu_type_id = gpu_type_id
        self.instance_id = "cpu3c-2-4" if gpu_type_id is None else None
        self.gpu_count = gpu_count
        self.network_volume_id = network_volume_id

        self.api_key = os.getenv("RUNPOD_API_KEY")
        runpod.api_key = self.api_key

        self._get_user_ssh() # TODO: Make this a class level method?

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

    def _get_user_ssh(self) -> None:
        """Retrieve user SSH info."""
        ssh_key_path = os.getenv("RUNPOD_SSH_KEY_PATH", "~/.ssh/id_ed25519")
        self.ssh_key_path = os.path.expanduser(ssh_key_path)
        self.ssh_user = os.getenv("RUNPOD_SSH_USER", "root")
        msg = f"Enter passphrase for key {self.ssh_key_path}: "
        self.passphrase = getpass.getpass(msg)

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
            username=self.ssh_user,
            key_filename=self.ssh_key_path,
            passphrase=self.passphrase,
        )
        return ssh

    def execute(self, command: str) -> str | None:
        """
        Execute a command on the pod over SSH.

        :param command: command to execute
        :return: output of the command or None if error occurs
        """
        ssh = self._connect_ssh()

        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()

        ssh.close()

        if error:
            logger.error(f"Command error: {error.strip()}")
            return None

        return str(output.strip())



class ZenMLHostPod(Pod):
    """Pod class to manage ZenML host CPU pod."""

    def __init__(
            self,
            name: str = "zenml-host",
            image_name: str = ImageNames.CPU,
            network_volume_id: str | None = "kh451m6un6",
    ):
        """
        Initialise ZenMLHostPod class.

        Starts ZenML server on the pod at startup.

        :param name: name of the pod
        :param network_volume_id: network volume id to mount to the pod
        """
        super().__init__(
            name=name,
            image_name=image_name,
            network_volume_id=network_volume_id,
        )

        # Up zenml server
        self._up_server()

    def _up_server(self) -> None:
        """
        Start ZenML server on the pod.

        :return: None
        """
        try:
            logger.info("Starting ZenML server on pod...")
            output = self.execute(
                command=ShellCommands.ZENML_HOST_STARTUP
            )

            if output:
                logger.success("ZenML server started successfully.")

        except Exception as e:
            logger.exception(f"Failed to start ZenML server: {e}")

class FinetuningPod(Pod):
    """Pod class to manage finetuning GPU pod."""

    def __init__(
            self,
            gpu_type_id: str,
            name: str = "voice-finetune",
            template_id: str = TemplateIds.FINETUNE,
            image_name: str = ImageNames.FINETUNE,
            gpu_count: int = 1,
    ):
        """
        Initialise finetuning pod.

        Sends ssh credentials at startup.

        :param gpu_type_id: id of the gpu to use
        :param name: name of the pod
        :param template_id: id of the template to use
        :param gpu_count: number of gpus to use
        """
        super().__init__(
            name=name,
            template_id=template_id,
            image_name=image_name,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
        )

        self._send_ssh_credentials()

    def _send_ssh_credentials(self) -> None:
        """Send SSH private key to the pod."""
        remote_path = "/root/.ssh/id_ed25519"
        local_path = self.ssh_key_path

        logger.info(f"Uploading SSH key from {local_path} to {remote_path}...")

        ssh = self._connect_ssh()
        sftp = ssh.open_sftp()

        try:
            sftp.mkdir("/root/.ssh", mode=0o700)
        except IOError:
            # Directory should exist anyway
            pass

        sftp.put(local_path, remote_path)
        sftp.chmod(remote_path, 0o600)
        sftp.close()
        ssh.close()

        logger.success("SSH key uploaded successfully to pod.")
