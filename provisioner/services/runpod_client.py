import runpod
import time
import paramiko
from typing import List, Dict, Any, Generator
import logging
from ..models import ProvisioningConfig, ModelSpec

logger = logging.getLogger(__name__)

class RunPodService:
    def __init__(self, api_key: str):
        runpod.api_key = api_key
        self.api_key = api_key

    def list_gpus(self) -> List[Dict[str, Any]]:
        """
        Fetches available GPU types.
        Note: runpod.get_gpus() is the standard way, but returns a list of dicts.
        """
        try:
            return runpod.get_gpus()
        except Exception as e:
            logger.error(f"Failed to list GPUs: {e}")
            return []

    def create_pod(self, config: ProvisioningConfig) -> Dict[str, Any]:
        """
        Creates a pod based on configuration.
        Returns the pod info.
        """
        try:
            # Create the pod
            pod = runpod.create_pod(
                name=f"ComfyUI-{int(time.time())}",
                image_name=config.template_id,
                gpu_type_id=config.gpu_type_id,
                cloud_type=config.cloud_type,
                container_disk_in_gb=config.container_disk_size_gb,
                volume_in_gb=config.volume_size_gb,
                ports="8188/http,22/tcp",  # ComfyUI default port + SSH
                env={
                    "JUPYTER_PASSWORD": "runpod", # Default for many templates
                    "HF_TOKEN": config.hf_token if config.hf_token else "",
                }
            )
            return pod
        except Exception as e:
            raise RuntimeError(f"Failed to create pod: {e}")

    def wait_for_pod(self, pod_id: str) -> Dict[str, Any]:
        """
        Waits for the pod to become running.
        """
        while True:
            pod = runpod.get_pod(pod_id)
            if pod['desiredStatus'] == 'RUNNING' and pod.get('runtime', {}).get('ports'):
                return pod
            if pod['desiredStatus'] == 'EXITED': # Should not happen ideally
                 raise RuntimeError("Pod exited unexpectedly")
            
            time.sleep(2)

    def generate_setup_script(self, config: ProvisioningConfig) -> str:
        """
        Generates a bash script to install ComfyUI and models.
        """
        script = [
            "#!/bin/bash",
            "set -e", # Exit on error
            "echo 'Starting ComfyUI Provisioning...'",
            "cd /workspace",
            "",
            "# Install ComfyUI",
            f"if [ ! -d 'ComfyUI' ]; then",
            f"  git clone {config.comfyui_repo}",
            "fi",
            "cd ComfyUI",
            "pip install -r requirements.txt",
            "",
            "# Install Manager",
            "cd custom_nodes",
            f"if [ ! -d 'ComfyUI-Manager' ]; then",
            f"  git clone {config.comfyui_manager_repo}",
            "fi",
            "cd ..", # Back to ComfyUI root
            "",
            "# Download Models",
        ]

        for model in config.models:
            path = f"models/{model.install_path}"
            # Check if URL is valid (basic check)
            if model.url.startswith("http"):
                script.append(f"echo 'Downloading {model.name}...'")
                
                wget_args = ""
                if "huggingface.co" in model.url and config.hf_token:
                     wget_args = f"--header='Authorization: Bearer {config.hf_token}'"
                
                # Use wget with content-disposition if name not explicit, but we have name
                script.append(f"wget {wget_args} -O '{path}/{model.name}' '{model.url}'")
            
        script.append("echo 'Provisioning Complete!'")
        return "\n".join(script)

    def execute_setup(self, pod: Dict[str, Any], config: ProvisioningConfig) -> Generator[str, None, None]:
        """
        Connects via SSH and executes the setup script.
        Yields logs.
        """
        # Extract SSH info
        ssh_port = None
        public_ip = None
        
        if pod.get('runtime') and pod['runtime'].get('ports'):
            for port in pod['runtime']['ports']:
                if port['privatePort'] == 22:
                    ssh_port = port['publicPort']
                    public_ip = port['ip']
                    break
        
        if not ssh_port or not public_ip:
             yield "Error: Could not find SSH port."
             return

        # Wait a bit for SSH to be up
        time.sleep(5)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Try connecting multiple times
        retries = 10
        for i in range(retries):
            try:
                # Standard RunPod templates typically use root/password or root key
                # We might need to handle keys. Ideally user provides one or we assume the 
                # template allows passwordless or we generate one. 
                # Many RunPod templates use keys. 
                # Wait, runpod.create_pod doesn't upload keys by default unless configured in user profile.
                # But we assume the user has configured their SSH keys in RunPod account? 
                # Or we can try to assume a default password if the template has one?
                # Most runpod/pytorch images don't have a default password set for root unless passed in env.
                # BUT, `runpod` library allows executing commands? No, looking at docs, `exec_command` is not in the library directly for pods?
                # Actually, `runpodctl` does it. 
                # Let's assume the user has their SSH key loaded in their local agent, or we can't easily SSH.
                
                # For this MVP, let's assume we rely on `paramiko` connecting. 
                # If this fails, we might need to ask user for SSH key path.
                
                client.connect(public_ip, port=ssh_port, username='root', timeout=10)
                break
            except Exception as e:
                if i == retries - 1:
                    yield f"Failed to connect via SSH: {e}"
                    return
                time.sleep(2)
                yield "Waiting for SSH..."

        # Execute
        script_content = self.generate_setup_script(config)
        
        # Write script to file
        cmd = f"cat << 'EOF' > /workspace/setup_comfy.sh\n{script_content}\nEOF\n"
        stdin, stdout, stderr = client.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status()
        
        if exit_status != 0:
            yield "Failed to write setup script."
            return

        # Run script
        stdin, stdout, stderr = client.exec_command("chmod +x /workspace/setup_comfy.sh && /workspace/setup_comfy.sh")
        
        # Stream output
        for line in iter(stdout.readline, ""):
            yield line.strip()
        
        client.close()

