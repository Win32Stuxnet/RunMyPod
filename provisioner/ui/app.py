from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Button, Input, Label, Select, Log, Static
from textual.screen import Screen
from textual.message import Message
from textual.reactive import reactive
from textual.worker import Worker

from ..models import ProvisioningConfig, ModelSpec, ModelType
from ..services.runpod_client import RunPodService

import os
import time

class ConfigScreen(Screen):
    BINDINGS = [("n", "next", "Next")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("RunPod API Key:"),
            Input(placeholder="rpa_...", password=True, id="api_key"),
            Label("Hugging Face Token (Optional, for gated models):"),
            Input(placeholder="hf_...", password=True, id="hf_token"),
            Label("SSH Private Key Path (Optional, defaults to system agent/keys):"),
            Input(placeholder="~/.ssh/id_ed25519", id="ssh_key"),
            Button("Next", variant="primary", id="next_btn")
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next_btn":
            self.submit()

    def submit(self):
        api_key = self.query_one("#api_key", Input).value
        hf_token = self.query_one("#hf_token", Input).value
        ssh_key = self.query_one("#ssh_key", Input).value
        
        if not api_key:
            self.notify("API Key is required", severity="error")
            return
            
        self.app.config_data.api_key = api_key
        if hf_token:
            self.app.config_data.hf_token = hf_token
        if ssh_key:
            self.app.ssh_key_path = ssh_key
            
        self.app.push_screen("provision_screen")

class ProvisionScreen(Screen):
    gpu_options = reactive([])

    def on_mount(self):
        self.run_worker(self.fetch_gpus, thread=True)

    def fetch_gpus(self):
        try:
            service = RunPodService(self.app.config_data.api_key)
            gpus = service.list_gpus()
            options = []
            # Sort by price
            gpus.sort(key=lambda x: x.get('communityPrice', 999))
            
            for g in gpus:
                display = f"{g.get('displayName', g.get('id'))} - ${g.get('communityPrice', 0):.2f}/hr ({g.get('memoryInGb', '?')}GB)"
                options.append((display, g['id']))
            
            if not options:
                self.app.call_from_thread(self.notify, "No GPUs found or API error.", severity="error")
                return

            self.app.call_from_thread(self.update_gpu_select, options)
        except Exception as e:
            self.app.call_from_thread(self.notify, f"Error: {str(e)}", severity="error")

    def update_gpu_select(self, options):
        select = self.query_one("#gpu_select", Select)
        select.set_options(options)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Select GPU:"),
            Select([], id="gpu_select", prompt="Loading GPUs..."),
            
            Label("Volume Size (GB):"),
            Input(value="40", id="volume_size"),
            
            Label("Container Disk Size (GB):"),
            Input(value="40", id="container_size"),

            Button("Next", variant="primary", id="next_btn")
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next_btn":
            gpu_id = self.query_one("#gpu_select", Select).value
            if not gpu_id:
                self.notify("Please select a GPU", severity="error")
                return

            try:
                self.app.config_data.gpu_type_id = gpu_id
                self.app.config_data.volume_size_gb = int(self.query_one("#volume_size", Input).value)
                self.app.config_data.container_disk_size_gb = int(self.query_one("#container_size", Input).value)
            except ValueError:
                self.notify("Sizes must be integers", severity="error")
                return
            
            self.app.push_screen("model_screen")

class ModelScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Add Models / Loras"),
            Horizontal(
                Input(placeholder="URL", id="model_url", classes="box url_input"),
                Input(placeholder="Filename", id="model_name", classes="box name_input"),
                Select([(t.value, t.name) for t in ModelType], id="model_type", value=ModelType.CHECKPOINT.value),
                Button("Add", id="add_btn"),
                classes="input_row"
            ),
            Label("Added Models:", classes="mt-1"),
            Vertical(id="model_list", classes="box model_list"),
            Button("Deploy", variant="success", id="deploy_btn", classes="mt-1")
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add_btn":
            url = self.query_one("#model_url", Input).value
            name = self.query_one("#model_name", Input).value
            m_type_val = self.query_one("#model_type", Select).value
            
            # Map value back to Enum
            m_type = next((t for t in ModelType if t.value == m_type_val), ModelType.CHECKPOINT)

            if not url or not name:
                self.notify("URL and Name are required", severity="error")
                return

            spec = ModelSpec(name=name, url=url, type=m_type)
            self.app.config_data.models.append(spec)
            
            # Add to display
            list_container = self.query_one("#model_list", Vertical)
            list_container.mount(Label(f"[{m_type.name}] {name}"))
            
            # Clear inputs
            self.query_one("#model_url", Input).value = ""
            self.query_one("#model_name", Input).value = ""

        elif event.button.id == "deploy_btn":
            self.app.push_screen("deploy_screen")

class DeployScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Provisioning & Setup..."),
            Log(id="log_view")
        )
        yield Footer()

    def on_mount(self):
        self.run_worker(self.perform_deployment, thread=True)

    def perform_deployment(self):
        log_view = self.query_one("#log_view", Log)
        
        def log(msg):
            self.app.call_from_thread(log_view.write_line, msg)

        log("Initializing RunPod Service...")
        service = RunPodService(self.app.config_data.api_key)
        
        try:
            log(f"Creating Pod (GPU: {self.app.config_data.gpu_type_id})...")
            pod = service.create_pod(self.app.config_data)
            pod_id = pod['id']
            log(f"Pod created: {pod_id}")
            log("Waiting for pod to start (this may take a few minutes)...")
            
            ready_pod = service.wait_for_pod(pod_id)
            log("Pod is RUNNING!")
            
            log("Connecting via SSH to install ComfyUI and Models...")
            for line in service.execute_setup(ready_pod, self.app.config_data):
                log(f"[REMOTE] {line}")
                
            log("------------------------------------------------")
            log("Provisioning Complete!")
            
            # Find public IP and port
            ports = ready_pod.get('runtime', {}).get('ports', [])
            http_port = next((p for p in ports if p['privatePort'] == 8188), None)
            if http_port:
                url = f"http://{http_port['ip']}:{http_port['publicPort']}"
                log(f"Access ComfyUI at: {url}")
            else:
                log("Could not determine public URL. Check RunPod dashboard.")
                
        except Exception as e:
            log(f"ERROR: {e}")

class ProvisionerApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    Container {
        width: 90%;
        height: 90%;
        border: solid green;
        padding: 1;
    }
    .box {
        border: solid white;
        padding: 1;
    }
    .input_row {
        height: auto;
        align: left middle;
        margin-bottom: 1;
    }
    .url_input {
        width: 40%;
    }
    .name_input {
        width: 30%;
    }
    .model_list {
        height: 1fr;
        overflow-y: auto;
    }
    .mt-1 {
        margin-top: 1;
    }
    Input {
        margin: 0 1;
    }
    Button {
        margin: 0 1;
    }
    Select {
        width: 20%;
    }
    """
    
    config_data = ProvisioningConfig(api_key="")
    ssh_key_path = None

    def on_mount(self):
        self.push_screen(ConfigScreen())

