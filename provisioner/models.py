from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ModelType(str, Enum):
    CHECKPOINT = "checkpoint"
    LORA = "lora"
    VAE = "vae"
    EMBEDDING = "embedding"
    CONTROLNET = "controlnet"

class ModelSpec(BaseModel):
    name: str = Field(..., description="Name of the model file (e.g. 'sd_xl_base_1.0.safetensors')")
    url: str = Field(..., description="Download URL for the model")
    type: ModelType = Field(default=ModelType.CHECKPOINT, description="Type of the model")
    subfolder: Optional[str] = Field(None, description="Specific subfolder in ComfyUI/models/ to place it")

    @property
    def install_path(self) -> str:
        """Returns the relative path inside ComfyUI/models/ where this should go."""
        if self.subfolder:
            return self.subfolder
        
        # Defaults based on type
        mapping = {
            ModelType.CHECKPOINT: "checkpoints",
            ModelType.LORA: "loras",
            ModelType.VAE: "vae",
            ModelType.EMBEDDING: "embeddings",
            ModelType.CONTROLNET: "controlnet",
        }
        return mapping.get(self.type, "checkpoints")

class ProvisioningConfig(BaseModel):
    api_key: str = Field(..., description="RunPod API Key")
    hf_token: Optional[str] = Field(default=None, description="Hugging Face API Token")
    gpu_type_id: str = Field(default="NVIDIA GeForce RTX 3090", description="GPU Type ID")
    cloud_type: str = Field(default="COMMUNITY", description="Cloud Type (COMMUNITY, SECURE)")
    volume_size_gb: int = Field(default=40, description="Volume Size in GB")
    container_disk_size_gb: int = Field(default=40, description="Container Disk Size in GB")
    template_id: str = Field(default="runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04", description="Docker Image/Template ID")
    
    models: List[ModelSpec] = Field(default_factory=list, description="List of models to install")

    comfyui_repo: str = "https://github.com/comfyanonymous/ComfyUI"
    comfyui_manager_repo: str = "https://github.com/ltdrdata/ComfyUI-Manager.git"

