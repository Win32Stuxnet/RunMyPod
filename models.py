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
        
        #Map the download and pray that it works based on type
        mapping = {
            ModelType.CHECKPOINT: "checkpoints",
            ModelType.LORA: "loras",
            ModelType.VAE: "vae",
            ModelType.EMBEDDING: "embeddings",
            ModelType.CONTROLNET: "controlnet",
        }
        return mapping.get(self.type, "checkpoints")