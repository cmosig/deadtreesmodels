import torch
import os

# Add these environment variables to help debug NCCL issues
os.environ["NCCL_DEBUG"] = "INFO"

# Verify CUDA is available
if not torch.cuda.is_available(): 
    raise RuntimeError("CUDA is not available")

from doitall import inference_deadwood

# Use device directly instead of set_device
inference_deadwood("./data/ortho.tif")