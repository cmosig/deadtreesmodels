import torch
import os
from doitall import save_poly

# Add these environment variables to help debug NCCL issues
os.environ["NCCL_DEBUG"] = "INFO"

# Verify CUDA is available
# if not torch.cuda.is_available():
#     raise RuntimeError("CUDA is not available")

from doitall import inference_deadwood

# Use device directly instead of set_device
polygons = inference_deadwood("./data/ortho.tif")

save_poly("./data/ortho.geojson", polygons, "epsg:4326")
