from unidepth.models import UniDepthV1

model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14") # or "lpiccinelli/unidepth-v1-cnvnxtl" for the ConvNext backbone

import torch
import numpy as np
from PIL import Image

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
image_path = "MealsRetrieval-ROOMELSA/Ouput_perspective/output_0_colors (1).png/perspective_00_2.png"
# Load the RGB image and the normalization will be taken care of by the model
rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W

predictions = model.infer(rgb)

# Metric Depth Estimation
depth = predictions["depth"]

# Point Cloud in Camera Coordinate
xyz = predictions["points"]

# Intrinsics Prediction
intrinsics = predictions["intrinsics"]

