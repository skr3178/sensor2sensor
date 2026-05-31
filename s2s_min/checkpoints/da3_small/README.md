---
license: apache-2.0
tags:
- depth-estimation
- computer-vision
- monocular-depth
- multi-view-geometry
- pose-estimation
library_name: depth-anything-3
pipeline_tag: depth-estimation
---

# Depth Anything 3: DA3-SMALL

<div align="center">

[![Project Page](https://img.shields.io/badge/Project_Page-Depth_Anything_3-green)](https://depth-anything-3.github.io)
[![Paper](https://img.shields.io/badge/arXiv-Depth_Anything_3-red)](https://arxiv.org/abs/)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/depth-anything/Depth-Anything-3)  # noqa: E501
<!-- Benchmark badge removed as per request -->

</div>

## Model Description

DA3 Small model for multi-view depth estimation and camera pose estimation. Efficient foundation model with unified depth-ray representation.

| Property | Value |
|----------|-------|
| **Model Series** | Any-view Model |
| **Parameters** | 0.08B |
| **License** | Apache 2.0 |



## Capabilities

- ✅ Relative Depth
- ✅ Pose Estimation
- ✅ Pose Conditioning

## Quick Start

### Installation

```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3
cd depth-anything-3
pip install -e .
```

### Basic Example

```python
import torch
from depth_anything_3.api import DepthAnything3

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3-small")
model = model.to(device=device)

# Run inference on images
images = ["image1.jpg", "image2.jpg"]  # List of image paths, PIL Images, or numpy arrays
prediction = model.inference(
    images,
    export_dir="output",
    export_format="glb"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
)

# Access results
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
```

### Command Line Interface

```bash
# Process images with auto mode
da3 auto path/to/images \
    --export-format glb \
    --export-dir output \
    --model-dir depth-anything/da3-small

# Use backend for faster repeated inference
da3 backend --model-dir depth-anything/da3-small
da3 auto path/to/images --export-format glb --use-backend
```

## Model Details

- **Developed by:** ByteDance Seed Team
- **Model Type:** Vision Transformer for Visual Geometry
- **Architecture:** Plain transformer with unified depth-ray representation
- **Training Data:** Public academic datasets only

### Key Insights

💎 A **single plain transformer** (e.g., vanilla DINO encoder) is sufficient as a backbone without architectural specialization.  # noqa: E501

✨ A singular **depth-ray representation** obviates the need for complex multi-task learning.

## Performance

🏆 Depth Anything 3 significantly outperforms:
- **Depth Anything 2** for monocular depth estimation
- **VGGT** for multi-view depth estimation and pose estimation

For detailed benchmarks, please refer to our [paper](https://depth-anything-3.github.io).  # noqa: E501

## Limitations

- The model is trained on academic datasets and may have limitations on certain domain-specific images  # noqa: E501
- Performance may vary depending on image quality, lighting conditions, and scene complexity


## Citation

If you find Depth Anything 3 useful in your research or projects, please cite:

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},  # noqa: E501
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Links

- 🏠 [Project Page](https://depth-anything-3.github.io)
- 📄 [Paper](https://arxiv.org/abs/)
- 💻 [GitHub Repository](https://github.com/ByteDance-Seed/depth-anything-3)
- 🤗 [Hugging Face Demo](https://huggingface.co/spaces/depth-anything/Depth-Anything-3)
- 📚 [Documentation](https://github.com/ByteDance-Seed/depth-anything-3#-useful-documentation)

## Authors

[Haotong Lin](https://haotongl.github.io/) · [Sili Chen](https://github.com/SiliChen321) · [Junhao Liew](https://liewjunhao.github.io/) · [Donny Y. Chen](https://donydchen.github.io) · [Zhenyu Li](https://zhyever.github.io/) · [Guang Shi](https://scholar.google.com/citations?user=MjXxWbUAAAAJ&hl=en) · [Jiashi Feng](https://scholar.google.com.sg/citations?user=Q8iay0gAAAAJ&hl=en) · [Bingyi Kang](https://bingykang.github.io/)  # noqa: E501
