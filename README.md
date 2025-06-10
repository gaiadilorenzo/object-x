<p align="center">
  <h2 align="center"> Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations </h2>
    <p align="center">
    <a>Gaia Di Lorenzo</a><sup>1</sup>
    .
    <a>Federico Tombari</a><sup>3</sup>
    .
    <a>Marc Pollefeys</a><sup>1, 2</sup>
    .
    <a>Dániel Béla Baráth</a><sup>1, 3</sup>
    .
  </p>
  <p align="center">
    <sup>1</sup>ETH Zürich · <sup>2</sup>Microsoft · <sup>3</sup>Google
  </p>
</p>
<p align="center">
<a href="https://arxiv.org/abs/2506.04789"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://gaiadilorenzo.github.io'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</p>
<p align="center">
  <a href="">
    <img src="assets/teaser.png" width="70%">
  </a>
</p>

## 📃 Abstract

Learning effective multi-modal 3D representations of objects is essential for numerous applications, such as augmented reality and robotics. Existing methods often rely on task-specific embeddings that are tailored either for semantic understanding or geometric reconstruction. As a result, these embeddings typically cannot be decoded into explicit geometry and simultaneously reused across tasks. In this paper, we propose Object-X, a versatile multi-modal object representation framework capable of encoding rich object embeddings (e.g., images, point cloud, text) and decoding them back into detailed geometric and visual reconstructions. Object-X operates by geometrically grounding the captured modalities in a 3D voxel grid and learning an unstructured embedding fusing the information from the voxels with the object attributes. The learned embedding enables 3D Gaussian Splatting-based object reconstruction, while also supporting a range of downstream tasks, including scene alignment, single-image 3D object reconstruction, and localization. Evaluations on two challenging real-world datasets demonstrate that Object-X produces high-fidelity novel-view synthesis comparable to standard 3D Gaussian Splatting, while significantly improving geometric accuracy.Moreover, Object-X achieves competitive performance with specialized methods in scene alignment and localization Critically, our object-centric descriptors require 3-4 orders of magnitude less storage compared to traditional image- or point cloud-based approaches, establishing Object-X as a scalable and highly practical solution for multi-modal 3D scene representation.

## ⏩ Code Release
- [ ] Add code and instructions for evaluation
- [ ] Add code and instructions for baseline evaluation
- [ ] Release checkpoints and metadata for 3RScan and ScanNet

## 🔨 Installation Guide

The code is tested with the following dependencies

- **Operating System**: Ubuntu
- **Architecture**: x86_64 GNU/Linux
- **Python Version**: 3.9.18
- **CUDA Version**: 12.4
- **NVIDIA Driver Version**: 550.144.03
- **GPU**: NVIDIA A100 PCIe 40GB
- **Total GPU Memory**: Above 40GB

### Setting Up the Virtual Environment

1. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   *(Ensure that your virtual environment is activated before proceeding with the installation.)*

2. **Install dependencies from `requirements.txt`**:
   ```bash
   pip install -r requirements.txt # add [--no-deps] if installation causes dependency issues
   pip install -r other_deps.txt
   ```

3. **Install dependencies separately**
    ```bash
   pip install dependencies/gaussian-splatting
   pip install git+https://github.com/nerfstudio-project/gsplat.git # Needed for evaluation
   pip install dependencies/2d-gaussian-splatting # Needed for baselines evaluation
   ```

### Checking Installation

After installing the dependencies, verify your environment:

- **Check Python Version**:
  ```bash
  python --version
  ```
  *(Should output `Python 3.9.18`)*

- **Check CUDA & GPU Availability**:
  ```bash
  nvidia-smi
  ```
  *(Ensure the GPU is detected and available for computation.)*


## 🪑 Dataset Generation

### 1. Downloading the Datasets
This section outlines the required datasets and their organization within the root directory.

#### **1.1 3RScan, 3DSSG, and ScanNet**
1. **3RScan**: Download from [here](https://github.com/WaldJohannaU/3RScan) and move all files to `\<root_dir>/scenes/`.
2. **3DSSG**: Download from [here](https://3dssg.github.io/) and place all files in `\<root_dir>/files/`.
3. **ScanNet**: Download from [here](http://www.scan-net.org) and move the scenes to `\<root_dir>/scenes/`.
4. **Additional Meta Files**: Download from [this link](https://drive.google.com/drive/folders/1pdZsvAqsVjTkRbNuR3xMDMlkDnf-yyS6?usp=share_link) and move them to `\<root_dir>/files/`.

After this step, the directory structure should look as follows:

```
├── <root_dir>
│   ├── files                 <- Meta files and annotations
│   │   ├── <meta_files_0>
│   │   ├── <meta_files_1>
│   │   ├── ...
│   ├── scenes                <- Scans (3RScan/ScanNet)
│   │   ├── <id_scan_0>
│   │   ├── <id_scan_1>
│   │   ├── ...
```

---

### 2. Preprocessing 3RScan Dataset

#### **2.1 Generating `labels.instances.align.annotated.v2.ply`**
To generate `labels.instances.align.annotated.v2.ply` for each 3RScan scan, refer to the repository:
[3DSSG Data Processing](https://github.com/ShunChengWu/3DSSG/blob/master/data_processing/transform_ply.py).

#### **2.2 Preprocessing Scene Graph Information**
1. The preprocessing code for 3RScan is located in the [dependencies/VLSG](dependencies/VLSG) directory.
2. Ensure the following environment variables are set:
   - `VLSG_SPACE` = Repository path
   - `DATA_ROOT_DIR` = Path to the downloaded dataset (i.e., `root_dir`)
   - `CONDA_BIN` = `.venv/bin` (as linked in installation)
3. Execute the preprocessing script:
   ```bash
   cd dependencies/VLSG && bash scripts/preprocess/scan3r_data_preprocess.sh
   ```

#### **2.3 Generating Ground Truth Patch-Object Annotation**
Run the following command to generate pixel-wise and patch-level ground truth annotations:
```bash
cd dependencies/VLSG && bash scripts/gt_annotations/scan3r_gt_annotations.sh
```

#### **2.4 Generating Patch-Level Features (Optional: Image Localization Training)**
Precompute patch-level features using [Dino v2](https://dinov2.metademolab.com/):
```bash
cd dependencies/VLSG && bash scripts/features2D/scan3r_dinov2.sh
```

#### **2.5 Generating Featured Voxel Annotations**
Run the following command to generate featured voxel annotations:
```bash
bash scripts/voxel_annotations/voxelise_features.sh --split {split}
# 2.5.1 Generating Subscenes Annotations (Optional)
# Follow the instructions from SGAligner at dependencies/sgaligner to create subscenes annotations
# Then run the following command:
# bash scripts/voxel_annotations/voxelise_features_scene_alignment.sh --split {split}
```

#### **2.6 Generating Gaussian Splat Annotations (Optional: Baseline Computation)**
Generate Gaussian splat annotations using the following commands:
```bash
bash scripts/gs_annotations/map_to_colmap.sh --split {split}
bash scripts/gs_annotations/annotate_gaussian.sh --split {split}
```

---

### 3. Preprocessing ScanNet Dataset

ScanNet requires scene graph annotations generated using [SceneGraphFusion](https://github.com/ShunChengWu/SceneGraphFusion).

#### **3.1 Download and Set Up SceneGraphFusion**
1. Download the pretrained model from [here](https://drive.google.com/file/d/1_745ofaOUyP_iFK8A3cSW60L4V7TlWa7/view) and move it to `dependencies/SCENE-GRAPH-FUSION/`.
2. Build SceneGraphFusion by following the instructions in its [repository](https://github.com/ShunChengWu/SceneGraphFusion?tab=readme-ov-file#prerequisites).

#### **3.2 Generating Scene Graph Annotations for ScanNet**
Run the following commands:
```bash
python preprocessing/scene_graph_anno/scenegraphfusion_prediction.py
python preprocessing/scene_graph_anno/scenegraphfusion2scan3r.py
```

#### **3.3 Generating Ground Truth Patch-Object Annotation**
Generate ground truth annotations with:
```bash
python preprocessing/gt_anno/scannet_obj_projector.py
```

#### **3.4 Generating Featured Voxel Annotations**
Run the following script to generate featured voxel annotations:
```bash
bash scripts/voxel_annotations/voxelise_features_scannet.sh
```

#### **3.5 Generating Gaussian Splat Annotations (Optional: Baseline Computation)**
To generate Gaussian splat annotations, execute:
```bash
bash scripts/gs_annotations/map_to_colmap_scannet.sh
bash scripts/gs_annotations/annotate_gaussian_scannet.sh
```

---
### Outline
After the above preprocessing, the directory structure should look as follows:

```
├── <root_dir>
│   ├── files                      <- Meta files and annotations
│   │   ├── Features2D             <- (Step 2.4)
│   │   ├── gt_projection          <- (Step 2.3)
│   │   ├── orig                   <- (Step 2.1)
│   │   ├── patch_anno             <- (Step 2.3)
│   │   ├── gs_annotations         <- (Step 2.5/2.6)
│   │   ├── gs_annotations_scannet <- (Step 3.4/3.5)
│   │   ├── <meta_files_0>
│   │   ├── <meta_files_1>
│   │   ├── ...
│   ├── scenes                     <- Scans (3RScan/ScanNet)
│   │   ├── <id_scan_0>
│   │   ├── <id_scan_1>
│   │   ├── ...
│   ├── scene_graph_fusion         <- (Step 3.1)
│   │   ├── <id_scan_0>
│   │   ├── <id_scan_1>
│   │   ├── ...
│   ├── out                        <- (Step 2.5.1)
│   │   ├── files ...
│   │   ├── scenes ...
```

## 🏃‍♀️ Training

Refer to the [TRAIN.md](TRAIN.md) for training instructions.

## 📕 BibTeX 
```
@misc{dilorenzo2025objectxlearningreconstructmultimodal,
      title={Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations}, 
      author={Gaia Di Lorenzo and Federico Tombari and Marc Pollefeys and Daniel Barath},
      year={2025},
      eprint={2506.04789},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.04789}, 
}
 ```

## ♻️ Acknowledgments

In this project we use (parts of) the official implementations of the following works:

- SceneGraphLoc: [SceneGraphLoc](https://github.com/y9miao/VLSG)
- Trellis: [Trellis](https://github.com/Microsoft/TRELLIS)
- SGAligner: [SGAligner](https://github.com/sayands/sgaligner)
- GSplat: [GSplat](https://github.com/nerfstudio-project/gsplat)
- 2D Gaussian Splatting: [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)
- 3DSSG: [3DSSG](https://3dssg.github.io/)
- SceneGraphFusion: [SceneGraphFusion](https://github.com/ShunChengWu/SceneGraphFusion)
