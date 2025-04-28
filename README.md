<p align="center">
  <h2 align="center"> Graph2Splat: Learning 3D Gaussian Splats from Scene Graph Embeddings </h2>
    <p align="center">
    <a>Gaia Di Lorenzo</a><sup>1</sup>
    .
    <a>Federico Tamberi</a><sup>3</sup>
    .
    <a>Marc Pollefeys</a><sup>1, 2</sup>
    .
    <a>Dániel Béla Baráth</a><sup>1, 3</sup>
    .
  </p>
  <p align="center">
    <sup>1</sup>ETH Zürich · <sup>2</sup>Microsoft Spatial AI Lab · <sup>3</sup>Google
  </p>
</p>

<p align="center">
  <a href="">
    <img src="assets/teaser.png" width="70%">
  </a>
</p>

## 📃 Abstract

3D scene representation is crucial for many computer vision applications such as robotics and augmented reality. Traditional methods like point clouds require significant storage, hindering their practicality. While 3D scene graphs offer a lightweight alternative by representing scenes as interconnected objects, they lack the geometric and visual detail needed for many tasks.
In this work, we introduce a novel method **Graph2Splat** to learn rich scene graph node embeddings, enabling both efficient 3D scene reconstruction and direct application in other downstream tasks. Graph2Splat predicts a 3D Gaussian Splat (3DGS) representation for each object from its embedding, capturing both appearance and geometry.
Evaluated on two real-world datasets, Graph2Splat achieves high-fidelity novel view synthesis comparable to direct 3DGS reconstruction, while significantly improving geometric accuracy.
Importantly, it needs 3-4 orders of magnitude less storage than other approaches, making it a practical and versatile solution for scene representation.


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
   pip install -r requirements.txt
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
