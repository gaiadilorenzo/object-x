# 🏃‍♀️ Training

Ensure the following environment variables are set:
   - `DATA_ROOT_DIR` = Path to the downloaded dataset (i.e., `root_dir`) as described in the [README](README.md)
   - `SCRATCH` = Path to the directory where you want to save the model

## Structured Latent Representations
Download the pretrained weights from Trellis from [here](https://huggingface.co/JeffreyXiang/TRELLIS-image-large) and save them to `SCRATCH`.
In order to train the structured latent representation, make sure the env `SCRATCH` is set to the path where you want to save the model.

Then run the following command:
```bash
bash scripts/trainval/train_latent_autoencoder.sh
```

> **Important:** The next step assumes the final snapshot is saved to `pretrained/u3dgs_pretrained.pth.tar` otherwise set `inference.slat_model_path` to the path of the snapshot for the next step.

Once converged, you can create intermediate results by running:

```bash
bash scripts/inference/run_pipeline_slat.sh --split train [--visualize]
bash scripts/inference/run_pipeline_slat.sh --split val [--visualize]
bash scripts/inference/run_pipeline_slat.sh --split test [--visualize]
```

> **Important:** The next step assumes the final snapshot is saved to `pretrained/slat_pretrained.pth.tar` otherwise set `autoencoder.encoder.voxel.pretrained` to the path of the snapshot for the next step.

## U-3DGS Representation
In order to train the U-3DGS representation, make sure the env `SCRATCH` is set to the path where you want to save the model. It is expected the previous step has been completed, the intermediate results are saved and the pretrained weights are also saved.

Then run the following command:
```bash
bash scripts/trainval/train.sh \
    autoencoder.encoder.voxel.channels=[16,32,64] # Res 16
```


> **Important:** The next step assumes the final snapshot is saved to `pretrained/u3dgs_pretrained_16.pth.tar` otherwise set `inference.u3dgs_model_path` to the path of the snapshot for the next step.

Once converged, you can create intermediate results by running:

```bash
bash scripts/inference/run_pipeline_u3dgs.sh --split train [--visualize]
bash scripts/inference/run_pipeline_u3dgs.sh --split val [--visualize]
bash scripts/inference/run_pipeline_u3dgs.sh --split test [--visualize]
```

> **Important:** The next step assumes the final snapshot is saved to `pretrained/u3dgs_pretrained_16.pth.tar` otherwise set `autoencoder.encoder.pretrained` to the path of the snapshot for the next step.

## Learning Other Tasks
In order to train the other tasks, make sure the env `SCRATCH` is set to the path where you want to save the model. It is expected the previous steps have been completed, the intermediate results are saved and the pretrained weights are also saved.

Then run the following command:
```bash
bash scripts/train_val/train_scene_graph_loc.sh \
    train.optim.free_sgaligner_epoch=-1 train.optim.free_voxel_epoch=10000 # Till epoch 3

# Unfreeze the voxel encoder/decoder

bash scripts/train_val/train_scene_graph_loc.sh \
    train.optim.free_sgaligner_epoch=-1 train.optim.free_voxel_epoch=-1 \
    --snapshot <path_to_frozen_snapshot>/snapshots/best_snapshot.pth.tar \
    train.optim.lr=1e-4
```
