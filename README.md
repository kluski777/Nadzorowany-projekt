# Latent Space Inpainter Project

This project implements a latent space inpainting system for reconstructing masked image representations. The workflow involves training autoencoders, generating latent spaces, clustering them, and training cluster-specific inpainter models.

## Overview

The pipeline consists of several stages:

1. **Data Preparation**: Generate train/val/test splits
2. **Autoencoder Training**: Train an autoencoder to encode images into latent spaces
3. **Latent Space Generation**: Extract latent representations (masked and target)
4. **Feature Extraction & Clustering**: Reduce dimensionality and cluster latent spaces
5. **Inpainter Training**: Train cluster-specific convolutional inpainter models

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Configuration

Edit `config.yaml` to configure:
- Data paths and splits
- Model architecture and hyperparameters
- Training parameters
- Inpainter-specific settings

## Workflow

### Step 1: Generate Dataset Splits

First, create train/val/test splits from the dataset:

```bash
uv run src/main.py generate_splits --config config.yaml
```

This creates CSV files in `data/splits/` with train/val/test indices.

### Step 2: Train Autoencoder

Train an autoencoder model to learn latent representations:

```bash
uv run src/main.py train_autoencoder --config config.yaml
```

Or resume from a checkpoint:

```bash
uv run src/main.py train_autoencoder \
    --config config.yaml \
    --checkpoint checkpoints/autoencoder-epoch-10.ckpt
```

The trained model will be saved in `checkpoints/`.

### Step 3: Generate Latent Spaces (Without Clusters)

Generate latent spaces for all images in the splits. This step does NOT require cluster labels:

```bash
uv run src/main.py generate_latent_spaces \
    --config config.yaml \
    --checkpoint checkpoints/AE-latent2k.ckpt \
    --output-dir data/latent_spaces
```

This creates `train.npz`, `val.npz`, and `test.npz` files containing:
- `indices`: Image indices
- `masked_latent`: Latent space of masked images (shape: `[n_samples, latent_channels, 8, 8]`)
- `target_latent`: Latent space of original images (shape: `[n_samples, latent_channels, 8, 8]`)
- `images`: masks coming out from the encoder.

### Step 4: Train Feature Extractor

Fit a PCA-based feature extractor on the latent spaces:

```bash
uv run src/main.py fit_feature_extractor \
    --input-dir data/latent_spaces \
    --output-dir data/models \
    --n-components 12
```

This saves `data/models/feature_extractor.pkl`.

### Step 5: Generate Latent Components

Reduce latent spaces to lower-dimensional components:

```bash
uv run src/main.py generate_latent_components \
    --input-dir data/latent_spaces \
    --checkpoint data/models/feature_extractor.pkl \
    --output-dir data/latent_components
```

### Step 6: Train Clusterizer

Fit a K-means clusterizer on the latent components:

```bash
uv run src/main.py fit_clusterizer \
    --input-dir data/latent_components \
    --output-dir data/models \
    --n-clusters 12
```

This saves `data/models/clusterizer.pkl` and `data/models/scaler.pkl`.

### Step 7: Generate Clusters

Assign cluster labels to latent components:

```bash
uv run src/main.py generate_clusters \
    --input-dir data/latent_components \
    --checkpoint data/models/clusterizer.pkl \
    --output-dir data/clusters
```

### Step 8: Regenerate Latent Spaces (With Clusters)

Regenerate latent spaces with cluster labels included:

```bash
uv run src/main.py generate_latent_spaces \
    --config config.yaml \
    --checkpoint checkpoints/AE-latent2k.ckpt \
    --output-dir data/latent_spaces \
    --feature-extractor-checkpoint data/models/feature_extractor.pkl \
    --clusterizer-checkpoint data/models/clusterizer.pkl
```

This creates npz files with an additional `cluster` field containing cluster IDs.

### Step 9: Train Inpainter Models

You can train either cluster-specific inpainters or a common inpainter that works for all clusters.

#### Option A: Train a Common Inpainter (Recommended for starting)

Train a single inpainter on all data (used as fallback when cluster-specific inpainter is missing):

```bash
uv run src/main.py train_inpainter \
    --config config.yaml \
    --latent-dir data/latent_spaces
```

This saves `checkpoints/inpainter-common-final.ckpt`.

#### Option B: Train Cluster-Specific Inpainters

Train a separate inpainter model for each cluster:

```bash
# Train inpainter for cluster 0
uv run src/main.py train_inpainter \
   --config config.yaml \
    --cluster-id 0 \
    --latent-dir data/latent_spaces 

# Train inpainter for cluster 1
uv run src/main.py train_inpainter \
    --config config.yaml \
    --cluster-id 1 \
    --latent-dir data/latent_spaces

# ... repeat for all clusters
```

Each model is saved as `checkpoints/inpainter-cluster{id}-*.ckpt`.

**Note:** At inference time, if a cluster-specific inpainter is not available, the pipeline automatically falls back to the common inpainter (`checkpoints/inpainter-common-final.ckpt`).

## Available Commands

### Data Preparation
- `generate_splits` - Generate train/val/test splits

### Model Training
- `train_autoencoder` - Train autoencoder model
- `train_bottleneck` - Train bottleneck layers for progressive reduction
- `train_inpainter` - Train inpainter model (common or cluster-specific)

### Feature Extraction & Clustering
- `fit_feature_extractor` - Fit PCA feature extractor
- `fit_clusterizer` - Fit K-means clusterizer
- `generate_latent_components` - Generate reduced-dimension components
- `generate_clusters` - Assign cluster labels

### Latent Space Generation
- `generate_latent_spaces` - Generate latent spaces (with optional clustering)

### Visualization
- `visualize_umap` - Visualize UMAP embeddings
- `visualize_elbow` - Generate elbow plot for optimal cluster count

## Data Structure

### Latent Spaces (without clusters)
```
data/latent_spaces/
├── train.npz
│   ├── indices: [n_samples]
│   ├── masked_latent: [n_samples, latent_channels, 8, 8]
│   └── target_latent: [n_samples, latent_channels, 8, 8]
├── val.npz
└── test.npz
```

## Model Architecture

### Autoencoder
- Encodes images (256x256) to latent space (8x8)
- Various architectures available: `res_convt`, `pixelshuffle_ae`, `resnet18_ae`, `vae`, etc.

### Inpainter
- Convolutional network with residual blocks
- Input: Masked latent space `[B, latent_channels, 8, 8]`
- Output: Reconstructed latent space `[B, latent_channels, 8, 8]`
- Uses residual learning: `output = input + correction`

## Configuration

Key settings in `config.yaml`:

```yaml
model:
  architecture: "final_2k"
  latent_channels: 128
  learning_rate: 0.001

inpainter:
  latent_channels: 128
  hidden_channels: 256
  num_blocks: 4
  learning_rate: 0.001
  loss_type: "mse"
  batch_size: 64
  max_epochs: 100
```

## Notes

- The workflow is designed in two phases:
  1. **Phase 1**: Generate latent spaces without clusters (Steps 1-3)
  2. **Phase 2**: Add clustering and train inpainters (Steps 4-9)

- Cluster assignment is optional when generating latent spaces. If `--feature-extractor-checkpoint` and `--clusterizer-checkpoint` are not provided, latent spaces are saved without cluster labels.

- Each cluster gets its own inpainter model, allowing specialized reconstruction for different image types/styles.
