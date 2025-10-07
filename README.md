# Multi-Scale VQVAE for Point Cloud Processing

This project implements a Vector Quantized Variational Autoencoder (VQVAE) for processing point clouds at multiple scales. The multi-scale approach allows the model to capture both fine-grained details and broader structural information from point cloud data.

## Key Features

- Multi-scale point cloud representation
- VQVAE architecture for efficient encoding and decoding


## Dataset
The point clouds are uniformly sampled from meshes from ShapeNetCore dataset (version 2) and use the official split. Please use this [link](https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j) to download the ShapeNet point cloud. The point cloud should be placed into data directory.

## File Structure

- `vqvae_model.py`: Contains the VQVAE model implementation, including Encoder, Decoder, and VectorQuantizer classes.
- `data_processing.py`: Handles data loading and preprocessing for the ShapeNetV2 dataset.
- `train.py`: Implements the training loop and hyperparameter settings.
- `README.md`: Project documentation (this file).

