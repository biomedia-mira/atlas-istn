# Atlas-ISTN: Joint segmentation, registration and atlas construction with image-and-spatial transformer networks

This repository contains the code for the paper
> M. Sinclair, A. Schuh, K. Hahn, K. Petersen, Y. Bai, J. Batten, M. Schaap, B. Glocker. [_Atlas-ISTN: Joint segmentation, registration and atlas construction with image-and-spatial transformer networks_](https://doi.org/10.1016/j.media.2022.102383). 2022. Medical Image Analysis, Vol. 78

## Code

For running the code, we recommend setting up a dedicated Python environment.

### Setup Python environment using conda

Create and activate a Python 3.8 conda environment:

   ```shell
   conda create -n pyatlas python=3.8
   conda activate pyatlas
   ```
   
Install PyTorch using conda (for CUDA Toolkit 11.3):
   
   ```shell
   conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
   ```
   
### Setup Python environment using virtualenv

Create and activate a Python 3.8 virtual environment:

   ```shell
   virtualenv -p python3 <path_to_envs>/pyatlas
   source <path_to_envs>/pyatlas/bin/activate
   ```
   
Install PyTorch using pip:
   
   ```shell
   pip install torch torchvision
   ```
   
### Install additional Python packages:
   
   ```shell
   pip install matplotlib jupyter pandas seaborn scikit-learn SimpleITK==1.2.4 tensorboard tensorboardX attrdict tqdm pyyaml pytorch-lightning torchio
   ```
