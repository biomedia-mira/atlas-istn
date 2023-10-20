# Atlas-ISTN: Joint segmentation, registration and atlas construction with image-and-spatial transformer networks

When using this code, please cite the following paper:
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
   
### Usage

Two example use-cases are provided:

#### Example 1: 2D Synthetic Letter B
To run training and test-set evaluation with the 2D synthetic letter B dataset:

    python atlas-istn-letter-b.py

Tensorboard can be used to monitor training with:

    tensorboard --logdir=output/synth2d/full-stn/
 
#### Example 2: 3D Synthetic Cardiac Dataset
While the CCTA dataset used in the paper is not public, a synthetic 3D dataset is provided, which 
can be downloaded from [here](https://imperialcollegelondon.box.com/s/6xicbiw1wtu1uhcd5wlaqttgm5m64uc3). Unzip the data under `data/synth3d`.

To run training and test-set evaluation with a synthetic 3D cardiac dataset:  
    
    python atlas-istn-synth-cardiac.py
    
Tensorboard can be used to monitor training with:

    tensorboard --logdir=output/synth3d/full-stn/

## License
This project is licensed under the [Apache License 2.0](LICENSE).
