# MoDL

PyTorch implementation of MoDL: Model Based Deep Learning Architecture for Inverse Problems (Not official!)

Official code: https://github.com/hkaggarwal/modl

## Reference paper: 

MoDL: Model Based Deep Learning Architecture for Inverse Problems  by H.K. Aggarwal, M.P Mani, and Mathews Jacob in IEEE Transactions on Medical Imaging,  2018 

Link: https://arxiv.org/abs/1712.02862

IEEE Xplore: https://ieeexplore.ieee.org/document/8434321/

## Dataset

The brain dataset used in the original paper is publically available. You can download the dataset from the following link and locate in under the `data` directory.

**Download Link** : https://drive.google.com/file/d/1qp-l9kJbRfQU1W5wCjOQZi7I3T6jwA37/view?usp=sharing

## Configuration file

The configuration files are in `config` folder. Every setting is the same as the paper.

Configuration files for K=1 and K=10 are provided. The authors trained the K=1 model first, and then trained the K=10 models using the weights of K=1 model.

## Train

You can change the configuration file for training by modifying the `train.sh` file.

```
scripts/train.sh
```

## Test

You can change the configuration file for testing by modifying the `test.sh` file.

```
scripts/test.sh
```

## Saved models

The saved models for K=1 and K=10 are `workspace/base_modl,k=1/checkpoints/final.epoch0049-score37.3514.pth` and `workspace/base_modl,k=10/checkpoints/final.epoch0049-score39.6311.pth`
