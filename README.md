
## Frequency-Aware Mamba: Exploiting Frequency-Domain Priors to Alleviate Class Imbalance in Medical Image Segmentation

👉 Accepted at **IEEE ICASSP 2026**: This repository is the official implementation of the paper Frequency-Aware Mamba.

## Datasets

### DDTI
The DDTI dataset is characterized by low contrast and ambiguous boundaries, making it challenging for precise segmentation. Following prior work, we adopt **five-fold cross-validation** for evaluation.

### TN3K
The TN3K dataset exhibits large-scale variation and class imbalance. We follow the official split in [1]: 2,303 training, 576 validation, and 614 testing images.



## Requirements

Some important required packages include:
* Python 3.8
* CUDA 11.8
* causal-conv1d 1.2.0.post2
* einops 0.8.0
* imageio 2.35.1
* mamba-ssm 2.2.2
* matplotlib 3.7.5
* medpy 0.5.2
* nibabel 5.2.1
* opencv-python 4.11.0.86
* pillow 10.2.0
* scikit-image 0.21.0
* scipy 1.10.1
* simpleitk 2.4.1
* six 1.17.0
* tensorboardx 2.6.2.2 
* torch 2.0.0+cu118
* torchaudio 2.0.1+cu118
* torchvision 0.15.1+cu118
* tqdm 4.67.1 

## Training

To train the model, run this command:

```python train.py```



