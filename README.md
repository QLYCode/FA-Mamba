
# RSR-Mamba: Recurrent Self-Reasoning forUncertainty-Aware Ultrasound Segmentation

#### [📌] The paper is currently under review; the full source code will be released upon publication.

This repository is the official implementation of the paper Frequency-Aware Mamba.

## Datasets

### DDTI
(a) TG3K contains 3,585 thyroid ultrasound frames; we use 2,758 for training and 827 for testing.


### BUSI
BUSI includes 780 breast ultrasound images, we focus on 647 benign and malignant cases and perform 5-fold cross-validation.


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

```python run.py```



