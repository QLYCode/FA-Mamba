
# Frequency-Aware Mamba: Exploiting Frequency-Domain Priors to Alleviate Class Imbalance in Medical Image Segmentation

#### [📌] The paper is currently under review (not yet published), and the source code will be released only after official acceptance.

This repository is the official implementation of the paper SPC-Seg
## Datasets

### DDTI
For DDTI, which is challenging due to low contrast and ambiguous boundaries, we use five-fold cross-validation with mutually exclusive folds for training and validation. 

### TN3K
For TN3K, which exhibits large nodule-scale variation and class imbalance: 2,303 images for training, 576 for validation, and 614 for testing. All experiments adopt 5-fold cross-validation. Performance is evaluated by Accuracy, Dice, IoU, FLOPs, and Params.


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


## Evaluation

To evaluate the model, run this command:

``` python test.py ```
