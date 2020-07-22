# PQRST Segmentation using UNet
This is a project that can segment a EKG signal to P-wave, QRS-wave, and T-wave parts.  
First version is using UNet like Neural Network which is the implementation of this paper: [Deep Learning for ECG Segmentation](https://arxiv.org/pdf/2001.04689.pdf) 

## Data Preprocessing
### Training
Training using LUDB dataset, which have 200 subjects, 12 leads, 10 seconds and 500Hz frequency data.
preprocessing process mostly describe in that paper.
1. cut the first and last 2 seconds because LUDB don't have label at the start and the end.  
2. randomly choose 4 seconds from 6 seconds to avoid overfitting. In this project, this will generate 3 signals from 1 singal which are 2nd to 6th second, 3rd to 7th second, and 4th to 8th second segments.
3. smoothing signals using function "smooth_signal"  
### Testing
Testing using IEC dataset, which have 100 signals, 1 lead, 10 seconds and 500 Hz frequency data.  
There is no data preprocessing on testing set because I found out that with original signal input will have better performance.  

## Evaluate function
1. The evaluate method the paper used is to count the correct onset/offset prediction within tolerance value. But the tolerance value mentioned in the paper is way too high. In this project, the tolerance value is set to 15 data points (which is 30ms if data is 500Hz), and can get about 0.93 F1-score.  
2. Another evaluate method this project provided is to calculate 4 segments duration, which are P-duration, PQ-interval, QRS-duration, QT-interval. And the tolerance used here is also set to 30ms.
# PQRST Segmentation using RetinaNet1D
The second approach here is to use one stage object detector model structure to predict P-wave, QRS-wave and T-wave position. Here we simply use RetinaNet which use Resnet (SE-Net) as backbone, connect with a simple FPN. Then concatenate with location head and class head.
## Data preprocessing
### Training
Because training data have some missing labels, those missing labels' data are removed during training.
Currently there is no data preprocessing on training data. But there are some options available:
1. training data denoising using wavelet thresholding.
2. training data augmentation by adding gaussian noise or scale some part of waves.
### Testing
Testing data will denoise using wavelet thresholding before predicting.
## Evaluation
This model use the same evaluation method as previous approach. But a little bit difference to tolerance value.  
The tolerance used in second evaluation method is fixed to [10, 10, 10, 25] on mean and [15, 10, 10, 30] on standard deviation. And the first evaluation method tolerance was set to be 10ms.
## Result
This approach can get 0.968 F1-score which is better than previous approach using UNet. And can reach about 0.8 accuracy on IEC dataset using second evaluation method.

# Usage
## Dependencies
1. Use `pipenv` to install dependencies and activate virtualenv:
    ```
    pipenv sync
    pipenv shell
    ```
2. `wandb` setup:
    ```
    wandb login
    ```
## Data Preprocessing
Apply denoising and normalize to EKG.

Setup `config.cfg`
- Set `use_gpu_num` for training
- Set directories and label files for `RetinaNet`
- Set training and testing options for model training and testing
    - Set denoise option to `True` to activate denosing, otherwise set the option to `False`

## Training
```
python3 train.py --model retinanet
```

## Prediction
See [`tutorial.ipynb`](./tutorial.ipynb) for more details about prediction.