# PQRST Segmentation usisng UNet
This is a project that can segment a EKG signal to P-wave, QRS-wave, and T-wave parts.  
First version is using UNet like Neural Network which is the implemntation of this paper: [Deep Learning for ECG Segmentation](https://arxiv.org/pdf/2001.04689.pdf) 

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
The second approach here is to use one stage object detector model structure to predict P-wave, QRS-wave and T-wave position. Here we simply use RetinaNet which use Resnet as backbone, connect with a simple FPN. Then concatenate with location head and class head.
## Data preprocessing
Currently there is no data preprocessing on training data and testing data.
## Result
This approach can get 0.968 F1-score which is better than previous approach using UNet.