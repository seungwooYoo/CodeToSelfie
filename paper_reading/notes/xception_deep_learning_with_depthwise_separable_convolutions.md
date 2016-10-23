# Xception: Deep learning with depthwise separable convolutions

--------------------
### Introduction
- The idea behind Inception module is as follows : 
  - For example, the typical inception module firstly looks at cross-cahnnel correlations via a set of 1x1 convolutions (reduce original input space). And then, use 3x3 or 5x5 convolutions to map all correlations in these smaller 3D spaces
  - Cross-channle correlations and spatial correlations are sufficiently decoupled - it is preferable not to map them jointly
- Extreme version of Inception module - use 1x1 convolution to map cross channel correlations - then separately map the spatial correaltions of every ouput channel (almost identical to so called as depthwise separable convolution)
  - Difference between depthwise seprable convolution (separable convolution) and extreme version of inception module
    - The order of the operations : Depthwise separable convolutions peform first channel wise spatial convolution and then perform 1x1 convlution (inception module performs the 1x1 convolution first)
    - The presence or absence of a non-linearity after the first operation (inception module use ReLU in every operations / depth wise separable convolutions are usually implmented without non-lineariies) 
- Suggest to replace Inception modules with depthwise separable convolutions 
 
### Prior work
- Depthwise separable convolutions 
  - Start from 2013 in Google Brain. Used them in AlexNet to obtain small gains in accuracy and large gains in convergence speed, as well as a significant reduction in model size
  - Currenty, this work is only possible due to the inclusion of an efficient implementation of depthwise separable convolutions in TensorFlow framework  

### The Xception architecture
- Extreme Inception (Xception)
  - Hypothesis : The mapping of cross channels correlations and spatial correlations in the feature maps of convolutional neural networks can be enirely decoupled 
  - 36 convolutional layers - structured into 14 modules, all of which have linear residual connections around them (except for the first and the last modules)
  - A linear stack of depthwise separable convolution layers with residual connections
  - Absence of any nonlinearity leads to both faster convergence and better final performance
    - Depth of the intermediate feature spaces are helpfule of non-linearity (i.e., for deep feature spaces (found in Inception modules), the nonlinearity is helpful, but for shallow ones (depthwise separable convolutions) it becomes harmful) 
