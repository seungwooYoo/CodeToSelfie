# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

--------------------
### Introduction
- SR problem has ill-posed nature
- Optimization target of supervised SR 
    - Usually the minimization of the MSD between HR and ground truth 
    - It is convenient : minimizing MSE -> maxizing PSNR (common measure used to evaluate SR algorithm)
    - However, MSE is limited since the measure is defined based on the pixel-wise image differences
- Propose SR generative adversarial network (SRGAN) : deep residual network + novel perceptual loss using feature maps of the VGG network 
 
### Contribution  
- Firstly using deep ResNet architecutre using the concept of GANs
- Using a combination of content loss and adversarial loss 
    - Content loss : euclidean distance between the last convolutional feature maps of the VGG networks 
    - Adversarial loss : From adversarial learning
    - Total variation loss - employ a regularizer to encourage spatially coherent solutions

