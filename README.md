# Sharpness-Aware-Minimization-TensorFlow
This repository provides a minimal implementation of sharpness-aware minimization (SAM) ([Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)) in TensorFlow 2. SAM is motivated by the connections between the geometry of the loss landscape of deep neural networks and their generalization ability. SAM attempts to simultaneously minimize loss value as well as loss curvature thereby seeking parameters in neighborhoods having uniformly low loss value. This is indeed different from traditional SGD-based optimization that seeks parameters having low loss values on an individual basis. The figure below (taken from the original paper) demonstrates the effects of using SAM - 

<p align="center">
<img src="https://i.ibb.co/1zP7gJN/image.png" width=700></img>
</p>

My goal with this repository is to be able to quickly train neural networks with and without SAM. All the experiments are shown in the `SAM.ipynb` notebook ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayakpaul/Sharpness-Aware-Minimization-TensorFlow/blob/main/SAM.ipynb)). The notebook is end-to-end executable on Google Colab. Furthermore, they utilize the free TPUs (TPUv2-8) Google Colab provides allowing readers to experiment very quickly.

## Notes

Before moving to the findings, please be aware of the following notable differences in my implementation:

* ResNet20 (attributed to [this repository](https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/blob/master/zoo/resnet/resnet_cifar10.py)) is used as opposed to PyramidNet and WideResNet. 
* ShakeDrop regularization has not been used.
* Two simple augmentation transformations (random crop and random brightness) have been used as opposed to Cutout, AutoAugment. 
* Adam has been used as the optimizer with the default arguments as provided by TensorFlow with a `ReduceLROnPlateau`. Table 1 of the original paper suggests using SGD with different configurations. 
* Instead of training for full number of epochs I used early stopping with a patience of 10.

SAM has only one hyperparameter namely `rho` that controls the neighborhood of the parameter space. In my experiments, it's defaulted to 0.05. For other details related to training configuration (i.e. network depth, learning rate, batch size, etc.) please refer to the notebooks.

## Findings

|             | Number of Parameters (million) | Final Test Accuracy (%) |
|-------------|:------------------------------:|:-----------------------:|
|   With SAM  |            0.575114            |           80.5          |
| Without SAM |            0.575114            |           83.1          |


## Acknowledgements
* David Samuel's [PyTorch implementation](https://github.com/davda54/sam)
