# RCDNet: An Interpretable Rain Convolutional Dictionary Network for Single Image Deraining
 
[Hong Wang](https://hongwang01.github.io/), Qi Xie, Qian Zhao, Yuexiang Li, Yong Liang, Yefeng Zheng, and [Deyu Meng](http://gr.xjtu.edu.cn/web/dymeng)

 [[Google Drive]](https://drive.google.com/file/d/1aWpc0xsGXqjyOBqL5NHweP-i0EccDnoV/view?usp=sharing)[[Arxiv]](https://arxiv.org/abs/2107.06808) 
 
We extend the previous work [RCDNet](https://github.com/hongwang01/RCDNet) to the training-testing mismatch case. In this work, we propose a dynamic rain kernel prediction mechanism, which makes the DRCDNet have the potential to obtain better generalization performance.



## Abstract
As a common weather, rain streaks adversely degrade the image quality and tend to negatively affect the performance
of outdoor computer vision systems. Hence, removing
rains from an image has become an important issue in the field.
To handle such an ill-posed single image deraining task, in this
paper, we specifically build a novel deep architecture, called rain convolutional dictionary network (RCDNet), which embeds the
intrinsic priors of rain streaks and has clear interpretability. In specific, we first establish a RCD model for representing rain streaks and utilize the proximal gradient descent technique to design an iterative algorithm only containing simple operators for solving the model. By unfolding it, we then build the RCDNet in which every network module has clear physical meanings and corresponds to each operation involved in the algorithm. This good interpretability greatly facilitates an easy visualization and analysis on what happens inside the network and why it works well in inference process. Moreover, taking into account the domain gap issue in real scenarios, we further design a novel dynamic RCDNet, where the rain kernels can be dynamically inferred corresponding to input rainy images so as to ensure a fine generalization performance in the inconsistent scenarios of rain types between training and testing data. By end-toend training such an interpretable network, all involved rain kernels and proximal operators can be automatically extracted, faithfully characterizing the features of both rain and clean background layers, and thus naturally lead to better deraining performance. Comprehensive experiments implemented on a series of representative synthetic and real datasets substantiate the superiority of our method, especially on its well generality to diverse testing scenarios and good interpretability for all its modules, as compared with state-of-the-art single image derainers both visually and quantitatively.

## Motivation
<img src="motivation.png" height="50%" width="100%" alt=""/>

## Dynamic Rain Kernel Inference
<img src="dynamic rain kernel.png" height="50%" width="100%" alt=""/>

## Rain Removal Performance 

### Training-Testing Match Case:
<img src="match.png" height="50%" width="100%" alt=""/>

### Training-Testing Mismatch Case:
<img src="generalization_dense.png" height="50%" width="100%" alt=""/>

<img src="generalization.png" height="50%" width="100%" alt=""/>

## Dataset & Training & Testing
Please refer to the original coding framework [RCDNet](https://github.com/hongwang01/RCDNet) or the simple framework [RCDNet_simple](https://github.com/hongwang01/RCDNet_simple) to execute the training and testing process.



## Citations

```
@article{ji2021survey,
  title={RCDNet: An Interpretable Rain Convolutional Dictionary Network for Single Image Deraining},
  author={Wang, Hong and Xie, Qi and Zhao, Qian and Li, Yuexiang and Yong, Liang and Zheng, Yefeng and Meng, Deyu},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}
}
```

## Contact
If you have any question, please feel free to concat Hong Wang (Email: hongwang01@stu.xjtu.edu.cn)
