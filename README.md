# UDOS: Open-world Instance Segmentation with Bottom-up Supervision

[[project page](https://tarun005.github.io/UDOS/)] [[paper](https://arxiv.org/pdf/)]

UDOS (**U**p-**D**own **O**pen-world **S**egmentation) is a simple and efficient method for open-world instance segmentation to detect and segment novel objects unseen during training. We leverage bottom-up supervision from unsupervised proposal generation method such as selective search and guide it to learn top-down networks such as MaskRCNN to segment open-world objects not annotated during training. UDOS first predicts part-mask associated with both seen and unseen objects, which is followed by lightweight grouping and refinement modules to predict instance level masks. 

## Overview of UDOS

<center><img src="./assets/problemFig.png" width="90%"></center>

## Testing Pre-trained models

## Training UDOS

## Citation

You can cite our work as follows.
```
@article{kalluri2023udos
        author    = {Kalluri, Tarun and Wang, Weiyao and Wang, Heng and Chandraker, Manmohan and Torresani, Lorenzo and Tran, Du},
        title     = {Open-world Instance Segmentation: Top-down Learning with Bottom-up Supervision},
        journal   = {arxiv},
        year      = {2023},
      },
```

