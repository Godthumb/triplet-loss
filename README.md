# Triplet-loss-pytorch
Implement triplet-loss

## Introduction
![](figures/triplet-loss.png)

The triplet loss is a great choice for classification problems with N_CLASSES >> N_SAMPLES_PER_CLASS. For example, face recognition problems.

![](figures/train-steps.png)

The CNN architecture we use with triplet loss needs to be cut off before the classification layer. In addition, a L2 normalization layer has to be added.

## Usage
```
python train.py
python extract_embeddings.py
python model_on_top.py
```
The default model is resnet18

## Acknowledgement
Thanks for great inspiration from [https://github.com/alfonmedela/triplet-loss-pytorch/tree/master](https://github.com/alfonmedela/triplet-loss-pytorch/tree/master) and [https://github.com/chencodeX/triplet-loss-pytorch](https://github.com/chencodeX/triplet-loss-pytorch)

## License
All code within the repo is under [MIT license](https://mit-license.org/)