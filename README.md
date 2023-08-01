
# High Dynamic Range Imaging via Visual Attention Modules

Ali Reza Omrani, Davide Moroni

## Model Structure
![alt text](https://raw.githubusercontent.com/AlirezaOmrani95/HDR-VAM/main/Images/Total%20Structure.png)


## Abstract
Thanks to High Dynamic Range (HDR) imaging methods, the scope of photography has seen profound changes recently. To be more specific, such methods try to reconstruct the lost luminosity of the real world caused by the limitation of regular cameras from the Low Dynamic Range (LDR) images. Additionally, although the State-Of-The-Art methods in this topic perform well, they mainly concentrate on combining different exposures and have less attention to extracting the informative parts of the images. Thus, this paper aims to introduce a new model capable of incorporating information from the most visible areas of each image extracted by a visual attention module (VAM), which is a result of a segmentation strategy. In particular, the model, based on a deep learning architecture, utilizes the extracted areas to produce the final HDR image. The results demonstrate that our method outperformed most of the State-Of-The-Art algorithms.
## Usage

### Dataset

You can Register in Codalab and download the dataset.

```
https://codalab.lisn.upsaclay.fr/competitions 1514#participate-get-data
```
Sort the data into the following subfolders:
```
Train:
    LDR:
        short:
            short exposure files
            exposure files
        medium:
            medium exposure files
        long:
            long exposure files
    HDR:
        GT files
        align ratio files

Valid:
    $ Same subfolders as Train $
Test:
    $ Same subfolders as Train $
```

### Train

To train the model you need to change the train_dir and val_dir in the train.py file.

## Results
![alt text](https://raw.githubusercontent.com/AlirezaOmrani95/HDR-VAM/main/Images/Results.jpg)


# Citation
For more information please refer to the paper https://arxiv.org/abs/2307.14705. Additionally, if you use this code for your research, please cite our paper:

```
Omrani, Ali Reza; and Davide Moroni. "High Dynamic Range Imaging via Visual Attention Modules." arXiv preprint arXiv:2307.14705 (2023).
```
