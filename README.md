# Fork from https://github.com/divamgupta/image-segmentation-keras

# Image Segmentation Keras : Implementation of Segnet, FCN, UNet, PSPNet in Keras.

Implementation of various Deep Image Segmentation models in keras.

Link to the full blog post with tutorial : https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html

# Models

Following models are supported:

| model_name       | Base Model        | Segmentation Model |
|------------------|-------------------|--------------------|
| fcn_32_vgg       | VGG 16            | FCN32              |
| vgg_pspnet       | VGG 16            | PSPNet             |
| vgg_unet         | VGG 16            | U-Net              |
| vgg_segnet       | VGG 16            | Segnet             |


# Getting Started

## Prerequisites

* Keras 2.0
* opencv for python
* Theano / Tensorflow / CNTK

```shell
sudo apt-get install -y libsm6 libxext6 libxrender-dev
pip3 install opencv-python
```

## Installing

```shell
git clone https://github.com/divamgupta/image-segmentation-keras
cd image-segmentation-keras
python3 setup.py install
```

## Preparing the data for training

You need to make two folders

*  Images Folder - For all the training images
* Annotations Folder - For the corresponding ground truth segmentation images

The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same.

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

Example code to generate annotation images :

```python
import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )
```

Only use bmp or png format for the annotation images.

## Download the sample prepared dataset

Download and extract the following:

https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing

You will get a folder named dataset1/


## Using the python module

### You can import keras_segmentation in  your python script and use the API

```python
from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=51 ,  input_height=416, input_width=608  )

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )

```
### For training
Config in file `test.py` and run 
```python
python3 test.py
```
### For predict
Config in file `testLoadModel.py` and run 
```python
python3 testLoadModel.py
```
### For visualize result in predict
Config in file `test2.py` and run 
```python
python3 test2.py
```
# Usage via command line
You can also use the tool just using command line

## Visualizing the prepared data

You can also visualize your prepared annotations for verification of the prepared data.


```shell
python -m keras_segmentation verify_dataset \
 --images_path="dataset1/images_prepped_train/" \
 --segs_path="dataset1/annotations_prepped_train/"  \
 --n_classes=50
```

```shell
python -m keras_segmentation visualize_dataset \
 --images_path="dataset1/images_prepped_train/" \
 --segs_path="dataset1/annotations_prepped_train/"  \
 --n_classes=50
```


## Training the Model

To train the model run the following command:

```shell
python -m keras_segmentation train \
 --checkpoints_path="path_to_checkpoints" \
 --train_images="dataset1/images_prepped_train/" \
 --train_annotations="dataset1/annotations_prepped_train/" \
 --val_images="dataset1/images_prepped_test/" \
 --val_annotations="dataset1/annotations_prepped_test/" \
 --n_classes=50 \
 --input_height=320 \
 --input_width=640 \
 --model_name="vgg_unet"
```

Choose model_name from the table above


## Getting the predictions

To get the predictions of a trained model

```shell
python -m keras_segmentation predict \
 --checkpoints_path="path_to_checkpoints" \
 --input_path="dataset1/images_prepped_test/" \
 --output_path="path_to_predictions"

```

## Model Evaluation 

To get the IoU scores 

```shell
python -m keras_segmentation evaluate_model \
 --checkpoints_path="path_to_checkpoints" \
 --images_path="dataset1/images_prepped_test/" \
 --segs_path="dataset1/annotations_prepped_test/"
```



# Fine-tuning from existing segmentation model

The following example shows how to fine-tune a model with 10 classes .

```python
from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50

pretrained_model = pspnet_50_ADE_20K()

new_model = pspnet_50( n_classes=51 )

transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model

new_model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)


```

# Projects using keras-segmentation
Here are a few projects which are using our library :
* https://github.com/quyvsquy/demoUnet_Segnet_FCN_PSPnet_  **This Project**
* https://github.com/SteliosTsop/QF-image-segmentation-keras [paper](https://arxiv.org/pdf/1908.02242.pdf)
* https://github.com/willembressers/bouquet_quality
* https://github.com/jqueguiner/image-segmentation
* https://github.com/pan0rama/CS230-Microcrystal-Facet-Segmentation
* https://github.com/theerawatramchuen/Keras_Segmentation
* https://github.com/neheller/labels18
* https://github.com/Divyam10/Face-Matting-using-Unet
* https://github.com/shsh-a/segmentation-over-web
* https://github.com/chenwe73/deep_active_learning_segmentation
* https://github.com/vigneshrajap/vision-based-navigation-agri-fields
* https://github.com/ronalddas/Pneumonia-Detection
* https://github.com/Aiwiscal/ECG_UNet
* https://github.com/TianzhongSong/Unet-for-Person-Segmentation
* https://github.com/Guyanqi/GMDNN
* https://github.com/kozemzak/prostate-lesion-segmentation
* https://github.com/lixiaoyu12138/fcn-date
* https://github.com/sagarbhokre/LyftChallenge
* https://github.com/TianzhongSong/Person-Segmentation-Keras
* https://github.com/divyanshpuri02/COCO_2018-Stuff-Segmentation-Challenge
* https://github.com/XiangbingJi/Stanford-cs230-final-project
* https://github.com/lsh1994/keras-segmentation
* https://github.com/SpirinEgor/mobile_semantic_segmentation
* https://github.com/LeadingIndiaAI/COCO-DATASET-STUFF-SEGMENTATION-CHALLENGE
* https://github.com/lidongyue12138/Image-Segmentation-by-Keras
* https://github.com/laoj2/segnet_crfasrnn
* https://github.com/rancheng/AirSimProjects
* https://github.com/RadiumScriptTang/cartoon_segmentation
* https://github.com/dquail/NerveSegmentation
* https://github.com/Bhomik/SemanticHumanMatting
* https://github.com/Symefa/FP-Biomedik-Breast-Cancer
* https://github.com/Alpha-Monocerotis/PDF_FigureTable_Extraction
* https://github.com/rusito-23/mobile_unet_segmentation
* https://github.com/Philliec459/ThinSection-image-segmentation-keras
If you use our code in a publicly available project, please add the link here ( by posting an issue or creating a PR )

