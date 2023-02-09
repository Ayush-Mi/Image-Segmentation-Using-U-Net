# Image Segmentation Using U-Net

Image segmentation is the process of classifying the pixels in a digital image to different classes. There are two types of image segmentation : Semantic Segmentation and Instance Segmentation.

In semantic segmentation, pixels belonging to a given class as classified as one label whereas in instance segmentation each instance of a specific class are assigned a separate label.

![](https://www.sentisight.ai/wp-content/uploads/2022/08/segmentation-example.png)

In this repo I have implemented a simple U-net to demonstrate the use case of image segmentation.

## U-Net

## Dataset

The custom dataset used in this implementation was created by [Divam Gupta](https://github.com/divamgupta/image-segmentation-keras) using [CityScape Dataset](https://disq.us/url?url=https%3A%2F%2Fwww.cityscapes-dataset.com%2Fdataset-overview%2F%3A6phlceP6Z-8-tPaIGijFHjEViv0&cuid=5799521). He also has an excellent blog on [Semantic Segmentation using Keras](https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html). It has 367 annotated images of size 480p by 360p in train set and 101 annotated images of size 480p by 360p in validation set. The annotated image has pixels classified in 12 classes - 'sky', 'building','column/pole', 'road', 'side walk', 'vegetation', 'traffic light', 'fence', 'vehicle', 'pedestrian', 'byciclist', 'void'.

## Method

## Results

## Applications

## Reference
