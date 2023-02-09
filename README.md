# Image Segmentation Using U-Net

Image segmentation is the process of classifying the pixels in a digital image to different classes. There are two types of image segmentation : Semantic Segmentation and Instance Segmentation.

In semantic segmentation, pixels belonging to a given class as classified as one label whereas in instance segmentation each instance of a specific class are assigned a separate label.

![](https://www.sentisight.ai/wp-content/uploads/2022/08/segmentation-example.png)

In this repo I have implemented a simple U-net to demonstrate the use case of image segmentation.

## U-Net

U-Net, as the name suggest follows an encoder - decoder based model with decoder mirroring the encoders and a bottleneck layer. It was first introduced by Olaf Ronneberger et. al. in their paper [U-Net: Convolutional Network for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf) with training strategy that relies on the strong use of data augmentation and use the available annotated samples more efficiently.

![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

Each encoder block has two conv layer followed by a max pool layer and each decoder block has two conv layers followed by upsampling layers along with the output of the corresponding encoder layer as shown in the image above. 


## Dataset

The custom dataset used in this implementation was created by [Divam Gupta](https://github.com/divamgupta/image-segmentation-keras) using [CityScape Dataset](https://disq.us/url?url=https%3A%2F%2Fwww.cityscapes-dataset.com%2Fdataset-overview%2F%3A6phlceP6Z-8-tPaIGijFHjEViv0&cuid=5799521). He also has an excellent blog on [Semantic Segmentation using Keras](https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html). It has 367 annotated images of size 480p by 360p in train set and 101 annotated images of size 480p by 360p in validation set. The annotated image has pixels classified in 12 classes - 'sky', 'building','column/pole', 'road', 'side walk', 'vegetation', 'traffic light', 'fence', 'vehicle', 'pedestrian', 'byciclist', 'void'.

## Method

A U-Net architecture with input and output shape of 128X128X3 was used. It has a down sampler or 2 encoder blocks of each - 64,128,256 and 512 filters
and upsampler or 2 decoder blocks of each - 512,256,128 and 64 filters. It also uses a bottleneck of 2 1024 filters. The model had ~34.5M parameters and was trained for 20 epochs with Adam optimizer and catergorical cross entropy as loss function.

## Results

With the above method following results were achieved
![](https://github.com/Ayush-Mi/Image-Segmentation-Using-U-Net/tree/main/images/accuracy_plot.png) 
![](https://github.com/Ayush-Mi/Image-Segmentation-Using-U-Net/tree/main/images/loss_plot.png)

![](https://github.com/Ayush-Mi/Image-Segmentation-Using-U-Net/tree/main/images/output.png)

## Applications

Image segmentation can has application under different sector
- Medical Imaging
- Autonomous Driving
- Satellite Imaging

## Reference
1. [Divam Gupta](https://github.com/divamgupta/image-segmentation-keras)
2. [CityScape Dataset](https://disq.us/url?url=https%3A%2F%2Fwww.cityscapes-dataset.com%2Fdataset-overview%2F%3A6phlceP6Z-8-tPaIGijFHjEViv0&cuid=5799521)
3. [CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
4. [U-Net](https://arxiv.org/abs/1505.04597)
