# Semantic Segmentation
### Introduction
The goal of this project is to construct a fully convolutional neural network (FCN) based on the VGG-16 image classifier architecture for performing semantic segmentation to identify drivable road area from an car dashcam image (trained and tested on the KITTI data set), by labelling the pixels of a road in images.

### Setup
##### Frameworks and Packages
Make sure the following are installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - [Pillow](https://python-pillow.org/)
 - [tqdm](https://pypi.python.org/pypi/tqdm)
 
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py
```

### FCN Architecture
Following the approach in the [Fully Convolutional Networds for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf) paper, my network was based on the FCN-8 architecture (right) that was built using the VGG network (left). 

<img src="./images/VGG.jpg" width="400"><img src="./images/fcn.jpg" width="400">

The encoder portion of the network consists of the convolution and pooling layers of the VGG network (pretrained model) with the final two fully connected layers replaced with 1x1 convolutions to prevent the complete loss of spatial information. The decoder portion of the network consists of 1x1 convolution, upsampling, skip layers and summation layers.

<img src="./images/fcn8.jpg" width="800">

The 1x1 convolution layers reduce the encoder's output depth from 4096 to the number of classes that the network is trained to recognize. The upsampling layers increase the encoder's output spatial dimensions from 7x7 to the original input image dimensions. The summation layers add together the upsampling and pooling layers. The pooling layers are from upstream of the encoder output and therefore contain more spatial information which improves the network's inference accuracy.

### Implementation
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)

- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. 

- The pretrained VGG-16 model is already fully convolutionalized, i.e. it already contains the 1x1 convolutions that replace the fully connected layers. THOSE 1x1 convolutions are the ones that are used to preserve spatial information that would be lost if we kept the fully connected layers. we need to add 1x1 convolutions on top of the VGG-16 network. The purpose of the 1x1 convolutions that we are adding on top of the VGG is merely to reduce the number of filters from 4096 to whatever the number of classes for our model is, that is all. 

- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. To include the scaling layers, simply add them to your model like so:

        pool3_out_scaled = tf.multiply(pool3_out, 0.0001, name=‘pool3_out_scaled’)
        pool4_out_scaled = tf.multiply(pool4_out, 0.01, name=‘pool4_out_scaled’)

    where pool3_out and pool4_out are the outputs of the VGG-16. You then feed the scaled outputs into your 1x1 convolutions and everything is as before from there.

- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to the loss function, otherwise regularization is not implemented. To compute the total loss of the whole network, manually add the regularization loss as follows:

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.01  # Choose an appropriate one.
        loss = my_normal_loss + reg_constant * sum(reg_losses)
        
- The model was trained with a batch size of 10 and in 10 epochs, using dropout with a keep probability of 0.5. 
    The network was trained to recognize two classes: road and not road.
    The loss function for the network is cross-entropy, and an Adam optimizer is used.

The following are the final network's parameters and hyperparameters:

| (Hyper)Parameter                  | Value   |
| --------------------------------- |--------:|
| Number of classes                 | 2       |
| Epochs                            | 10      |
| Batch size                        | 10      |
| Initialization standard deviation | 1e-2    |
| Regularization scale              | 1e-3    |
| Dropout keep probability          | 0.5     |
| Adam learning rate                | 0.0009  |


**Conclusion:**
The result from this semantic segmentation project are satisfactory as the model can label most pixels of the road close to the best solution. In case of identifying only the road plane, the results here can be combined with an advanced lane finding pipeline to obtain more accurate road plane segmentation.

