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

### Architecture
- Following the approach in the [Fully Convolutional Networds for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf) paper, the final model consisted of an encoder part using the VGG pretrained model and a decoder part obtained by upsampling and skip layers to output same size as the input image.

- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)

- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. 

- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. To include the scaling layers, simply add them to your model like so:

        pool3_out_scaled = tf.multiply(pool3_out, 0.0001, name=‘pool3_out_scaled’)
        pool4_out_scaled = tf.multiply(pool4_out, 0.01, name=‘pool4_out_scaled’)

    where pool3_out and pool4_out are the outputs of the VGG-16. You then feed the scaled outputs into your 1x1 convolutions and everything is as before from there.

- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented. To compute the total loss of the whole network for some batch inputs, manually add the regularization loss as follows:

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

