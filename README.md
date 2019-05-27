# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./img.jpg "Recovery Image"
[image3]: ./examples/model.png "Recovery Image"
[image4]: ./examples/first_image_uncropped.jpg "Recovery Image"
[image5]: ./examples/first_image_cropped.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* train.py containing the actual neural net architecture and training files
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py and train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (train.py lines 22-38) 

The model includes ELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer (code line 15). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (train.py lines 42 and 51). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py  lines 64-127). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually but learning rate was reduced(0.00001) (train.py line 81).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a the full dataset provided as sample along with image augmentation, generating data on my own with simulator did not work very well so I skipped that part initially and then went on working with already provided data

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to look into already existing working models

My first step was to use a convolution neural network model similar to the Nvidia paper (https://devblogs.nvidia.com/deep-learning-self-driving-cars/), I thought this model might be appropriate because it was suggested and shown to be working well on our lake data in simulator

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my my inidial models has a little higher train accuracy than validation accuracy and the model was not following the lanes all the time, so I used a bigger network and increased the training data.

To combat the overfitting, I modified the model so that it had dropout layers at two points.

Then I used used opencv image flip function to reduce left-right bias and thus reduce overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track with the initial models but with the Nvidia type model this issue was rectified.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 158, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 33, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 77, 36)        21636     
_________________________________________________________________
activation_2 (Activation)    (None, 15, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 37, 48)         43248     
_________________________________________________________________
activation_3 (Activation)    (None, 6, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 35, 64)         27712     
_________________________________________________________________
activation_4 (Activation)    (None, 4, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 64)         36928     
_________________________________________________________________
activation_5 (Activation)    (None, 2, 33, 64)         0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 2, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               540800    
_________________________________________________________________
activation_6 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
activation_7 (Activation)    (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
_________________________________________________________________
activation_8 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 681,065
Trainable params: 681,065
Non-trainable params: 0
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the full dataset provided as sample and used data augmentation techniques such as image flipping.

![alt text][image2]


To augment the data sat, I also flipped images and angles thinking that this would it would remove the left-right bias present in image and give me more data

After the collection process, I had around 40000 number of data points. I then preprocessed this data by by cropping and mean normalisation.  

![alt text][image4]
![alt text][image5]  

I finally randomly shuffled the data set and put 15% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by converge of accuracy and loss vlaues. I used an adam optimizer so that manually training the learning rate wasn't necessary.
