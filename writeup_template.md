# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2016_12_01_13_31_15_308.jpg "Center Image"
[image3]: ./examples/left_2016_12_01_13_35_38_405.jpg "Recovery Image"
[image4]: ./examples/right_2016_12_01_13_35_38_405.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/normal.png "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3  image   							| 
| Cropping     	|  outputs 65x320x3 	|
| Convolution (relu)				|	5x5 stride,	Outputs 61x316x6										|
| Max pooling	      	| 2x2 stride,  outputs 30x158x6 				|
| Convolution (relu)	    |  outputs 26x158x6 |
| Max Pooling |  outputs 13x77x6 |
| Flatten | 6006 |
| Fully Connected |  Output: 60 |
| Dropout | 0.2 |
|Fully Connected |  Output: 10 |
|Fully Connected | Output: 1 |



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 56). 

The model was trained and validated on large dataset which was created using data augmentation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 60).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to achieve a least loss possible and avoiding overfitting

My first step was to use a convolution neural network model similar to the traffic classifier project. I thought this model might be appropriate because it has dropout layers which helps the model to generalize better.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that added Dropout layers which helps the model to generalize better.

Then I used data augmentation in which I used the same dataset to generate more training samples by simply reversing the image using np.fliplr and adding negative to the corresponding measurement or the angle.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like after the bridge and to improve the driving behavior in these cases, I trained the model again with few different parameters

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 46-58) consisted of a convolution neural network with the following layers and layer sizes | Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3  image   							| 
| Cropping     	|  outputs 65x320x3 	|
| Convolution (relu)				|	5x5 stride,	Outputs 61x316x6										|
| Max pooling	      	| 2x2 stride,  outputs 30x158x6 				|
| Convolution (relu)	    |  outputs 26x158x6 |
| Max Pooling |  outputs 13x77x6 |
| Flatten | 6006 |
| Fully Connected |  Output: 60 |
| Dropout | 0.2 |
|Fully Connected |  Output: 10 |
|Fully Connected | Output: 1 |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I then preprocessed this data by cropping the image by adding a cropping layer, since excess data is provided in the top such as sky, trees which are unneccesary.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by model.fit() I used an adam optimizer so that manually training the learning rate wasn't necessary.
