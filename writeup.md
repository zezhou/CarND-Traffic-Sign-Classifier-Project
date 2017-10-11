#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[class_sample]: ./examples/class_sample_image.png "samples of class"
[data_explore]: ./examples/data_explore.png "data exploring"
[grayscaling]: ./examples/grayscaling.png "gray scaling"
[augment_new]: ./examples/data_augmenting2.png "origin image of augment"
[augment_origin]: ./examples/data_augmenting.png "generated image of augment" 
[normalize]: ./examples/normalize.png "normalize effect" 
[web_images]: ./examples/web_image.png "images found from web" 

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zezhou/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
I print out one image for each class.

![sample of class][class_sample]

We can see that the brightness of images is very different. This increases the complexity of prediction. We can use gray scaling to eliminate this problem. 

It is a bar chart showing how the data class distribution.

![data exploring][data_explore]

We can see that the class distribution in train and test sets is similary. Besides, the train dataset is not balanced, which may decreases the accuracy of the model predicting. This problem can be sovled by data augmenting.


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the brightness of samples are different, and the brightness between the training data and reality images are also different. However, the brightness of image is no relation  with the content of image. This means that the brightness of image increases the complexity of predicting. The gray scale can eliminate this effect, and make predicting accuracy improve.

Here is an example of a traffic sign image before and after grayscaling.

![grayscaling][grayscaling]


Secondly, I decided to generate additional data because the training dataset is unbalanced, which makes some classes hard to be predicted. With additional data, we can get a robustly model.   

To add more data to the the data set, I used the image brightness and image transforming techniques. 

Here is an example of an original image and an augmented image:

![original image][augment_origin]
![aaugment image][augment_new]

The difference between the original data set and the augmented data set is the brightness and transform angle of image. 

Thirdly, I normalized the image data because it can make different features similar distributionbs which benifits to the training and accuracy. 

Here is an example of a traffic sign image before and after normalizing.

![normalize image][normalize]

We can see that the image has a little changes.

As a last step, I use shuffling function to randomize train datasets and valid datasets. The images in training dataset and valid datasets were taken from the same vehicle at close intervals, thus the adjacent images are similar. Shuffling them before training can break their connectings and hugely improve the finally performance.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I use LeNet in the course with some improvements including dropout and layer-briging inspired by this paper(http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRAY image   							| 
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs  14x14x6 				|
| Convolution 1x1	    | 1x1 stride, valid padding, outputs 10x10x16   |
| Max pooling	      	| 2x2 stride,  outputs  5x5x16                  |
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam algorithm. Optimizer uses backpropagation to update the network and minimize training loss. I used following training paramenters:
* EPOCHS = 100 
* BATCH_SIZE = 128
* learning rate = 0.001
* dropout = 0.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![images found from web][web_images]

The second image might be difficult to classify because it contains extra content which lacks in the training data.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work      		| Road Work   									| 
| Speed limit 50    	| Roundabout mandatory               			|
| Stop sign				| Stop sign										|
| Turn right ahead   	| Turn right ahead					 			|
| No entry  			| No entry          							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



