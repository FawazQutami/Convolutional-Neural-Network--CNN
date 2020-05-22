# Introduction
The objective of this project is to train a neural network that is capable of identifying fruits from images, as well as to obtain a good classifier. 
In this document I will show the technique that I have implemented to reach the desired result, with some explanations.

# Dataset
 Fruits 360 dataset – is a dataset of images containing fruits and vegetables (dated 2020.05.18). 
 This dataset is available for download from Kaggle: Fruits-360 dataset, or, from GitHub: [Fruits-360 dataset](https://github.com/Horea94/Fruit-Images-Dataset')

## Dataset properties:

	•	Total number of images	: 90483
	•	Training set size		: 67692 images
	•	Test set size			: 22688 images
	•	Number of classes		: 131 (fruits and vegetables)
	•	Image size				: 100x100 pixels

# Project Technique

## Data Preprocessing
	•	I created a “Training” and “test” directories with their respective images. 
	•	Split the images of each directory into two groups: X  and y. 
	•	Split X into train, test, and validation sets, and did the same for y. 
	•	Next, converted X sets into arrays instead of images, 
	•	Finally, rescaled X sets by dividing over 255.
	
Hence, training sets, testing sets, and validation sets are all ready and available for more further processing, and to illustrate that, we can show sample images:

![Show 16 image](/images/sample16.png)

## Convolution Neural Network (CNN) 
I will try to build a CNN for multi-class classification for the fruit’s dataset - I used KERAS package to build the CNN.
### First, we build the model:
	•	Build the input layer with dimensions: (100,100,3).
	•	Followed with a pooling layer.
	•	Stacked a 3 convolution layers.
	•	Each convolution layer followed by pooling layer (3 polling layers).
	•	Add dropout to ignore randomly selected neurons - regularization technique
	•	Flatten - converting matrix to single array.
	•	Add two fully connected layers(Dense) to produce the output, where first dense layer will 
	    use ‘relu’ activation to train the network, and the second one will use ‘softmax’ activation.

Now, we can generate the model image:

![Show the CNN Model Image](/images/modelImage.png)

### Next, we compile the model with the following:
	1.	Adam with learning rate = 0.001. (or you can use other algorithms such as: RMSprop, SGD, etc.).
	    The purpose of loss functions is to compute the quantity that a model should seek to minimize during training 
		- Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of 
		first-order and second-order moments.
	2.	A Categorical Cross-entropy function. The purpose is to compute the cross-entropy loss between 
	    true labels and predicted labels.

I have noticed that learning rate has a big impact on the accuracy of the model.

### Next, we train and fit the model:
We first train the model by fitting X and y train sets into the model, where batch size = 32 and epochs = 15 with shuffling, and then store the resulted check points, locally, in a file called ‘cnn_fruits.hdf5’.
Next, we load and evaluate the stored wights:
We load the wights from the locally saved file (cnn_fruits.hdf5’) and evaluate the model using X and y test sets to calculate the accuracy score of the model, which is 98.70% ( not bad at all!).

## Make a prediction 
Here, I will use 12 images and predict the result, If the prediction is right, then its value will be shown in green color. If not, then the value will be shown in red:

![Show the CNN Model Image](/images/sample12.png)

The result above shows that our prediction is almost good

## Plot the performance of the train and test sets - loss and accuracy vs epochs

![Show the CNN Model Image](/images/performance.png)
Accuracy shows that our prediction – using the Test set, is almost reaching the same level as the Train set with each increase in the epochs – accuracy on Train set increase as well as the accuracy on Test set, this means that the model is good and there was not overfitting.
Loss shows a decreasing in both Train and Test curves with each increase in the epochs (because of the learning rate).

## Final Notes:
I have used the common-used (basic) architecture where I stacked a few convolutional layers followed by a pooling layer(to reduce the dimensions and avoid overfitting). I wanted to increase the depth of the network with each level, so I increased the number of filters while moving forward. I tried different number of convolutional layers(3,4,5,6), and the number of nodes/filters before I achieved this model and get good accuracy.

