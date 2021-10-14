# IRIS Flower prediction using Tensorflow
In this code, I want you to show how you can use the **Tensorflow** to train **Iris Dataset**, a model that can categorize data.

## Classification Algorithm

As we know, the Supervised Machine Learning algorithm can be broadly classified into Regression and Classification Algorithms. In Regression algorithms, we predict the output for continuous values, but to predict the categorical values, we need Classification algorithms.

The Classification algorithm is a Supervised Learning technique that is used to identify the category of new observations on the basis of training data. In Classification, a program learns from the given dataset or observations and then classifies new observation into a number of classes or groups. Such as, Yes or No, 0 or 1, Spam or Not Spam, cat or dog, etc. Classes can be called as targets/labels or categories.

Unlike regression, the output variable of Classification is a category, not a value, such as "Green or Blue", "fruit or animal", etc. Since the Classification algorithm is a Supervised learning technique, hence it takes labeled input data, which means it contains input with the corresponding output.

The algorithm which implements the classification on a dataset is known as a classifier. There are two types of Classifications:

   1.  **Binary Classifier**: If the classification problem has only two possible outcomes, then it is called as Binary Classifier.
       Examples: YES or NO, MALE or FEMALE, SPAM or NOT SPAM, CAT or DOG, etc.
    2. **Multi-class Classifier**: If a classification problem has more than two outcomes, then it is called as Multi-class Classifier.
       Example: Classifications of types of crops, Classification of types of music.

## IRIS Dataset
This is probably the most versatile, easy and resourceful data set in pattern recognition literature. Nothing could be simpler than iris data set to learn classification techniques. If you are totally new to data science, this is your start line. The data has only 150 rows & 4 columns.
The data set contains 50 records of 3 species of Iris:
- Iris virginica 
- Iris setosa
- Iris versicolor
<img style="float: center;" width = 500px; src="http://python.astrotech.io/_images/iris-flowers.png"/>

Each records contains 4 features:

Sepal length

Sepal width

Petal length

Petal width

and each record has a species (class) assigned.

# Problem:
Predict the flower class based on available attributes.

# Prerequisites
1. Tensorlfow
2. Numpy
3. Urlib

# API Description
TensorFlow’s API (tf.contrib.learn) is used to configure, train, and evaluate the models. In this, we’ll use **tf.contrib.learn** to construct a neural network classifier and train it on the **Iris data set** to predict flower species based on sepal/petal geometry.

In this firstly we'll load the iris csv's data to Tensorflow, the iris dataset contains 150 samples of data in which 50 samples of each flower species. In this 120 samples are of train dataset and 30 samples are of test dataset.

Next, we'll construct a neural network classifier using the **tf.contrib.learn**

tf.contrib.learn offers the variety of predifined models, called **Estimator** which can be used for training and evaluating the model.

Now then after configuring our model, now we'll fit the model using the **fit** method. Now we have fit our model on the training data; now we can check its accuracy using the **evaluate** method.

To classify new samples we'll use **predict()** method. The model thus predict the following samples.
The same can be extended to **Keras** and **PyTorch** frameworks.
Another simple solution is to use **Naive Bayes** algorithm.
