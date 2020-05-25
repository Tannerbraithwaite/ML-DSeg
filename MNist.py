##MNIst.py
##Problem
##with an input of 70,000 number, create an algorithm that categorizes the numbers
##Solution
##think of each number as a 28x28 matrix matrix where each value can take a value between 0-255
##Flatten each image into a vector length of 784(28x28) where each input is the intensity of the algorithm
##We can use two hidden layers for good accuracy
##output layer will have 10 units(numbers 0-9)
##we can use a softmax algorithm for the output layers
##step 1)prepocess our data, create training, validation, and testing data sets
##step 2) outline the model and choose our activation function
##step 3) set the appropriate advanced optimizers and the loss functions
##step 4)MAKE IT LEARN! back propogate to accuracy
##step 5) test our model against the testing data

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

mnist_dataset, mnist_info = tfds.load(name='mnist', as_supervised = True, with_info = True)

##Split the data set into test and training
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

num_validation_samples = 0.1*mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64) ##Ensure it's a whole number

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)
##used when you can't shuffle the entire data set
BUFFER_SIZE = 10000

shuffle_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
##extract the validation data
validation_data = shuffle_train_and_validation_data.take(num_validation_samples).cache().repeat()

##extract the training data
train_data = shuffle_train_and_validation_data.skip(num_validation_samples)

BATCH_SIZE= 100
##Set a column to tell the program the batch_size
train_data=train_data.batch(BATCH_SIZE)
##Set the batch size of the validation data to the whole set, no need to batch since we are not updating weights
validation_data = validation_data.batch(num_validation_samples)

##set the batch size of the test data to the whole test
test_data= test_data.batch(num_test_samples)

##Mnist data is in tuple format, so we extract the targets and the inputs
validation_inputs, validation_targets = next(iter(validation_data))

##outline the model
input_size = 784
output_size = 10
hidden_layer_size = 100
##Build you model
model = tf.keras.Sequential([
                                tf.keras.layers.Flatten(input_shape=(28,28,1)), ##Flattern the input Matrix into a single vector
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'), ##Take the inputs and apply the dot product of the weights and add biases, we can also add non-linearity to it here
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                tf.keras.layers.Dense(output_size, activation='softmax') ##Specify the output and apply a softmax solution
                            ])
##outline the model for optimization and minimize the loss function
model.compile(optimizer = 'Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

##Train our model
num_epochs = 5
model.fit(train_data, epochs = num_epochs, validation_data=(validation_inputs,validation_targets), verbose=2, validation_steps=10)
##test our model
test_loss, test_accuracy = model.evaluate(test_data)

print(test_loss, test_accuracy)
