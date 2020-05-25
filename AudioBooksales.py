##AudioBooksales.py
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
raw_csv_data = np.loadtxt('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_7_Deep_Learning/S51_L356/Audiobooks_data.csv', delimiter = ',')

unscaled_inputs_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]

num_one_targets = int(np.sum(targets_all))
zero_targets_counters = 0
indices_to_remove = []


for i in range(targets_all.shape[0]):
    if targets_all[i]==0:
        zero_targets_counters+=1
        if zero_targets_counters>num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove,axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8*samples_count)
validation_samples_count = int(.1*samples_count)
test_samples_count = samples_count - train_samples_count-validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)

print(np.sum(train_targets), train_samples_count,np.sum(train_targets)/train_samples_count)
print(np.sum(validation_targets), validation_samples_count,np.sum(validation_targets)/validation_samples_count)
print(np.sum(test_targets), test_samples_count,np.sum(test_targets)/test_samples_count)

##Typically done in an extra file
npz = np.load('Audiobooks_data_train.npz')
train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_validation.npz')
validation_inputs = npz['inputs'].astype(np.float)
validation_targets = npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_test.npz')
test_inputs = npz['inputs'].astype(np.float)
test_targets = npz['targets'].astype(np.int)

input_size = 10
output_size = 2
hidden_layer_size = 100
##Build you model
model = tf.keras.Sequential([
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'), ##Take the inputs and apply the dot product of the weights and add biases, we can also add non-linearity to it here
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                tf.keras.layers.Dense(output_size, activation='softmax') ##Specify the output and apply a softmax solution
                            ])

model.compile(optimizer = 'Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)

##Train our model
batch_size = 100
max_epochs = 100
model.fit(train_inputs, train_targets, batch_size=batch_size, epochs=max_epochs, validation_data =(validation_inputs, validation_targets), verbose = 2, callbacks = [early_stopping])
##test our model
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
