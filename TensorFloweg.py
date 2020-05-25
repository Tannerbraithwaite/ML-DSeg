##Tensorflow example
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

observations = 1000
##two variable method F(x,x2) = ax+bx2+C
##create the inputs of two randomly generated sets and combine them using column stack
xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10,10,(observations,1))

generated_inputs = np.column_stack((xs,zs))
##targets have noise to model real data
noise = np.random.uniform(-1,1,(observations,1))

##create the targets
generated_targets = 2*xs -3*zs + 5 + noise
np.savez('TF_intro', inputs = generated_inputs, targets=generated_targets) ##Saves it into a NPZ file

training_data = np.load('TF_intro.npz')
input_size = 2
output_size=1
model=tf.keras.Sequential([
                            tf.keras.layers.Dense(output_size,
                            kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                            bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
                            )
                            ])

custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)


##Extract weights and biases
print(model.layers[0].get_weights())
print(model.predict_on_batch(training_data['inputs']))
print(training_data['targets'].round(1))

##Plot the data to show a 45 degree line
plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()
