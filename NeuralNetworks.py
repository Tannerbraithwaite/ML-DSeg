import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import pandas as pd

observations = 1000
##two variable method F(x,x2) = ax+bx2+C
##create the inputs of two randomly generated sets and combine them using column stack
xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10,10,(observations,1))

inputs = np.column_stack((xs,zs))

##Create Targets
##targets have noise to model real data
noise = np.random.uniform(-1,1,(observations,1))

##create the targets
targets = 2*xs -3*zs + 5 + noise

#Plot the training data
# targets = targets.reshape(observations,)
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.plot(xs,zs,targets)
# ax.set_xlabel('xs')
# ax.set_ylabel('zs')
# ax.set_zlabel('Targets')
# ax.view_init(azim=100)
# # plt.show()
# targets=targets.reshape(observations,1)

#initialize weights
init_range = 0.1
weights = np.random.uniform(-init_range, init_range, size=(2,1))
biases = np.random.uniform(-init_range, init_range, size=1)
learning_rate = 0.02

for i in range(100):
    outputs = np.dot(inputs, weights) + biases ##calculates the outs for comparison
    deltas = outputs - targets ##determines the loss difference between the outputs and the targets
    loss=np.sum(deltas**2)/ 2 / observations ##Determines to loss of our function(we attempt to minimize this)
    # print(loss)
    deltas_scaled=deltas/observations ##Scale the deltas for each operation
    weights = weights-learning_rate*np.dot(inputs.T,deltas_scaled) #Adjust the weights according to the opimization formula
    biases = biases - learning_rate*np.sum(deltas_scaled)##adjust the biases according to the optimization formula

# print(weights)
# print(biases)
# plt.plot(outputs,targets)
# plt.xlabel('outputs')
# plt.ylabel('targets')
# # plt.show()
