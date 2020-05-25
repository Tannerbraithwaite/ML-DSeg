#Source for Matrix examples
import numpy as np


s=5#simple scalars
v=np.array([4,-2,4])#Use arrays for vectors(default is a row vector)
m=np.array([[5,12,6],[-3,0,14]])
v.reshape(1,3)


##Tensors
m2=np.array([[9,8,7],[1,3,-5]])
t=np.array([m,m2])

#addition of Tensors(condition, the two must have the same dimensions(adds corresponding values of each matrix))
print(m+m2)

##Transposing Matricews
print(m.T)

#Dot Product - scalar product or multiplication
x=np.array([2,8,-4])
y=np.array([1,-7,3])
dotprod=np.dot(x,y)
scalarmult= x*5#can take the product of a scalar and a vector(keeps the same shape but scales values)

##for Matrices, we can only multiply an mxn matric with an nxk matrix(second dimension of the first must match the first dimension of the second)
dotproductmatrix=np.dot(m,m2.reshape(3,2))
print(dotproductmatrix)
#useful in data science (inputs matrix)*(cefficients matrix) = (outputs matrix) Array computations are significantly faster
