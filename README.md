## Numpy-Notes
Creating Arrays:
•	np.array(): Create a NumPy array from a Python list or tuple.
•	np.zeros(): Create an array filled with zeros.
•	np.ones(): Create an array filled with ones.
•	np.empty(): Create an empty array.
•	np.arange(): Create an array with a range of values.
•	np.linspace(): Create an array with evenly spaced values.

## Creating array with .array()
import numpy as np
a = np.array([1,2,3,4,5])
b =np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(b)
print(a)
            


## Creating with np.zeros
a = np.zeros((2,2))
print(a)

a = np.ones((5,5))
print(a)

## Creating Array with arange
## Always Create 1D but you can reshape it latter

a = np.arange(1,6)
print(a)

import numpy as np

a = np.linspace(1, 10, 5)  # Create a one-dimensional array with 5 values from 1 to 10

print(a)



Array Manipulation:
•	np.shape(): Get the shape (dimensions) of an array.
•	np.reshape(): Reshape an array.
•	np.transpose(): Transpose an array.
•	np.concatenate(): Join arrays along an axis.
•	np.split(): Split an array into smaller arrays.
•	np.vstack() and np.hstack(): Stack arrays vertically and horizontally.
•	arr = np.arange(1,11)
•	brr = arr.reshape(2,5)
•	print(arr)
•	print(brr)
•	arr.shape
•	brr.shape

## Transpose 
new = brr.transpose()
print(new)

## concatenates arrays b and a together, resulting in a single arra

a= np.arange(1,5)
b= np.arange(6,10)
Result = np.concatenate((b,a))
print(Result)
print(a)
print(b)

##np.concatenate() combines arrays along an existing axis.
## np.stack() stacks arrays along a new axis, creating a higher-dimensional array.
Result = np.stack((a,b))
print(Result)

Result = np.dstack((a,b)) ## H staxck for 3d array 
ustack =np.vstack((a,b))  ## Align Vertically Along with Rows
hsatck =np.hstack((a,b))  ## Along with Column 

print(Result ,"\n")
print(ustack ,"\n")
print(hsatck)


arr = np.array([1, 2, 3, 4, 5, 6])
result = np.split(arr, 3) ## it will devide into to sub array
for i in result:
    print (i)

print(result)


Mathematical Operations:
•	np.add(), np.subtract(), np.multiply(), np.divide(): Element-wise arithmetic operations.
•	np.dot(): Matrix multiplication.
•	np.sum(), np.mean(), np.median(), np.std(): Statistical operations.
•	np.min(), np.max(): Find minimum and maximum values.
•	np.exp(), np.log(), np.sqrt(): Exponential, logarithmic, and square root functions
•	a = np.arange(71,76)
•	b= np.arange(6,11)
•	# zerto =np.zeros((5,))
•	
•	Result = np.add(a,b)
•	Result2 = np.subtract(b,a)
•	print(Result,'\n')
•	print(Result2,'\n')

Result3 = np.multiply(a,b)
Result4 = np.divide(a,b)
print(Result3,'\n')
print(Result4)

## In order to perform matrix multiplication, the number of columns in the first matrix 
## should be equal to the number of rows in the second matrix.
aa = np.array([[1, 2, 3], [4, 5, 6]])
bb = np.array([[8, 9], [10, 11], [12, 13]])

Result5 = np.dot(aa, bb)

print(Result5)

## Mean Medium
## For Min Max Value

new = np.mean(a)
new2 = np.median(a)
new3 = np.min(a)
new4 = np.max(a)
new5 = np.sqrt(b)
print(new5)
print(new4)
print(new2)
print(new)
print(new3)


Linear Algebra:
•	np.linalg.inv(): Compute the inverse of a matrix.
•	np.linalg.det(): Compute the determinant of a matrix.
•	np.linalg.eig(): Compute eigenvalues and eigenvectors.
•	np.linalg.solve(): Solve linear equations.



## Random Number Generation:

#### np.random.rand(): Generate random numbers from a uniform distribution.
#### np.random.randn(): Generate random numbers from a standard normal distribution.
#### np.random.randint(): Generate random integers.
#### np.random.choice(): Randomly choose elements from an array.## 


File I/O:
•	np.save(): Save an array to a binary file.
•	np.load(): Load an array from a binary file.
•	np.savetxt(): Save an array to a text file.
•	np.loadtxt(): Load an array from a text file
array Aggregation:
•	np.unique(): Find unique elements in an array.
•	np.sort(): Sort elements in an array.
•	np.argsort(): Get the indices that would sort an array.
•	np.bincount(): Count occurrences of each value in an array of non-negative integers.





