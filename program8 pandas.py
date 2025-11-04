import numpy as np
arr1=np.array([1,2,3,4,5])
print("1d array: ",arr1)
arr2=np.array([[1,2,3],[4,5,6]])
print("2d array: ",arr2)
print("shape of arr2; ",arr2.shape)
print("data type of arr2: ",arr2.dtype)
arr_sum=arr1+5;
print("after adding 5 to the array1 : ",arr_sum)
arr_mul=arr1*2
print("after multiplying 2 with the array1 : ",arr_mul)

arr3=np.array([10,20,30,40,50])
arr_add=arr1+arr3
print("element - wise addition : ",arr_add)


