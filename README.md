# Practice-ML
import pandas as pd
dict={'name':['sushmita','saloni','aliva','avisya'],'age':[23,21,20,22]}
var=pd.Series(dict)
print(var)

lst=[1,2,3,4,5]
s=pd.Series(lst)
print(s)
print(type(s))

s1=pd.Series(12,index=[1,2,3,4,5,6])
s2=pd.Series(12,index=[1,2,3,5])
print(s1+s2)

lst=[3,45,34,32]
var=pd.DataFrame(lst)
print(var)
print(type(var))

dict={'a':[11,12,13,14,15],'b':['x','y','z','a','b']}
var1=pd.DataFrame(dict, index=[1,2,3,4,5])
print(var1)
var2=pd.DataFrame(dict)
print(var2['a'][3])
lst=[[1,2,3,4],[7,8,9,10]]
var3=pd.DataFrame(lst)

import numpy as np
l1=[10,11,12]
l2=[10,50,60]
print(l1+l2)

a=np.array([10,20,30])
b=np.array([20,40,50])
print(a)
print(b)
print(a*b)
print(a+b)
print(var3)

s={'s':pd.Series([1,2,3,4]),'r':pd.Series([1,2,3,4])}
var4=pd.DataFrame(s)
print(var4)
print("The required no. is : ")
print(var4['s'][2])

#correct in case in list
l1=([[10,20],[30,40,50]]) 
print(l1)
print(type(l1))
#incorrect in case of array
l2=np.array([[10,20],[30,40,50]])
print(l2)

import numpy as np
l3=np.array([[10,20,30,40],[40,50,60,70],[20,60,40,80],[45,67,89,76]])
print(l3[0:4])
print(l3[2,0:4])
print(l3[0:4,0:2])
print(np.shape(l3))
# no of rows and columns 
print(np.size(l3))
#no. of elements in total
#ndim= no. of dimensions #rows and columns
print(np.ndim(l3))
print(l3.dtype)
print(l3.astype(float))
print(np.vstack(l3))
print(np.hstack(l3))
print(np.array_split(l3,2))
