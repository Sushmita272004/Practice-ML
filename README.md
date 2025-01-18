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
print(var3)

s={'s':pd.Series([1,2,3,4]),'r':pd.Series([1,2,3,4])}
var4=pd.DataFrame(s)
print(var4)
print("The required no. is : ")
print(var4['s'][2])
