# Practice-ML

# 1
#PANDAS
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
#NUMPY
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
#no of rows and columns 
print(np.size(l3))
#no. of elements in total
#ndim= no. of dimensions #rows and columns
print(np.ndim(l3))
print(l3.dtype)
print(l3.astype(float))
print(np.vstack(l3))
print(np.hstack(l3))
print(np.array_split(l3,2))

import matplotlib.pyplot as plt

x=['python','c','c++','java']
y=[59,50,35,39]
z=[10,20,30,40]

#MATPLOTLIB
plt.xlabel('Language',fontsize=15)
plt.ylabel('Numbers',fontsize=15)
plt.title('Assignment',fontsize=15)
#for multi colors : col=['m','g','y','r']
plt.bar(x,y,width=0.4,color='g',align='center',edgecolor='b',linewidth=2,linestyle='dotted',alpha=0.6,label='hacknitr')
plt.legend()
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x=['python','c','c++','java']
y=[59,50,35,39]
z=[10,20,30,40]
width=0.3
s=np.arange(len(x))
s1=[j+width for j in s]  #doubt
print(s)
print(s1)


plt.xlabel('Language',fontsize=15)
plt.ylabel('Numbers',fontsize=15)
plt.title('Assignment',fontsize=15)
plt.bar(s,y,width,color='g',label='hacknitr',alpha=0.5)
plt.bar(s1,z,width,color='y',label='hacknitr1',alpha=0.7)
plt.xticks(s+width/2,x,rotation=20)
plt.legend()
plt.show()

import matplotlib.pyplot as plt
x=['python','c','c++','java']
y=[59,50,35,39]
z=[10,20,30,40]

plt.xlabel('Language',fontsize=15)
plt.ylabel('Numbers',fontsize=15)
plt.title('Assignment',fontsize=15)
plt.barh(x,y,color='g',label='hacknitr')
plt.barh(x,z,color='y',label='hacknitr1')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
x=[1,2,3,4,7,8,4,3,2]
y=[3,6,7,7,4,2,8,4,9]
plt.xlabel('data',fontsize=15)
plt.ylabel('data nos',fontsize=15)
plt.title("scatter plot",fontsize=15)
size=[9,9,9,9,90,90,90,9,98]
plt.scatter(x,y,color='g',edgecolor='b',s=size,linewidth=1.4,)
plt.show()

# 2
import sklearn
from sklearn.datasets import load_iris
load_iris()

#python(bfs/dfs)
from collections import deque

def bfs(graph, start, target):
    visited = set()
    queue = deque([(start, [start])])
    
    while queue:
        vertex, path = queue.popleft()
        
        if vertex == target:
            return path
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return "No path found"

def dfs(graph, start, target, path=None, visited=None):
    if path is None:
        path = [start]
    if visited is None:
        visited = set()
    
    visited.add(start)
    
    if start == target:
        return path
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            result = dfs(graph, neighbor, target, path + [neighbor], visited)
            if result:
                return result
    
    return "No path found"

#Taking input from the user
graph = {}
num_nodes = int(input("Enter the number of nodes: "))
for _ in range(num_nodes):
    node = input("Enter the node: ")
    neighbors = input(f"Enter the neighbors of {node} separated by space: ").split()
    graph[node] = neighbors

start_node = input("Enter the start node: ")
target_node = input("Enter the target node: ")

#Finding paths using BFS and DFS
bfs_path = bfs(graph, start_node, target_node)
dfs_path = dfs(graph, start_node, target_node)

#Displaying the results
print("BFS path (shortest path):", bfs_path)
print("DFS path (any valid path):", dfs_path)

import pandas as pd
d={'Name':['Alice','Bob','Charlie','David'],'Age':[24,30,29,35],'City':['New York','Chicago','San Francisco','New York']}
df=pd.DataFrame(d,index=[1,2,3,4])
s=df[df['Age']>28]
print("the data is :")
print(df)
print()
print("the data with age greater than 28 is :")
print(s)
# 3
import pandas as pd
import numpy as np

np.random.seed(0)

data_size = 500
data = {
    'ID': range(1, data_size + 1),
    'TV': np.random.rand(data_size) * 100,  
    'Radio': np.random.rand(data_size) * 100,  
    'Newspaper': np.random.rand(data_size) * 100, 
    'Sales': np.random.rand(data_size) * 50 + 10 
}
sales_data = pd.DataFrame(data)

print(sales_data.head())

from sklearn.model_selection import train_test_split

X = sales_data[['TV', 'Radio', 'Newspaper']]
y = sales_data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


print("Training set (X):")
print(X_train.head())
print("\nTesting set (X):")
print(X_test.head())

print("\nTraining set (y):")
print(y_train.head())
print("\nTesting set (y):")
print(y_test.head())

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Coefficients:")
print(model.coef_)
print("\nIntercept:")
print(model.intercept_)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2:.2f}")

input_numbers = input("Enter numbers separated by spaces: ").split()
numbers = []
for number in input_numbers:
    numbers.append(int(number))
l = []
for number in numbers:
    if number > 500:
        break
    if number > 150:
        continue
    if number % 5 == 0:
        l.append(number)

print(l)

#ques 5 count total digits in a no. (while loop)

number = int(input("Enter a number: "))
count = 0
while number != 0:
    number //= 10  
    count += 1   
print("Total number of digits:", count)



