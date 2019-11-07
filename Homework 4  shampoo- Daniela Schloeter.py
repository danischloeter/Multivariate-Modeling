#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy import linalg as LA
import sympy
import matplotlib.pyplot as plt
import statistics


# # Homework #4- Simple Forecasting Methods

# In this homework, you will use the codes developed in lab#4 for 4 different dataset and compare the performance of 3 simple forecasting methods:
# • Average Method
# • Naïve Method
# • Drift Method
# Using the Python program and the appropriate libraries perform the following tasks:
# # 7- Repeat set 1, 2, 3, 4, 5, 6 on “Shampoo.csv” dataset. You will be predicting the Sales for this dataset.
# 
# ## 1- Load the Sales values of dataset “Shampoo.csv”. The purpose is to perform one-step ahead prediction on Sales value.

# In[2]:


Shampoo=pd.read_csv('Shampoo.csv')
Shampoo.head()
Shampoo.info()


# ## 2- Edit the code “test_mean_method.py” (from LAB#4) to perform one-step ahead prediction for Sales value using mean method.

# In[3]:


Yhatmean=[]
for i in range(3,37):
    yhatmean=Shampoo.iloc[:i-1,1]
    yhatmean=yhatmean.sum()/len(yhatmean)
    Yhatmean.append(yhatmean)
len(Yhatmean)


# ## 3- Edit the code “test_naive_method.py” (from LAB#4) to perform one-step ahead prediction prediction for Sales value using Naive method.

# In[4]:


Yhatnaive=[]
for i in range(3,37):
    Yhatnaive.append(Shampoo.iloc[i-2,1])
len(Yhatnaive)


# ## 4- Edit the code “test_drift_method.py” (from LAB#4) to perform one-step ahead prediction for Sales value using drift method.

# In[5]:


import sys
h=1
Yhatdrift=[]
for i in range(3,37): 
    Yhatdrift.append((Shampoo.iloc[i-2,1]+(h*((Shampoo.iloc[i-2,1]-Shampoo.iloc[0,1])/(i-2)))))
    
len(Yhatdrift)


# ## 5- For the step 2, 3 and 4:
# 
#    ### a. Plot the True values versus Predicted values in one graph with different marker. Add an appropriate title, legend, x-label, y-label to your plot.
# 

# In[6]:


obspred=range(3,37)
obstrue=range(1,37)


# In[7]:


plt.rcParams["figure.figsize"] = (40,25)
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize']=40
plt.rcParams['ytick.labelsize']=40
plt.rcParams['legend.fontsize']=40
plt.show()

# #### Mean method

# In[8]:


plt.plot(obspred,Yhatmean,label='Predicted values for mean method',marker="*",markersize=15,color="purple")
plt.plot(obstrue,Shampoo.iloc[:,1],label='True values',marker="o",markersize=15,color="green")
plt.title("Average Forecasting Method",pad=40)
plt.ylabel("Value")
plt.xlabel("Observation #")
plt.legend()
ax = plt.gca()
ax.grid(which='major', axis='y', linestyle='-')

plt.show()
# #### Naive method

# In[9]:


plt.plot(obspred,Yhatnaive,label='Predicted values naive method',marker="*",markersize=40,color="purple")
plt.plot(obstrue,Shampoo.iloc[:,1],label='True value',marker="o",markersize=20,color="green")
plt.title("Naive Forecasting Method",pad=40)
plt.ylabel("Value")
plt.xlabel("Observation #")
plt.legend()
ax = plt.gca()
ax.grid(which='major', axis='y', linestyle='-')
plt.show()

# #### Drift method

# In[10]:


plt.plot(obspred,Yhatdrift,label='Predicted value drift method',marker="*",markersize=20,color="purple")
plt.plot(obstrue,Shampoo.iloc[:,1],label='True value',marker="o",markersize=20,color="green")
plt.title("Drift Forecasting Method",pad=40)
plt.ylabel("Value")
plt.xlabel("Observation #")
plt.legend()
ax = plt.gca()
ax.grid(which='major', axis='y', linestyle='-')
plt.show()

# ### b. Calculate the residuals and display the SSE

# #### Mean method

# In[11]:


resmean=Shampoo.iloc[2:,1]-Yhatmean
print('The residuals for this estimate are: ',resmean)
print('\nThe mean of the residuals for this estimate is: ',statistics.mean(resmean))
print('\nThe variance of the residuals for this estimate is: ',statistics.variance(resmean))
sse=0
SSEmean=[]
for j in resmean:
    sse=sse+(j**2)
    SSEmean.append(sse)

print('\nThe sum square error for this estimate is: ',SSEmean[-1])
plt.hist(resmean)
plt.title('Residuals Distribution for the Mean Forecasting Method',pad=40)
plt.xlabel('Residual value')
plt.ylabel('Number of Observations')
plt.figure()
plt.show()

# #### Naive method

# In[12]:


resnaive=Shampoo.iloc[2:,1]-Yhatnaive
print('The residuals for this estimate are: ',resnaive)
print('\nThe mean of the residuals for this estimate is: ',statistics.mean(resnaive))
print('\nThe variance of the residuals for this estimate is: ',statistics.variance(resnaive))

sse=0
SSEnaive=[]
for j in resnaive:
    sse=sse+(j**2)
    SSEnaive.append(sse)

print('\nThe sum square error for this estimate is: ',SSEnaive[-1])
plt.hist(resnaive)
plt.title('Residuals Distribution for the Naive Forecasting Method',pad=40)
plt.xlabel('Residual value')
plt.ylabel('Number of Observations')
plt.figure()
plt.show()

# #### Drift method

# In[13]:


resdrift=Shampoo.iloc[2:,1]-Yhatdrift
print('The residuals for this estimate are: ',resdrift)
print('\nThe mean of the residuals for this estimate is: ',statistics.mean(resdrift))
print('\nThe variance of the residuals for this estimate is: ',statistics.variance(resdrift))
sse=0
SSEdrift=[]
for j in resdrift:
    sse=sse+(j**2)
    SSEdrift.append(sse)

print('\nThe sum square error for this estimate is: ',SSEdrift[-1])
plt.hist(resdrift)
plt.title('Residuals Distribution for the Drift Forecasting Method',pad=40)
plt.xlabel('Residual value')
plt.ylabel('Number of Observations')
plt.figure()
plt.show()

# ### c. Plot and display the ACF of the residuals (lags = 20). Add an appropriate title, legend, x- label, y-label to your plot.

# In[14]:


def autocorrelation(y):
    '''It returns the autocorrelation'''
    t=len(y)
    k=range(0,t)
    meany=np.mean(y)
    tk=[]
    for i in k:
        numerator=0
        denominator=0
        for ti in range(i,t):
            numerator+=((y[ti]-meany)*(y[ti-i]-meany))
        denominator=np.sum((y-meany)**2)
        tk.append(numerator/denominator)
    return tk


# In[15]:


x2=list()
for i in range(0,21):
    x2.append(i)
x1=[-1,-2,-3,-4,-5,-6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20]
x=x1[::-1]+x2


# #### Mean method

# In[16]:


resmean=np.array(resmean)
#print('The ACF values of the residuals for this estimate are: \n',autocorrelation(resmean))


# In[17]:


autocorrmean=autocorrelation(resmean)
autocorrmean=autocorrmean[:21]
autocorrmean1=autocorrmean[::-1]
autocorrmean=autocorrmean[1:]
autocorrmean=autocorrmean1+autocorrmean


# In[18]:


plt.stem(x,autocorrmean, use_line_collection=True)
plt.title('Autocorrelation function for the residuals of the Mean Forecasting Method',pad=40)
plt.ylabel('Autocorrelation value')
plt.xlabel('Lag')
plt.figure()
plt.show()

# #### Naive method

# In[19]:


resnaive=np.array(resnaive)
#print('The ACF values of the residuals for this estimate are: ',autocorrelation(resnaive))


# In[20]:


autocorrnaive=autocorrelation(resnaive)
autocorrnaive=autocorrnaive[:21]
autocorrnaive1=autocorrnaive[::-1]
autocorrnaive=autocorrnaive[1:]
autocorrnaive=autocorrnaive1+autocorrnaive


# In[21]:


plt.stem(x,autocorrnaive, use_line_collection=True)
plt.title('Autocorrelation function for the residuals of the Naive Forecasting Method',pad=40)
plt.ylabel('Autocorrelation value')
plt.xlabel('Lag')
plt.figure()
plt.show()

# #### Drift method

# In[22]:


resdrift=np.array(resdrift)
#print('The ACF values of the residuals for this estimate are: ',autocorrelation(resdrift))


# In[23]:


autocorrdrift=autocorrelation(resdrift)
autocorrdrift=autocorrdrift[:21]
autocorrdrift1=autocorrdrift[::-1]
autocorrdrift=autocorrdrift[1:]
autocorrdrift=autocorrdrift1+autocorrdrift


# In[24]:


plt.stem(x,autocorrdrift, use_line_collection=True)
plt.title('Autocorrelation function for the residuals of the Drift Forecasting Method',pad=40)
plt.ylabel('Autocorrelation value')
plt.xlabel('Lag')
plt.figure()
plt.show()

# ### d. Calculate the Q value for this estimate and display the Q value.

# #### Mean method

# In[25]:


T=34
# As k=1 and h=1 then
r=resmean
r=autocorrelation(r)
#r=r[1]
r2=[]
for i in r:
    r2.append(i**2)

Qmean=[]

for i in range(1,len(r2)+1):
    r22=r2[:i]
    Qmean.append(T*sum(r22))
print('The Q value for this estimate is = ',Qmean[-1])


# #### Naive method

# In[26]:


T=34
# As k=1 and h=1 then
r=resnaive
r=autocorrelation(r)
#r=r[1]
r2=[]
for i in r:
    r2.append(i**2)
Qnaive=[]

for i in range(1,len(r2)+1):
    r22=r2[:i]
    Qnaive.append(T*sum(r22))
print('The Q value for this estimate is = ',Qnaive[-1])


# #### Drift method

# In[27]:


T=34
# As k=1 and h=1 then
r=autocorrelation(resdrift)
#r=r[1]
r2=[]
for i in r:
    r2.append(i**2)
Qdrift=[]

for i in range(1,len(r2)+1):
    r22=r2[:i]
    Qdrift.append(T*sum(r22))
print('The Q value for this estimate is = ',Qdrift[-1])


# ## 6- Based on your observation and by looking at Q values, ACF, SSE, mean, variance and distribution for the residuals pick the best estimator for this dataset. You need to justify your answer why the picked estimator is better than the others

# In[28]:


GeneralTable=pd.DataFrame()
GeneralTable['Forecasting Method']=['Average','Naive','Drift']
GeneralTable['SSE']=[SSEmean[-1],SSEnaive[-1],SSEdrift[-1]]
GeneralTable['Res mean']=[statistics.mean(resmean),statistics.mean(resnaive), statistics.mean(resdrift)]
GeneralTable['Res variance']=[statistics.variance(resmean),statistics.variance(resnaive),statistics.variance(resdrift)]
GeneralTable['Q']=[Qmean[-1],Qnaive[-1],Qdrift[-1]]

print(GeneralTable)


# 

# In[ ]:





# In[ ]:





# In[ ]:




