#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
from numpy import linalg as LA
import sympy
import matplotlib.pyplot as plt
import statistics


# # LAB#4- Simple Forecasting Methods

# ## 10- Create a .py file and name it as “test-drift-method.py”. With the same test dataset y, repeat step 1 through 8 for the Naïve method. The drift estimate is simply as follows. Where T is number of samples and h is the number of steps ahead prediction. In this case since we are ding one-step ahead prediction, h =1.

# ### 1- Generate a test sample as 𝑦 = [1.5,2.1,3.9,4.4,5.2] The above vector is the true values. Using the naive method perform one-step ahead prediction. Where T is number of samples and h is the number of steps ahead prediction. In this case since we are ding one-step ahead prediction, h =1.

# In[33]:


import sys
y= np.array([1.5,2.1,3.9,4.4,5.2])
T =5
h=1
Ytrue=y
Yhat=[]
for i in range(3,6):
        Yhat.append((y[i-2]+(h*((y[i-2]-y[0])/(i-2)))))
    
Yhat


# ### 2- Plot the True values versus Predicted values in one graph with different marker. Add an appropriate title, legend, x-label, y-label to your plot.

# In[34]:


plt.rcParams["figure.figsize"] = (30,15)
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize']=40
plt.rcParams['ytick.labelsize']=40
plt.rcParams['legend.fontsize']=40


# In[35]:


obs=np.array([1,2,3,4,5])
plt.plot(obs[2:],Yhat,label='Predicted value',marker="*",markersize=20,color="purple")
plt.plot(obs,Ytrue,label='True value',marker="o",markersize=20,color="green")
plt.title("Drift Forecasting Method",pad=20)
plt.ylabel("Value")
plt.xlabel("Observation #")
plt.legend()
ax = plt.gca()
ax.grid(which='major', axis='y', linestyle='-')
plt.show()

# ### 3- Display the true value and the predicted value as:
# #### The predicted values are: ___________________
# #### The true values are: ______________________

# In[36]:


print('The predicted values are:',Yhat)
print('The true values are:',Ytrue[2:])


# ### 4- Calculate the residuals which is the difference between predicted values and true values and display it as:
# #### The residuals for this estimate are: ______________

# In[37]:


res=Ytrue[2:]-Yhat
print('The residuals for this estimate are: ',res)

print('\nThe mean of the residuals for this estimate is: ',statistics.mean(res))
print('\nThe variance of the residuals for this estimate is: ',statistics.variance(res))

plt.hist(res)
plt.title('Residuals Distribution',pad=40)
plt.xlabel('Residual value')
plt.ylabel('Number of Observations')
plt.figure()
plt.show()

# ### 5- Calculate the sum square of the residuals. This is norm is called “Sum Square Error” or simply “SSE”. Display the message as follow:
# #### The sum square error for this estimate is: _________

# In[38]:


sse=0
SSE=[]
for j in res:
    sse=sse+(j**2)
    SSE.append(sse)

print('The sum square error for this estimate is: ',SSE[-1])


# ### 6- Plot and display the ACF of the residuals (lags = 3). Add the Add an appropriate title, legend, xlabel, y-label to your plot. The display message should be like the following:
# #### The ACF values of the residuals for this estimate are: ___¶

# In[39]:


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


# In[40]:


print('The ACF values of the residuals for this estimate are: ',autocorrelation(res))


# In[41]:


x2=list()
for i in range(0,3):
    x2.append(i)
x1=[-1,-2]
x=x1[::-1]+x2
x


# In[42]:


autocorr=autocorrelation(res)
autocorr1=autocorr[::-1]
autocorr=autocorr[1:]
autocorr=autocorr1+autocorr
autocorr


# In[43]:


plt.stem(x,autocorr, use_line_collection=True)
plt.title('Autocorrelation function for the residuals of the Drift Forecasting Method',pad=20)
plt.ylabel('Autocorrelation value')
plt.xlabel('Lag')
plt.figure()
plt.show()

# ### 7- Calculate the Q value for this estimate and display the following message:
# #### The Q value for this estimate is = ____________

# In[44]:


T=5
# As k=1 and h=1 then
r=autocorrelation(res)
#r=r[1]
r2=[]
for i in r:
    r2.append(i**2)
#print(r)
#print(r2)
Q=[]

for i in range(1,len(r2)+1):
    r22=r2[:i]
    T=len(r22)
    Q.append(T*sum(r22))
print('\nThe Q value for this estimate is = ',Q[-1])


# ### 8- Create a table and display the true values and estimated values at different time steps, Q value and SSE.
# 

# In[45]:


table=pd.DataFrame()
table['Ypredicted']=['y3','y4','y5']
table['True value']=Ytrue[2:]
table['Estimated value']=Yhat
table['SSE']=[np.nan,np.nan,SSE[-1]]
table['Q']=[np.nan,np.nan,Q[-1]]
print(table)


# In[47]:


GeneralTable=pd.DataFrame()
GeneralTable['Forecasting Method']=['Average','Naive','Drift','True value']
GeneralTable['Y3|2']=[1.8,2.1,2.7,3.9]
GeneralTable['Y4|3']=[2.5,3.9,5.1,4.4]
GeneralTable['Y5|4']=[2.975,4.4,5.367,5.2]
GeneralTable['SSE']=[12.97,4.13,1.958,np.nan]
GeneralTable['Res mean']=[2.075,1.033, 0.111,np.nan]
GeneralTable['Res variance']=[0.027,0.463,0.96,np.nan]
GeneralTable['Q']=[3.988507,3.394463,3.426,np.nan]

print(GeneralTable)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




