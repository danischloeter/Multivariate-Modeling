
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Homework1

# ## 1. Load the time series data called AirPassengers.csv

# In[3]:


air=pd.read_csv('AirPassengers.csv')


# # 2. This date relates to the air passengers between 1949-1960

# In[4]:


air.info()


# In[5]:


air['Month']=pd.to_datetime(air['Month'], format='%Y/%m')


# In[6]:


air.info()


# ## 3. Write a Python code to display the first 5 rows of the data

# In[7]:


air.head()


# ## 4. Plot the entire dataset, where you can learn more about the data set pattern ( trend, seasonality, cyclic, ...). Add the label to the horizontal and vertical axis as Month and Sales Number. Add the title as “Shampoo Sales Dataset without differencing”. Add an appropriate legend to your plot. Do you see any trend, seasonality or cyclical behavior in the plotted dataset? If yes, what is it?

# In[8]:


plt.rcParams["figure.figsize"] = (30,15)
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize']=24
plt.rcParams['ytick.labelsize']=24
plt.rcParams['legend.fontsize']=24


# In[9]:


plt.plot(air['Month'],air['#Passengers'],marker="o",markersize=10,color="orange",linestyle="solid",linewidth=2)
plt.title("Airline number of passengers by month from 1949 to 1960 without differencing",pad=40)
plt.ylabel("Number of passengers")
plt.xlabel("Month")
plt.grid()


# ## 5. Is this a non-stationary dataset? Justify your answer. Calculate the average over the entire dataset and show the average plot.

# Yes, it is a non-stationary dataset because it has a trend and therefore the statistics ... This can be better appreciated by looking at the average and variance of each subset of our dataset.

# In[10]:


means=[]
variances=[]
a=[]
for i in range(0,144):
    if i==0:
        a=air.iloc[i,1]
        means.append(a)
        variances.append(0)
    else:
        a=air.iloc[:i+1,1]
        means.append(a.mean())
        variances.append(a.var())
salesmeans=means
salesvariances=variances


# In[11]:


x=range(0,144)
plt.plot(x,salesmeans,marker="o",markersize=10,color="orange",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title('Mean number of passengers by subset' ,pad=40)
plt.ylabel("Mean number of passengers")
plt.xlabel("Subset")
plt.grid()


# In[12]:


x=range(0,144)
plt.plot(x,salesvariances,marker="o",markersize=10,color="orange",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title("Variance for number of passengers by subset", pad=40)
plt.ylabel("Variance for number of passengers")
plt.xlabel("Subset")
plt.grid()


# ## 6. If the answer to the previous question is yes, write a python code that detrend the dataset by 1st difference transformation. Plot the detrended dataset.

# First order differentiation ∆y(t) = y(t) − y(t − 1)

# In[13]:


differentiation=[]
b=[]
for i in range(0, 144):
    if i==0:
        b=0
        differentiation.append(b)
    else:
        b= air.iloc[i,1] - air.iloc[i - 1,1]
        differentiation.append(b)
#differentiation


# In[14]:


air['differentiation']=differentiation
#air


# In[15]:


plt.plot(air['Month'],air['differentiation'],marker="o",markersize=10,color="purple",linestyle="solid",linewidth=2)
plt.title("Airline number of passengers by month from 1949 to 1960 detrended with 1st differentiation",pad=40)
plt.ylabel("Differentiation")
plt.xlabel("Month")
plt.grid()


# ## 7. Is the detrended dataset stationary? Justify your answer. Calculate the average over the entire dataset and show the average plot.

# In[16]:


means=[]
variances=[]
a=[]
for i in range(0,144):
    if i==0:
        a=air.iloc[i,2]
        means.append(a)
        variances.append(0)
    else:
        a=air.iloc[:i+1,2]
        means.append(a.mean())
        variances.append(a.var())
salesmeansdetrend=means
salesvariancesdetrend=variances


# In[17]:


x=range(0,144)
plt.plot(x,salesmeansdetrend,marker="o",markersize=10,color="purple",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title('Mean number of passengers by subset detrended' ,pad=40)
plt.ylabel("Mean number of passengers detrended")
plt.xlabel("Subset")
plt.grid()


# In[18]:


x=range(0,144)
plt.plot(x,salesvariancesdetrend,marker="o",markersize=10,color="purple",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title("Variance for number of passengers by subset detrended", pad=40)
plt.ylabel("Variance for number of passengers detrended")
plt.xlabel("Subset")
plt.grid()


# ## 8. Using the logarithmic transformation method, and differencing method, detrend the data. You need to use the numpy library and convert the air passenger numbers into logarithmic scale . Then take the difference of two adjacent observation and plot the result. The result should be almost stationary.

# In[19]:


air['lognumpass']=np.nan
for i in range(0,144):
    air.iloc[i,3]=np.log(air.iloc[i,1])
air.head()


# In[23]:


logdifferentiation=[]

for i in range(0, 144):
    if i==0:
        c=0
        logdifferentiation.append(c)
    else:
        c= air.iloc[i,3] - air.iloc[(i - 1),3]
        logdifferentiation.append(c)
#logdifferentiation


# In[24]:


air['logdifferentiation']=logdifferentiation
#air


# In[25]:


plt.plot(air['Month'],air['logdifferentiation'],marker="o",markersize=10,color="green",linestyle="solid",linewidth=2)
plt.title("Airline number of passengers by month from 1949 to 1960 detrended with 1st and logarithm differentiation",pad=40)
plt.ylabel("Differentiation")
plt.xlabel("Month")
plt.grid()


# ## 9. Is the transformed data now stationary? Justify your answer. Calculate the average over the entire dataset and show the average plot. Calculate the variance and show the transformation converts the non-stationary data into a stationary data.

# In[26]:


means=[]
variances=[]
a=[]
for i in range(0,144):
    if i==0:
        a=air.iloc[i,4]
        means.append(a)
        variances.append(0)
    else:
        a=air.iloc[:i+1,4]
        means.append(a.mean())
        variances.append(a.var())
salesmeanslogdetrend=means
salesvarianceslogdetrend=variances


# In[27]:


x=range(0,144)
plt.plot(x,salesmeanslogdetrend,marker="o",markersize=10,color="green",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title('Mean number of passengers by subset logdetrended' ,pad=40)
plt.ylabel("Mean number of passengers logdetrended")
plt.xlabel("Subset")
plt.grid()


# In[28]:


x=range(0,144)
plt.plot(x,salesvarianceslogdetrend,marker="o",markersize=10,color="green",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title("Variance for number of passengers by subset logdetrended", pad=40)
plt.ylabel("Variance for number of passengers logdetrended")
plt.xlabel("Subset")
plt.grid()

