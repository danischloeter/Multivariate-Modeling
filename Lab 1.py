
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # 1. Load the time series data called tute1

# In[2]:


tute1=pd.read_csv('tute1.csv')
tute1.head()


# This date relates to the quarterly sales for a small company over period 1981-2005.
# 
# Sales contains the quarterly sales, AdBudget is the advertisement budget and GPD is the gross domestic product for a small company.

# In[3]:


tute1.info()


# In[4]:


tute1['Date']=pd.to_datetime(tute1['Date'])


# In[5]:


tute1.info()


# # 2. Plot Sales, AdBudget and GPD versus time step.

# In[68]:


plt.rcParams["figure.figsize"] = (30,15)
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize']=24
plt.rcParams['ytick.labelsize']=24
plt.rcParams['legend.fontsize']=24


# In[69]:


plt.plot(tute1['Date'],tute1['Sales'],marker="o",markersize=10,color="green",linestyle="solid",linewidth=2)
plt.legend()
 
plt.title("Number of Sales by date",pad=40)
plt.ylabel("Number of sales")
plt.xlabel("Date")
plt.grid()


# In[70]:


plt.plot(tute1['Date'],tute1['AdBudget'],marker="o",markersize=10,color="blue",linestyle="solid",linewidth=2)
plt.legend()

plt.title("AdBudget by date",pad=40)
plt.ylabel("AdBudget")
plt.xlabel("Date")
plt.grid()


# In[71]:


plt.plot(tute1['Date'],tute1['GDP'],marker="o",markersize=10,color="red",linestyle="solid",linewidth=2)
plt.legend()

plt.title("GDP by Date",pad=40)
plt.ylabel("GDP")
plt.xlabel("Date")
plt.grid()


# # 3. Find the time series statistics (average, variance and standard deviation) of Sales, AdBudget and GPD and display the Average, variance and standard deviation

# In[72]:


tute1.describe()


# In[73]:


print('The Sales mean is :'+str(round(tute1['Sales'].mean(),2))+' and the variance is :'+str(round(tute1['Sales'].var(),2))+' with standard deviation :'+str(round(tute1['Sales'].std(),2)))
print()
print('The AdBudget mean is :'+str(round(tute1['AdBudget'].mean(),2))+' and the variance is :'+str(round(tute1['AdBudget'].var(),2))+' with standard deviation :'+str(round(tute1['AdBudget'].std(),2)))
print()
print('The GDP mean is :'+str(round(tute1['GDP'].mean(),2))+' and the variance is :'+str(round(tute1['GDP'].var(),2))+' with standard deviation :'+str(round(tute1['GDP'].std(),2)))


# # 3. Prove this time series data is stationary. Hint: In order to show a process is stationary, you need to show that data statistics is not changing by time. You need to create a 100 sub-sequences from the original sequence and save the mean and variance of each sun-sequence. Plot all means and variances and show that the means and variances are almost constant. To create sub-sequences, start with a sequence with the first sales data and find the mean. Then create another sub-sequence by adding the second sales date to the first sub-sequence, then find the corresponding mean. Repeat this process till you added the last sales date to the last sub- sequence and find the mean. Repeat the same procedures for variances.

# In[74]:


#Sales
means=[]
variances=[]
for i in range(0,100):
    if i==0:
        a=tute1.iloc[i,1]
        means.append(a)
        variances.append(0)
    else:
        a=tute1.iloc[:i+1,1]
        means.append(a.mean())
        variances.append(a.var())
salesmeans=means
salesvariances=variances


# In[75]:


# len(salesmeans)
# len(salesstd)
# len(salesvariances)


# In[76]:


#AdBudget
means=[]
variances=[]
for i in range(0,100):
    if i==0:
        a=tute1.iloc[i,2]
        means.append(a)
        variances.append(0)
    else:
        a=tute1.iloc[:i+1,2]
        means.append(a.mean())
        variances.append(a.var())
AdBudgetmeans=means
AdBudgetvariances=variances        


# In[77]:


#GDP
means=[]
variances=[]
for i in range(0,100):
    if i==0:
        a=tute1.iloc[i,3]
        means.append(a)
        variances.append(0)
    else:
        a=tute1.iloc[:i+1,3]
        means.append(a.mean())
        variances.append(a.var())
GDPmeans=means
GDPvariances=variances


# # 4. Plot all means and variance. Write down you observation about if this time series date is stationary or not? Why?

# In[78]:


x=range(0,100)
plt.plot(x,salesmeans,marker="o",markersize=10,color="green",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title("Sales means by subset", pad=40)
plt.ylabel("Sales means")
plt.xlabel("Subset")
plt.grid()


# In[79]:


x=range(0,100)
plt.plot(x,salesvariances,marker="o",markersize=10,color="green",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title("Sales variances by subset", pad=40)
plt.ylabel("Sales variances")
plt.xlabel("Subset")
plt.grid()


# In[80]:


x=range(0,100)
plt.plot(x,AdBudgetmeans,marker="o",markersize=10,color="blue",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title("AdBudget means by subset", pad=40)
plt.ylabel("AdBudget means")
plt.xlabel("Subset")
plt.grid()


# In[81]:


x=range(0,100)
plt.plot(x,AdBudgetvariances,marker="o",markersize=10,color="blue",linestyle="solid",linewidth=2)
plt.legend()

plt.title("AdBudget variances by subset",pad=40)
plt.ylabel("AdBudget variances")
plt.xlabel("Subset")
plt.grid()


# In[82]:


x=range(0,100)
plt.plot(x,GDPmeans,marker="o",markersize=10,color="red",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title("GDP means by subset",pad=40)
plt.ylabel("GDP means")
plt.xlabel("Subset")
plt.grid()


# In[83]:


x=range(0,100)
plt.plot(x,GDPvariances,marker="o",markersize=10,color="red",linestyle="solid",linewidth=2)
plt.legend()

 
plt.title("GDP variances by subset",pad=40)
plt.ylabel("GDP variances")
plt.xlabel("Subset")
plt.grid()

