#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # LAB # 3 (Autocorrelation Function)

# # Using the Python program and using only the “numpy” and “matplotlib” library perform the following tasks:
# ## 1- The equation to estimate autocorrelation is given as: Note: k is the # of lags and T is # of samples.

# ## 2- Let suppose y vectors is given as y(t) = [3, 9, 27, 81,243]. Without use of python or any other computer program, manually calculate the 𝜏0, 𝜏1, 𝜏2, 𝜏3, 𝜏4. Number of lags is 5.

# ## 3- Create a white noise with zero mean and standard deviation of 1 and 1000 samples.

# In[3]:


m=0
o=1
W=o * np.random.randn(1000) + m


# ## 4- Write a python code to estimate Autocorrelation Function. Note: You need to use the equation given in the textbook.

# In[4]:


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


# In[5]:


Y = [1, 2, 3, 4, 5]
autocorrelation(Y)


# In[6]:


yy = [3, 9, 27, 81,243]
autocorrelation(yy)


# ### a. Plot the ACF for the generated data in step 3. The ACF needs to be plotted using “stem” command.

# In[177]:


plt.rcParams["figure.figsize"] = (30,15)
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize']=40
plt.rcParams['ytick.labelsize']=40
plt.rcParams['legend.fontsize']=40


# In[178]:


x2=list()
for i in range(0,21):
    x2.append(i)
x1=[-1,-2,-3,-4,-5,-6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20]
x=x1+x2


# In[179]:


ADFWN1=autocorrelation(W)[1:21]
ADFWN2=autocorrelation(W)[0:21]
ADFWN=ADFWN1+ADFWN2
plt.stem(x,ADFWN, use_line_collection=True)
plt.title('Autocorrelation formula for White noise',pad=20)
plt.ylabel('Autocorrelation')
plt.xlabel('Lag #')
plt.show()


# ### b. Plot both the generated WN in step 3 versus time and plot the histogram.
# 

# In[180]:


plt.plot(W)
plt.title('White Noise through time',pad=40)
plt.xlabel('Time')
plt.ylabel('Generated White Noise')
plt.show()

plt.hist(W)
plt.title('White Noise Distribution',pad=40)
plt.xlabel('White Noise value')
plt.ylabel('Number of Observations')
plt.show()


# ### c. Write down your observations about the AFC plot, histogram and the time plot of the generated WN.

# ## 5- Load the time series dataset tute1.csv (from LAB#1)

# ### a. Using python code written in the previous step, plot the ACF for the “Sales” and “Sales” versus time next to each other. You can use subplot command.

# In[181]:


tute1=pd.read_csv('tute1.csv')
tute1.head()


# In[182]:


x2=list()
for i in range(0,21):
    x2.append(i)
x1=[-1,-2,-3,-4,-5,-6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20]
x=x1+x2


# In[183]:


#ADFsales1


# In[184]:


#ADFsales2


# In[185]:


#!pip install statsmodels

from statsmodels.tsa.stattools import adfuller
ADFX= adfuller(tute1['Sales'])
print('ADF Statistic for Sales: %f' % ADFX[0])
print('p-value for Sales: %f' % ADFX[1])
print('Critical Values for Sales:')
for key, value in ADFX[4].items():
    print('\t%s: %.3f' % (key, value))

ACFsaless=autocorrelation(tute1['Sales'])
ACFsales2=ACFsaless[0:21]
ACFsales1=ACFsaless[1:21]
ACFsales=ACFsales1+ACFsales2

plt.subplot(2,2,1)
plt.stem(x,ACFsales,use_line_collection=True)
plt.title('Autocorrelation for Sales',pad=20)
plt.ylabel('Autocorrelation')
plt.xlabel('Lag #')


plt.subplot(2,2,2)
plt.plot(tute1['Sales'])
plt.title('Sales Through Time',pad=10)
plt.xlabel('Time')
plt.ylabel('Sales value')


plt.tight_layout()
plt.show()


# ### b. Using python code written in the previous step, plot the ACF for the “AdBudget” and “AdBudegt” versus time next to each other. You can use subplot command.

# In[186]:


x2=list()
for i in range(0,21):
    x2.append(i)
x1=[-1,-2,-3,-4,-5,-6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20]
x=x1+x2


# In[187]:


#ADFAdbug1


# In[188]:


#ADFAdbug2


# In[189]:


ADFX= adfuller(tute1['AdBudget'])
print('ADF Statistic for AdBudget: %f' % ADFX[0])
print('p-value for AdBudget: %f' % ADFX[1])
print('Critical Values for AdBudget:')
for key, value in ADFX[4].items():
    print('\t%s: %.3f' % (key, value))

ACFAdbug=autocorrelation(tute1['AdBudget'])
ACFAdbug1=ACFAdbug[0:21]
ACFAdbug2=ACFAdbug[1:21]
ACFAdBudget=ACFAdbug2+ACFAdbug1


plt.subplot(2,2,1)
plt.stem(x,ACFAdBudget,use_line_collection=True)
plt.title('Autocorrelation for AdBudget',pad=20)
plt.ylabel('Autocorrelation')
plt.xlabel('Lag #')


plt.subplot(2,2,2)
plt.plot(tute1['AdBudget'])
plt.title('AdBudget Through Time',pad=10)
plt.xlabel('Time')
plt.ylabel('AdBudget value')

plt.tight_layout()
plt.show()


# ### c. Using python code written in the previous step, plot the ACF for the “GDP” and “GDP” versus time next to each other. You can use subplot command.
# 

# In[190]:


x2=list()
for i in range(0,21):
    x2.append(i)
x1=[-1,-2,-3,-4,-5,-6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20]
x=x1+x2


# In[191]:


ADFX= adfuller(tute1['GDP'])
print('ADF Statistic for GDP: %f' % ADFX[0])
print('p-value for GDP: %f' % ADFX[1])
print('Critical Values for GDP:')
for key, value in ADFX[4].items():
    print('\t%s: %.3f' % (key, value))
    
ACFGDPS=autocorrelation(tute1['GDP'])
ACFGDP1=ACFGDPS[0:21]
ACFGDP2=ACFGDPS[1:21]
ACFGDP=ACFGDP2+ACFGDP1


plt.subplot(2,2,1)
plt.stem(x,ACFGDP,use_line_collection=True)
plt.title('Autocorrelation for GDP',pad=20)
plt.ylabel('Autocorrelation')
plt.xlabel('Lag #')


plt.subplot(2,2,2)
plt.plot(tute1['GDP'])
plt.title('GDP Through Time',pad=10)
plt.xlabel('Time')
plt.ylabel('GDP value')

plt.tight_layout()
plt.show()


# ### d. Run the ADF-test for part a , b, and c and display them next to ACF and time plot in each section.
# 
# ### e. Write down your observations about the correlation between stationary and non- stationary time series (if there is any) and autocorrelation function?
# 
# ### f. The number lags used for this question is 20.

# In[ ]:




