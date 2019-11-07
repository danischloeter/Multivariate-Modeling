
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math  


# # Homework 2 -  Correlation coefficient and Scatter plot

# Using the Python program and “pandas”, “matplotlib.pyplot” and “numpy” library perform the following tasks:

# ## 1- Using Python, creates two random variables X and Y with normal distribution zero mean μ = 0 and standard deviation of 𝜎=1. Number of samples = 10000. Hint: You can use the following command in Python for the generation of normal distribution:
# “𝜎 * np.random.randn(# of samples) + μ

# In[20]:


m=0
o=1
X=o * np.random.randn(1000) + m
Y=o * np.random.randn(1000) + m


# ## 2- Using the ADF-test, check the stationarity of the random variable X and Y. Display the ADF- Statistics for X and Y. Are X and Y stationary? Justify your answer.

# In[21]:


from statsmodels.tsa.stattools import adfuller
ADFX= adfuller(X)
print('ADF Statistic for X: %f' % ADFX[0])
print('p-value for X: %f' % ADFX[1])
print('Critical Values for X:')
for key, value in ADFX[4].items():
    print('\t%s: %.3f' % (key, value))


# In[22]:


ADFY= adfuller(Y)
print('ADF Statistic for Y: %f' % ADFY[0])
print('p-value for Y: %f' % ADFY[1])
print('Critical Values for Y:')
for key, value in ADFY[4].items():
	print('\t%s: %.3f' % (key, value))


# ## 3- Create two more random variables G and Z where G = X and Z = -X

# In[23]:


G=X
Z=-X


# ## 4- Using the function written in LAB#2 ( correlation_coefficient_cal) find the correlation between (X, Y) and (X, Z) and (X,G).

# In[24]:


def correlation_coefficent_cal(x,y):
    '''It returns the correlation coefficient for two given datasets'''

    meanx=np.mean(x)
    meany=np.mean(y)
    numerator=np.sum((x-meanx)*(y-meany))
    denominator1=np.sum((x-meanx)**2)
    denominator2=np.sum((y-meany)**2)
    r=numerator/(np.sqrt(denominator1)*np.sqrt(denominator2))
    return r


# In[25]:


cor_XY=correlation_coefficent_cal(X,Y)
cor_XZ=correlation_coefficent_cal(X,Z)
cor_XG=correlation_coefficent_cal(X,G)


# In[26]:


#cor_XY


# In[27]:


#cor_XZ


# In[28]:


#cor_XG


# ## 5- Graph the histogram plot of X, Y, G, Z. Add an appropriate x-label, y-label and title.

# In[29]:


plt.hist(X)
plt.title('Histogram of X')
plt.xlabel('X value')
plt.ylabel('Number of results')
plt.show()


# In[30]:


plt.hist(Y)
plt.title('Histogram of Y')
plt.xlabel('Y value')
plt.ylabel('Number of results')
plt.show()


# In[31]:


plt.hist(G)
plt.title('Histogram of G')
plt.xlabel('G value')
plt.ylabel('Number of results')
plt.show()


# In[32]:


plt.hist(Z)
plt.title('Histogram of Z')
plt.xlabel('Z value')
plt.ylabel('Number of results')
plt.show()


# ## 6- Plot the scatter plot between X, Y. Update the graph title with the r value between X and Y, as a variable. Add an appropriate x-label and y-label to your graph.

# In[33]:


plt.scatter(X,Y,color="purple")
plt.title("Scatter plot of X and Y with r ={}".format(round(cor_XY,3)),pad=10)
plt.ylabel("Y")
plt.xlabel("X")
plt.grid()
plt.show()


# ## 7- Plot the scatter plot between X,Z. Update the graph title with the r value between X and Z, as a variable. Add an appropriate x-label and z-label to your graph.

# In[34]:


plt.scatter(X,Z,color="purple")
plt.title("Scatter plot of X and Z with r ={}".format(round(cor_XZ,3)),pad=10)
plt.ylabel("Z")
plt.xlabel("X")
plt.grid()
plt.show()


# ## 8- Plot the scatter plot between X,Z. Update the graph title with the r value between X and Z, as a variable. Add an appropriate x-label and y-label to your graph.

# In[35]:


plt.scatter(X,G,color="purple")
plt.title("Scatter plot of X and G with r ={}".format(round(cor_XG,3)),pad=10)
plt.ylabel("G")
plt.xlabel("X")
plt.grid()
plt.show()


# ## 9- Display the message as:
# ### i. The correlation coefficient between x and y is ______
# ### ii. The correlation coefficient between x and z is ______
# ### iii. The correlation coefficient between x and g is ______

# In[36]:


print('The correlation coefficient between x and y is '+str(round(cor_XY,3)))
print('The correlation coefficient between x and z is '+str(round(cor_XZ,3)))
print('The correlation coefficient between x and g is '+str(round(cor_XG,3)))


# ## 10- Does the r_xy value make sense with respect to the corresponding scatter plot. Explain why?

# ## 11- Does the r_xz value make sense with respect to the corresponding scatter plot. Explain why?

# ## 12- Does the r_xg value make sense with respect to the corresponding scatter plot. Explain why?

# ## Write a report and answer all the above questions. You need to include graphs and tables ( if needed) into your report. Upload your homework # 2 report as a single pdf file and the .py program to the Blackboard.
