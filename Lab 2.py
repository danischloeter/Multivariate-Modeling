
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Lab 2 - Correlation coefficient and Scatter plot

# ## 1-  Write a python function called “ correlation_coefficent_cal(x,y)” that implement the correlation coefficient. The formula for correlation coefficient is given below. The function should be written in a general form than can work for any dataset x and dataset y. The return value for this function is r.
#  

# In[33]:


def correlation_coefficent_cal(x,y):
    '''It returns the correlation coefficient for two given datasets'''

    meanx=np.mean(x)
    meany=np.mean(y)
    numerator=np.sum((x-meanx)*(y-meany))
    denominator1=np.sum((x-meanx)**2)
    denominator2=np.sum((y-meany)**2)
    r=numerator/(np.sqrt(denominator1)*np.sqrt(denominator2))
    return r


# ## 2- Test the “ correlation_coefficent_cal(x,y)” function with the following simple dataset. The x and y here are dummy variable and should be replaced by any other dataset.

# In[34]:


X = [1, 2, 3, 4, 5]
Y = [1, 2, 3, 4, 5]
Z = [-1, -2, -3, -4,-5] 
G = [1,1,0,-1,-1,0,1] 
H = [0,1,1,1,-1,-1,-1]


# In[35]:


r_xy=correlation_coefficent_cal(X,Y)
r_xz=correlation_coefficent_cal(X,Z)
r_gh=correlation_coefficent_cal(G,H)


# ### a. Plot the scatter plot between X, Y

# In[36]:


plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12
plt.rcParams['legend.fontsize']=12


# In[37]:


plt.scatter(X,Y,color="green")
 
plt.title("Scatter plot of X and Y with r ={}".format(round(r_xy,3)),pad=20)
plt.ylabel("Y")
plt.xlabel("X")
plt.show()


# ### b. Plot the scatter plot between X, Z

# In[38]:


plt.scatter(X,Z,color="orange")
 
plt.title("Scatter plot of X and Z with r ={}".format(round(r_xz,3)),pad=20)
plt.ylabel("Z")
plt.xlabel("X")
plt.show()


# ### c. Plot the scatter plot between G, H

# In[39]:


plt.scatter(G,H,color="purple")
 
plt.title("Scatter plot of G and H with r ={}".format(round(r_gh,5)),pad=20)
plt.ylabel("G")
plt.xlabel("H")
plt.show()


# ### d. Without using Python program, implement the above formula to derive the r_xy, r_xz, r_gh. You should NOT use computer to answer this section. You need to show all your work for this section on the paper.

# ### e. Calculate r_xy , r_xz and r_gh using the written function “correlation_coefficent_cal(x,y)”.

# In[40]:


r_xy


# In[41]:


r_xz


# In[42]:


r_gh


# ### f. Compare the answer in section d and e. Any difference in value?

#  The answers for questions d and e are exactly the same and imply a strong positive relationship for xy a strong negative relationship for xz and no relationship for gh

# ### g. Display the message as:

# #### i. The correlation coefficient between x and y is ______
# #### ii. The correlation coefficient between x and z is ______
# #### iii. The correlation coefficient between g and h is ______

# In[43]:


print('The correlation coefficient between x and y is '+str(round(r_xy,3)))
print('The correlation coefficient between x and z is '+str(round(r_xz,3)))
print('The correlation coefficient between g and h is '+str(round(r_gh,3)))


# ### h. Add an appropriate x-axis label and y-axis label to all your scatter graphs.

# ### i. Include the r_xy , r_xz, r_gh as a variable on the scatter plot title in part a and part b. The code should be written in a way that the r value changes on the figure title automatically. Hint: You can use the following command: plt.title("Scatter plot of X and Y with r ={}".format(r_xy))

# ### j. Does the calculated r_xy, r_xz and r_gh make sense with respect to the scatter plots? Explain why?

# Yes. The distribution of the observations in each scatter plots relates to the calculated r for that set.

# ## 3- Load the time series data called tute1. The tute1 dataset is the same dataset used in LAB1.

# In[44]:


tute1=pd.read_csv('tute1.csv')
tute1.head()


# In[45]:


tute1['Date']=pd.to_datetime(tute1['Date'])
tute1.info()


# ## 4- The dataset relates to the quarterly sales for a small company over period 1981-2005.

# ## 5- Sales contains the quarterly sales, AdBudget is the advertisement budget and GPD is the gross domestic product for a small company.
# 

# ## 6- Plot Sales, AdBudget and GPD versus time steps.

# In[47]:


plt.plot(tute1['Date'],tute1['Sales'],marker="o",markersize=10,color="green",linestyle="solid",linewidth=2)
plt.legend()
 
plt.title("Number of Sales by date",pad=40)
plt.ylabel("Number of sales")
plt.xlabel("Date")
plt.grid()
plt.show()


# In[48]:


plt.plot(tute1['Date'],tute1['AdBudget'],marker="o",markersize=10,color="blue",linestyle="solid",linewidth=2)
plt.legend()

plt.title("AdBudget by date",pad=40)
plt.ylabel("AdBudget")
plt.xlabel("Date")
plt.grid()
plt.show()


# In[49]:


plt.plot(tute1['Date'],tute1['GDP'],marker="o",markersize=10,color="red",linestyle="solid",linewidth=2)
plt.legend()

plt.title("GDP by Date",pad=40)
plt.ylabel("GDP")
plt.xlabel("Date")
plt.grid()
plt.show()


# ## 7- Graph the scatter plot for Sales and GDP. ( y-axis plot Sales and x-axis plot GDP). Add the appropriate x-label and y-label. Don’t add any title in this step. This needs to be updated in step 11.

# In[50]:


tuter_xy=correlation_coefficent_cal(tute1['GDP'],tute1['Sales'])
plt.scatter(tute1['GDP'],tute1['Sales'],color="purple")

plt.title("Scatter plot of GDP and Sales with r ={}".format(round((tuter_xy),5)), pad=20)
plt.ylabel("Sales")
plt.xlabel("GDP")
plt.grid()
plt.show()


# ## 8- Graph the scatter plot for Sales and AdBudget. ( y-axis plot Sales and x-axis plot AdBudget). Add the appropriate x-label and y-label. Don’t add any title in this step. This needs to be updated in step 11.

# In[51]:


tuter_yz=correlation_coefficent_cal(tute1['AdBudget'],tute1['Sales'])
plt.scatter(tute1['AdBudget'],tute1['Sales'],color="purple")

plt.title("Scatter plot of AdBudget and Sales with r ={}".format(round(tuter_yz,4)),pad=20)
plt.ylabel("Sales")
plt.xlabel("AdBudget")
plt.grid()
plt.show()


# ## 9- Call the function “correlation_coefficent_cal(x,y)” with y as the Sales data and the x as the GDP data. Save the correlation coefficient between these two variables as r_xy. Display the following message: “The correlation coefficient between the Sales value and GDP is _________”. Does the r_xy value make sense with respect to the scatter plot graphed in step 7. Explain why?

# In[20]:


tuter_xy=correlation_coefficent_cal(tute1['GDP'],tute1['Sales'])
print("The correlation coefficient between the Sales value and GDP is "+str(round(r_xy,3))+".")


# ## 10- Call the function “correlation_coefficent_cal(y,z)” with y as the Sales data and the z as the AdBudget data. Save the correlation coefficient between these two variables as r_yz. Display the following message: “The correlation coefficient between the Sales value and AdBudget is _________”. Does the r_yz value make sense with respect to the scatter plot graphed in step 8. Explain why?

# In[21]:


tuter_yz=correlation_coefficent_cal(tute1['AdBudget'],tute1['Sales'])
print("The correlation coefficient between the Sales value and GDP is "+str(round(tuter_xy,2))+".")
print("The correlation coefficient between the Sales value and AdBudget is "+str(round(tuter_yz,2))+".")


# ## 11- Include the r_xy and r_yz in the title of the graphs developed in step 5 and 6. Write your code in a way that anytime r_xy and r_yz value changes it automatically updated on the figure title. Hint: you can use the following command: plt.title("Scatter plot of GDP and Sales with r ={}".format(r_xy))

# ## 12- By looking at the correlation coefficients, write down your observation about the effect of AdBudget data and GDP data on the Sales revenue

# ## 13- Perform the ADF-test and plot the histogram plot on the raw Sales data, first order difference Sales data and the logarithmic transformation of the Sales data. Which Sales dataset is stationary and which Sales dataset is non-stationary? Justify your answer according to the ADF- Statistics and the histogram plot.

# In[22]:


from statsmodels.tsa.stattools import adfuller
ADFsales= adfuller(tute1['Sales'])
print('ADF Statistic for Sales: %f' % ADFsales[0])
print('p-value for Sales: %f' % ADFsales[1])
print('Critical Values for Sales:')
for key, value in ADFsales[4].items():
    print('\t%s: %.3f' % (key, value))


# In[23]:


plt.hist(tute1['Sales'])
plt.title('Histogram of Sales')
plt.xlabel('Sales')
plt.ylabel('Number of results')
plt.show()


# In[24]:


tute1.head()


# In[25]:


differentiation=[]
b=[]
for i in range(0, 100):
    if i==0:
        b=0
        differentiation.append(b)
    else:
        b= tute1.iloc[i,1] - tute1.iloc[i - 1,1]
        differentiation.append(b)
#differentiation
tute1['salesdifferentiation']=differentiation
from statsmodels.tsa.stattools import adfuller
ADFsalesdiff= adfuller(tute1['salesdifferentiation'])
print('ADF Statistic for Sales with 1st differentiation: %f' % ADFsalesdiff[0])
print('p-value for Sales with 1st differentiation: %f' % ADFsalesdiff[1])
print('Critical Values for Sales with 1st differentiation:')
for key, value in ADFsalesdiff[4].items():
    print('\t%s: %.3f' % (key, value))


# In[26]:


plt.hist(tute1['salesdifferentiation'])
plt.title('Histogram of Sales 1 st differentiation')
plt.xlabel('Sales 1st differentiation')
plt.ylabel('Number of results')
plt.show()


# In[27]:


tute1.head()


# In[31]:


tute1['lognumpass']=np.nan
for i in range(0,100):
    tute1.iloc[i,5]=np.log(tute1.iloc[i,1])
logdifferentiation=[]

for i in range(0, 100):
    if i==0:
        c=0
        logdifferentiation.append(c)
    else:
        c= tute1.iloc[i,5] - tute1.iloc[(i - 1),5]
        logdifferentiation.append(c)
#logdifferentiation
tute1['logdifferentiation']=logdifferentiation
#tute1
from statsmodels.tsa.stattools import adfuller

ADFsaleslog= adfuller(tute1['logdifferentiation'])
print('ADF Statistic for Sales with logarithmic transformation: %f' %ADFsaleslog[0])
print('p-value for Sales with logarithmic transformation: %f' % ADFsaleslog[1])
print('Critical Values for Sales with logarithmic transformation:')
for key, value in ADFsaleslog[4].items():
    print('\t%s: %.3f' % (key, value))


# In[29]:


plt.hist(tute1['logdifferentiation'])
plt.title('Histogram of Sales log transformation')
plt.xlabel('Sales log transformation')
plt.ylabel('Number of results')
plt.show()


# In[30]:


tute1.head()


# ## Write a report and answer all the above questions. You need to include graphs and tables ( if needed) into your report. Upload your homework # 1 report as a single pdf file and the .py program to Blackboard.
