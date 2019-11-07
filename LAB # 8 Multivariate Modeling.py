import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Multivariate Modeling
# LAB # 8 Partial Correlation Coefficient DATS 6450
# Using the “numpy” and “pandas” library in Python program perform the followings steps
# 1- Load the “tute1.csv” dataset.
#

tute1=pd.read_csv('tute1.csv')
tute1.head()
tute1.info()

# 2- Write a python program that calculate the correlation coefficient between Sales and AdBuget
# and display the following message on the console.
# “Correlation Coefficient between Sales and AdBugdet is ________”

def correlation_coefficent_cal(x, y):
    '''It returns the correlation coefficient for two given datasets'''
    meanx = np.mean(x)
    meany = np.mean(y)
    numerator = np.sum((x - meanx) * (y - meany))
    denominator1 = np.sum((x - meanx) ** 2)
    denominator2 = np.sum((y - meany) ** 2)
    r = numerator / (np.sqrt(denominator1) * np.sqrt(denominator2))
    return r

corr=correlation_coefficent_cal(tute1['Sales'], tute1['AdBudget'])
print('Correlation Coefficient between Sales and AdBugdet is ',round(corr,4))

#3- Write a python program that calculate the correlation coefficient between AdBuget and GDP and display the following message on the console:
#“Correlation Coefficient between AdBugdet and GDP is ________”

corr2=correlation_coefficent_cal(tute1['AdBudget'], tute1['GDP'])

print('Correlation Coefficient between AdBugdet and GDP is ',round(corr2,4))

#4- Write a python program that calculate coefficient between Sales and GDP and display the following message on the console:
#“Correlation Coefficient between Sales and GDP is ________”

corr3=correlation_coefficent_cal(tute1['Sales'], tute1['GDP'])
print('Correlation Coefficient between Sales and GDP is ',round(corr3,4))

#5- Using the hypothesis test (t-test) show whether the correlation coefficients in step 2, 3, and 4 are statistically significant? Assume the level of confident to be 95% with two tails (𝛼 = 0.05 ).
# The critical t-value can be calculated as

n=100
#confounding variable
k=1


def tstatistics(r,n,k):
    DOF = n - 2 - k
    den = 1 - np.square(r)
    t = r* np.sqrt(DOF / den)
    return t


# From tables 1.984

# Sales and ADBudget
t=tstatistics(corr,n,k)
print('t-value for Sales and ADBudget',round(t,4))

#  ADBudget and GDP
t2=tstatistics(corr2,n,k)
print('t-value for ADBudget and GDP',round(t2,4))

# Sales and GDP
t3=tstatistics(corr3,n,k)
print('t-value for Sales and GDP',round(t3,4))



#6- Write a python program that calculate the partial correlation coefficient between Sales and AdBudegt.
# Using the hypothesis test, step 5, shows whether the derived coefficient is statistically significant.
# Write down your observation. Hint: Partial correlation coefficient between variable A and B with confounding variable C can be calculated as :

def partialcorrelationcoefficient(rab,rbc,rac):
    rabc=(rab-(rac*rbc))/(np.sqrt(1-np.square(rac))*np.sqrt(1-np.square(rbc)))
    return rabc
rsalesadb=partialcorrelationcoefficient(corr,corr2,corr3)
print('The partial Correlation Coefficient between Sales  and AdBugdet is',round(rsalesadb,4))

t4=tstatistics(rsalesadb,n,k)
print('t-value for partial correlation Sales and ADBudget',round(t4,4))

#7- Write a python program that calculate the partial correlation coefficient between Sales and GDP.
# Using the hypothesis test, step 5, shows whether this coefficient is statistically significant.
# Write down your observation.

rsalesgdp=partialcorrelationcoefficient(corr3,corr2,corr)
print('The partial Correlation Coefficient between Sales and GDP is',round(rsalesgdp,4))

t5=tstatistics(rsalesgdp,n,k)
print('t-value for partial correlation Sales and GDP',round(t5,4))

#8- Write a python program that calculate the partial correlation coefficient between AdBudegt and GDP.
# Using the hypothesis test, step 5, shows whether this coefficient is statistically significant.
# Write down your observation.

radbgdp=partialcorrelationcoefficient(corr2,corr3,corr)
print('The partial Correlation Coefficient betweenradbgdp AdBugdet and GDP is',round(radbgdp,4))

t6=tstatistics(radbgdp,n,k)
print('t-value for partial correlation AdBudget and GDP',round(t6,4))

# 9- Create a table and place all the results from step 2 through 8 inside the table.
# Compare the correlation coefficients and partial correlation coefficients for (Sales, AdBudget), (Sales, GDP) and (AdBudegt, GDP).
# Write down your observation.

Results=pd.DataFrame(np.zeros((3,3)))
Results.columns=['Variables','Correlation Coefficient','t-test']
Results1=pd.DataFrame(np.zeros((3,3)))
Results1.columns=['Variables','Partial Correlation Coefficient','t-test']

Results.iloc[0,0]='AdBudget and Sales'
Results.iloc[1,0]='AdBudget and GDP'
Results.iloc[2,0]='Sales and GDP'


Results.iloc[0,1]=corr
Results.iloc[1,1]=corr2
Results.iloc[2,1]=corr3

Results.iloc[0,2]=t
Results.iloc[1,2]=t2
Results.iloc[2,2]=t3

Results1.iloc[0,0]='AdBudget and Sales'
Results1.iloc[1,0]='AdBudget and GDP'
Results1.iloc[2,0]='Sales and GDP'

Results1.iloc[0,1]=rsalesadb
Results1.iloc[1,1]=radbgdp
Results1.iloc[2,1]=rsalesgdp

Results1.iloc[0,2]=t4
Results1.iloc[1,2]=t6
Results1.iloc[2,2]=t5

print(Results)
print(Results1)