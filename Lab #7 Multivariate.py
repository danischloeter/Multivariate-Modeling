import numpy as np
import pandas as pd
# from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import signal
#


# Multivariate Modeling
# LAB # 7 Autoregressive & Moving Average Model DATS 6450
# Using the Python program and appropriate libraries perform the following tasks:
# 1- Let consider an AR(2) process as
# 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) − 0.2𝑦(𝑡 − 2) = 𝑒(𝑡)
# Where 𝑒(𝑡) is a white noise

# b) Write a python code and simulate above process for 100 samples when 𝑒(𝑡) as a WN (2,1).

T=100
Me=2
e=1*np.random.randn(T)+Me

def processar2(e):
    y=np.zeros(len(e))

    for i in range(len(e)):
        if i==0:
            y[i]=e[i]
        elif i==1:
            y[i]=0.5*y[i-1]+e[i]
        else:
            y[i]=0.5*y[i-1]+0.2*y[i-2]+e[i]
    return(y)

y=processar2(e)

# c) With the help of the software approximate the mean of y(t) and display the results on the console:

#“Mean of y(t) - 0.5 y(t-1) - .2y(t-2) = e(t)” is ___________

#My=Me/(1-0.5-0.2)
my=np.mean(y)
print("The Mean of y(t) - 0.5 y(t-1) - .2y(t-2) = e(t) for 100 samples is ",np.round(my,2))

# d) Repeat step b for 1000 and 10000 samples and compare the theoretical results (part a) and experimental results (part c).
Td1=1000
Me=2
ed1=1*np.random.randn(Td1)+Me

yd1=processar2(ed1)
myd1=np.mean(yd1)
print("The Mean of y(t) - 0.5 y(t-1) - .2y(t-2) = e(t) for 1000 samples is ",np.round(myd1,2))


Td=10000
Me=2
ed=1*np.random.randn(Td)+Me

yd=processar2(ed)
myd=np.mean(yd)
print("The Mean of y(t) - 0.5 y(t-1) - .2y(t-2) = e(t) for 10000 samples is ",np.round(myd,2))

# e) Create a table for the results achieved in the previous parts. Write down your observations.

Resultsmean=pd.DataFrame(np.zeros((3,3)))
Resultsmean.columns=['Samples','Theorical mean','Mean']
Resultsmean.iloc[0,0]=T
Resultsmean.iloc[1,0]=Td1
Resultsmean.iloc[2,0]=Td

Resultsmean.iloc[:,1]=6.666

Resultsmean.iloc[0,2]=my
Resultsmean.iloc[1,2]=myd1
Resultsmean.iloc[2,2]=myd

print(Resultsmean)

#  f) Let 𝑒(𝑡) be a WN (0,1) with 100 samples. Find the theoretical variance of above process. You should not use python code to answer this question and the answer needs to be done manually.

#g) Write a python code and find the approximated variance and display the results on the console:
#Variance of y(t) - 0.5 y(t-1) - .2y(t-2) = e(t)” is


Tg=100
Me=0
eg=1*np.random.randn(Tg)+Me

yg=processar2(eg)

varg=np.var(yg)
print( 'The Variance of y(t) - 0.5 y(t-1) - .2y(t-2) = e(t) for 100 samples is ',np.round(varg,3) )

# h) Increase the number of samples to 1000 and 10000 and compare the theoretical variance versus the experimental (approximated) variance. Create a table and demonstrate the results through a table. Write down your observations.

Th1=1000
Me=0
eh1=1*np.random.randn(Th1)+Me

yh1=processar2(eh1)

varh1=np.var(yh1)
print( 'The Variance of y(t) - 0.5 y(t-1) - .2y(t-2) = e(t) for 1000 samples is ',np.round(varh1,3) )

Th2=10000
Me=0
eh2=1*np.random.randn(Th2)+Me

yh2=processar2(eh2)

varh2=np.var(yh2)
print( 'The Variance of y(t) - 0.5 y(t-1) - .2y(t-2) = e(t) for 10000 samples is ',np.round(varh2,3) )

Resultsvar=pd.DataFrame(np.zeros((3,3)))
Resultsvar.columns=['Samples','Theorical variance','Variance']

Resultsvar.iloc[0,0]=Tg
Resultsvar.iloc[1,0]=Th1
Resultsvar.iloc[2,0]=Th2

Resultsvar.iloc[:,1]=1.7094

Resultsvar.iloc[0,2]=varg
Resultsvar.iloc[1,2]=varh1
Resultsvar.iloc[2,2]=varh2

print(Resultsvar)

# 2- Let consider an MA(2) process as
# 𝑦(𝑡) = 𝑒(𝑡) + 0.1𝑒(𝑡 − 1) + 0.4𝑒(𝑡 − 2)

#Where 𝑒(𝑡) is a white noise.
#a) Without using a software calculate the theoretical mean of above process when 𝑒(𝑡) as a WN (2,1).

#b) Write a python code and simulate above process for 100 samples when 𝑒(𝑡) as a WN (2,1).

T=100
Me=2
e=1*np.random.randn(T)+Me

def processar2b(e):
    y=np.zeros(len(e))
    for i in range(len(e)):
        if i == 0:
            y[i] = e[i]
        elif i == 1:
            y[i] = 0.1 * e[i - 1] + e[i]
        else:
            y[i] = 0.1 * e[i - 1] + 0.4 * e[i - 2] + e[i]
    return(y)

y=processar2b(e)

#c. With the help of the software approximate the mean of y(t) and display the results on the console:
#“Mean of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2)” is ___________
#My=Me/(1-0.5-0.2)
my=np.mean(y)
print("The Mean of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) for 100 samples is ",np.round(my,2))

# d) Repeat step b for 1000 and 10000 samples and compare the theoretical results (part a) and experimental results (part c).
Td1=1000
Me=2
ed1=1*np.random.randn(Td1)+Me

yd1=processar2b(ed1)
myd1=np.mean(yd1)
print("The Mean of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) for 1000 samples is ",np.round(myd1,2))


Td=10000
Me=2
ed=1*np.random.randn(Td)+Me

yd=processar2b(ed)
myd=np.mean(yd)
print("The Mean of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) for 10000 samples is ",np.round(myd,2))

# e) Create a table for the results achieved in the previous parts. Write down your observations.

Resultsmean=pd.DataFrame(np.zeros((3,3)))
Resultsmean.columns=['Samples','Theorical mean','Mean']
Resultsmean.iloc[0,0]=T
Resultsmean.iloc[1,0]=Td1
Resultsmean.iloc[2,0]=Td

Resultsmean.iloc[:,1]=3

Resultsmean.iloc[0,2]=my
Resultsmean.iloc[1,2]=myd1
Resultsmean.iloc[2,2]=myd

print(Resultsmean)


# g) Write a python code and find the approximated variance and display the results on the console:
#Variance of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2)”)” is


Tg=100
Me=0
eg=1*np.random.randn(Tg)+Me

yg=processar2b(eg)

varg=np.var(yg)
print( 'The Variance of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2)”)” for 100 samples is ',np.round(varg,3) )

# h) Increase the number of samples to 1000 and 10000 and compare the theoretical variance versus the experimental (approximated) variance. Create a table and demonstrate the results through a table. Write down your observations.

Th1=1000
Me=0
eh1=1*np.random.randn(Th1)+Me

yh1=processar2b(eh1)

varh1=np.var(yh1)
print( 'The Variance of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2)”)” for 1000 samples is ',np.round(varh1,3) )

Th2=10000
Me=0
eh2=1*np.random.randn(Th2)+Me

yh2=processar2b(eh2)

varh2=np.var(yh2)
print( 'The Variance of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2)”)” for 10000 samples is ',np.round(varh2,3) )





Resultsvar=pd.DataFrame(np.zeros((3,3)))
Resultsvar.columns=['Samples','Theorical variance','Variance']

Resultsvar.iloc[0,0]=Tg
Resultsvar.iloc[1,0]=Th1
Resultsvar.iloc[2,0]=Th2

Resultsvar.iloc[:,1]=1.7094

Resultsvar.iloc[0,2]=varg
Resultsvar.iloc[1,2]=varh1
Resultsvar.iloc[2,2]=varh2

print(Resultsvar)


# 3) Let consider an ARMA(2,2) process as  𝑦(𝑡) − 0.5𝑦(𝑡 − 1) − 0.2𝑦(𝑡 − 2) = 𝑒(𝑡) + 0.1𝑒(𝑡 − 1) + 0.4𝑒(𝑡 − 2)
# Where 𝑒(𝑡) is a white noise (0,1).
# y(t)=+0.5y(t-1)+0.2y(t-2)+e(t)+0.1e(t-1)+0.4e(t-2)
# a) Simulate the above ARMA process using a for loop for 100 samples. Plot y(t) versus time.

T2=100
np.random.seed(3)
Me=0
e2=1*np.random.randn(T2)+Me

def processarma22(e):
    y=np.zeros(len(e))

    for i in range(len(e)):
        if i==0:
            y[i]=e[i]
        elif i==1:
            y[i]=(0.5*y[i-1]+e[i]+0.1*e[i-1])
        else:
            y[i]=0.5*y[i-1]+0.2*y[i-2]+e[i]+0.1*e[i-1]+0.4*e[i-2]
    return(y)

y=processarma22(e2)


plt.figure()
plt.plot(y,color='orange')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('ARMA(2,2) y(t)=0.5*e(t-1)+0.2*e(t-2)+e(t)')
plt.show()


# b) Using the “scipy libaray” and “dlsim” command, simulate above ARMA model. Display the first 5 values of y(t) calculated in part a) and part b) at the console. These two arrays must be identical.

system=([1,0.1,0.4],[1,-0.5,-0.2],1)
t_in=np.arange(0)
t_out,y_out= signal.dlsim(system,e2)


print("First 5 values of y(t)- for the for loop method",y[:5])
print("First 5 values of y(t)- for dlsimp method",y_out[:5])

# c) Using ADF test and plot of mean a and variance over time, check the stationarity of this process? Explain your answer.

print('The mean is :'+str(round(y.mean(),2))+' and the variance is :'+str(round(y.var(),2))+' with standard deviation :'+str(round(y.std(),2)))
means=[]
variances=[]
for i in range(0,T2):
    if i==0:
        a=y[i]
        means.append(a)
        variances.append(0)
    else:
        a=y[:i+1]
        means.append(a.mean())
        variances.append(a.var())
Ymeans=means
Yvariances=variances

x = range(0, T2)


plt.figure()
plt.plot(y,label='Y values for ARMA(2,2)',color='orange')
plt.plot(x, Yvariances, marker="o", markersize=2,label='Y variances by subset', color="green", linestyle="solid", linewidth=2)
plt.plot(x, Ymeans, marker="x", markersize=2, color="purple",label="Y means by subset", linestyle="solid", linewidth=2)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('ARMA(2,2) y(t) − 0.5y(t − 1) − 0.2y(t − 2) = e(t) + 0.1e(t − 1) + 0.4e(t − 2)')
plt.legend()
plt.show()


from statsmodels.tsa.stattools import adfuller
ADFY= adfuller(y)
print('ADF Statistic for Y: %f' % ADFY[0])
print('p-value for Y: %f' % ADFY[1])
print('Critical Values for Y:')
for key, value in ADFY[4].items():
	print('\t%s: %.3f' % (key, value))

# d) Plot the autocorrelation of y(t) for 20, 40 and 80 lags. Write down your observations.

def autocorrelation(y,lags):
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
    x2 = list()
    x1=list()
    for i in range(0, lags+1):
        x2.append(i)
        if i!=lags:
            x1.append(-i-1)
    x = x1[::-1] + x2
    return tk,x

lags=20
autocorrmean,x=autocorrelation(y,lags)
autocorrmean=autocorrmean[:lags+1]
autocorrmean1=autocorrmean[::-1]
autocorrmean=autocorrmean[1:]
autocorrmean=autocorrmean1+autocorrmean

plt.stem(x,autocorrmean, use_line_collection=True)
plt.title('Autocorrelation function for the ARMA(2,2) 20 lags',pad=40)
plt.ylabel('Autocorrelation value')
plt.xlabel('Lag')
plt.figure()
plt.show()

lags=40
autocorrmean,x=autocorrelation(y,lags)
autocorrmean=autocorrmean[:lags+1]
autocorrmean1=autocorrmean[::-1]
autocorrmean=autocorrmean[1:]
autocorrmean=autocorrmean1+autocorrmean

plt.stem(x,autocorrmean, use_line_collection=True)
plt.title('Autocorrelation function for the ARMA(2,2) 40 lags',pad=40)
plt.ylabel('Autocorrelation value')
plt.xlabel('Lag')
plt.figure()
plt.show()

lags=80
autocorrmean,x=autocorrelation(y,lags)
autocorrmean=autocorrmean[:lags+1]
autocorrmean1=autocorrmean[::-1]
autocorrmean=autocorrmean[1:]
autocorrmean=autocorrmean1+autocorrmean

plt.stem(x,autocorrmean, use_line_collection=True)
plt.title('Autocorrelation function for the ARMA(2,2) 80 lags',pad=40)
plt.ylabel('Autocorrelation value')
plt.xlabel('Lag')
plt.figure()
plt.show()