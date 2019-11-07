import numpy as np
# from numpy import linalg as LA
# from scipy import signal
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from scipy import signal
#
#
#
# # Multivariate Modeling
# # LAB # 6 Autoregressive & Moving Average Model DATS 6450

## Using the Python program and appropriate libraries perform the following tasks: 1- Let consider an AR(2) process as
## 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) − 0.2𝑦(𝑡 − 2) = 𝑒(𝑡)
## Where 𝑒(𝑡) is a  WN (0,1).

##a) Using a for loop simulate above process for 1000 samples. Assume all initial conditions to be zero.

T=1000
np.random.seed(3)
e=np.random.randn(T)
y=np.zeros(len(e))

for i in range(len(e)):
    if i==0:
        y[i]=e[i]
    elif i==1:
        y[i]=0.5*y[i-1]+e[i]
    else:
        y[i]=0.5*y[i-1]+0.2*y[i-2]+e[i]

## b) Plot the y(t) with respect to number of samples.

plt.figure()
plt.plot(y)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('AR(2) y(t)-0.5*y(t-1)-0.2*y(t-2)=e(t)')
plt.show()



## c) Using the python code, developed in previous labs, calculate autocorrelations for 20 lags and plot them versus number of lags. Write down your observation about the ACF of above process.

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
x2=list()
for i in range(0,21):
    x2.append(i)
x1=[-1,-2,-3,-4,-5,-6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20]
x=x1[::-1]+x2

autocorrmean=autocorrelation(y)
autocorrmean=autocorrmean[:21]
autocorrmean1=autocorrmean[::-1]
autocorrmean=autocorrmean[1:]
autocorrmean=autocorrmean1+autocorrmean

plt.stem(x,autocorrmean, use_line_collection=True)
plt.title('Autocorrelation function for the residuals',pad=40)
plt.ylabel('Autocorrelation value')
plt.xlabel('Lag')
plt.figure()
plt.show()

## d) Display the first 5 values of y(t) at the console.

print("First 5 values of y(t)- for loop method",y[:5])

## e) Using various techniques learned in previous labs, Is this stationary process? Explain your answer.

print('The mean is :'+str(round(y.mean(),2))+' and the variance is :'+str(round(y.var(),2))+' with standard deviation :'+str(round(y.std(),2)))
means=[]
variances=[]
for i in range(0,T):
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

x = range(0, T)


plt.figure()
plt.plot(y,label='Y values for AR(2)')
plt.plot(x, Yvariances, marker="o", markersize=2,label='Y variances by subset', color="green", linestyle="solid", linewidth=2)
plt.plot(x, Ymeans, marker="x", markersize=2, color="purple",label="Y means by subset", linestyle="solid", linewidth=2)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('AR(2) y(t)-0.5*y(t-1)-0.2*y(t-2)=e(t)')
plt.legend()
plt.show()


from statsmodels.tsa.stattools import adfuller
ADFY= adfuller(y)
print('ADF Statistic for Y: %f' % ADFY[0])
print('p-value for Y: %f' % ADFY[1])
print('Critical Values for Y:')
for key, value in ADFY[4].items():
	print('\t%s: %.3f' % (key, value))



# 2. Using the “scipy libray” and “dlsim” command, simulate the AR(2) process in question 1.

system=([1,0,0],[1,-0.5,-0.2],1)
t_in=np.arange(0,T)
t_out,y_out= signal.dlsim(system,e)

## a) Display the first 5 values of y(t) at the console.
print("First 5 values of y(t)- for dlsimp method",y_out[:5])

## b) Show that your answer to the previous part is identical to the answer in part d of previous question.

difference=[]
for i in range(0,len(y)):
    difference.append(y[i] - y_out[i][0])

print('Difference',difference)

# 3- Write the AR(2) process in question 1, as multiple regression model and using the least square estimate (LSE), estimate the true parameters 𝑎1 and 𝑎2 (-0.5, -0.2). Display the estimated parameters values at the console.
# What is the affect of additional samples on the accuracy of the estimate? Justify the answer by running your code for 5000 and 10000 data samples.

y1 = np.array([y[2:]]).T
X = np.array([(-(y[1:len(y) - 1])), (-(y[:len(y) - 2]))]).T


def coeff(X, Y):
    '''The function returns the values of the multiple regression model coefficients'''
    coef= np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)),X.transpose()), Y)
    return coef

coef = coeff(X, y1)

print('The true parameters 𝑎1 and 𝑎2 are: ',coef)


# 4- Generalized your code in the previous question, such when the code runs it asks a user the following questions:
## a. Enter number of samples :
## b. Enter the order # of the AR process:
## c. Enter the corresponding parameters of AR process :
## Your code should simulate the AR process based on the entered information (a, b, c) and estimate the AR parameters accordingly. The estimated parameters must be close to the entered numbers in part c. Display the estimated parameters and the true values at the console.

order = int(input('Enter the order # of the AR process: '))
coefficients=list()
for i in range(order):
    coefficients.append(float(input('What is the coefficient '+str(i+1)+'?')))
samples=int(input("Enter number of samples:"))
#coefficients = [0.5, 0.2, -0.7]
np.random.seed(3)
e2 = np.random.randn(samples)
t = np.zeros(len(e2))

for i in range(order,len(e2)):
    if i == 0:
        t[i] = e2[i]
    else:
        sum = 0
        for k in range(order):
            sum += float(coefficients[k]) * t[i - (k+1)] * -1
            t[i] = sum + e2[i]

order_list = []
for i, j in zip(range(1, order + 1), range(order, 0, -1)):
    x_elements = t[j:len(t) - i]
    order_list.append(x_elements)


X = ((np.array(order_list)) * - 1).T
y1 = np.array([t[order + 1:]]).T

coef2 = coeff(X, y1)

print('The coefficients for the parameters specified are: ',coef2)

# 5- Let consider an MA(2) process as
# 𝑦(𝑡) = 𝑒(𝑡) + 0.5𝑒(𝑡 − 1) + 0.2𝑒(𝑡 − 2)
# Where 𝑒(𝑡) is a WN (0,1).


##a) Using a for loop simulate above process for 1000 samples. Assume all initial conditions to be zero.

T=1000
np.random.seed(3)
e=np.random.randn(T)
y=np.zeros(len(e))

for i in range(len(e)):
    if i==0:
        y[i]=e[i]
    elif i==1:
        y[i]=0.5*e[i-1]+e[i]
    else:
        y[i]=0.5*e[i-1]+0.2*e[i-2]+e[i]

## b) Plot the y(t) with respect to number of samples.

plt.figure()
plt.plot(y,color='orange')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('MA(2) y(t)-0.5*y(t-1)-0.2*y(t-2)=e(t)')
plt.show()



## c) Using the python code, developed in previous labs, calculate autocorrelations for 20 lags and plot them versus number of lags. Write down your observation about the ACF of above process.

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
x2=list()
for i in range(0,21):
    x2.append(i)
x1=[-1,-2,-3,-4,-5,-6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20]
x=x1[::-1]+x2

autocorrmean=autocorrelation(y)
autocorrmean=autocorrmean[:21]
autocorrmean1=autocorrmean[::-1]
autocorrmean=autocorrmean[1:]
autocorrmean=autocorrmean1+autocorrmean

plt.stem(x,autocorrmean, use_line_collection=True)
plt.title('Autocorrelation function for the MA(2)',pad=40)
plt.ylabel('Autocorrelation value')
plt.xlabel('Lag')
plt.figure()
plt.show()

## d) Display the first 5 values of y(t) at the console.

print("First 5 values of y(t)- for loop method for MA(2) process",y[:5])

## e) Using various techniques learned in previous labs, Is this stationary process? Explain your answer.

print('The mean is :'+str(round(y.mean(),2))+' and the variance is :'+str(round(y.var(),2))+' with standard deviation :'+str(round(y.std(),2)))
means=[]
variances=[]
for i in range(0,T):
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

x = range(0, T)


plt.figure()
plt.plot(y,label='Y values for MA(2)',color='orange')
plt.plot(x, Yvariances, marker="o", markersize=2,label='Y variances by subset', color="green", linestyle="solid", linewidth=2)
plt.plot(x, Ymeans, marker="x", markersize=2, color="purple",label="Y means by subset", linestyle="solid", linewidth=2)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('MA(2) y(t)=0.5*e(t-1)+0.2*e(t-2)+e(t)')
plt.legend()
plt.show()


from statsmodels.tsa.stattools import adfuller
ADFY= adfuller(y)
print('ADF Statistic for Y: %f' % ADFY[0])
print('p-value for Y: %f' % ADFY[1])
print('Critical Values for Y:')
for key, value in ADFY[4].items():
	print('\t%s: %.3f' % (key, value))



# 6. Using the “scipy libray” and “dlsim” command, simulate the MA(2) process in question 1.

system=([1,0.5,0.2],[1,0,0],1)
t_out,y_out= signal.dlsim(system,e)

## a) Display the first 5 values of y(t) at the console.
print("First 5 values of y(t)- for dlsimp method",y_out[:5])

## b) Show that your answer to the previous part is identical to the answer in part d of previous question.

difference=[]
for i in range(0,len(y)):
    difference.append(y[i] - y_out[i][0])

print('Difference',difference)