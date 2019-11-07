import numpy as np
import pandas as pd
# from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import signal
#

# Multivariate Modeling
# Homework # 8 ARMA Model- Theoretical mean-variance-autocorrelation DATS 6450
# Perform the following tasks:
# 1- Let consider an ARMA(1,1) process as
# 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) = 𝑒(𝑡) + 0.8𝑒(𝑡 − 1)
# Where 𝑒(𝑡) as a WN (2,1).

T2=100
np.random.seed(3)
Me=2
e2=1*np.random.randn(T2)+Me

def processarma11(e):
    y=np.zeros(len(e))

    for i in range(len(e)):
        if i==0:
            y[i]=e[i]
        elif i==1:
            y[i]=(0.5*y[i-1]+e[i]+0.8*e[i-1])
    return(y)

y=processarma11(e2)


plt.figure()
plt.plot(y,color='orange')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('ARMA(1,1) y(t)=0.5*y(t-1)+0.8*e(t-1)+e(t)')
plt.show()

#a. Calculate theoretical mean of above process. No need to use python.
# Show all your work in the report.
mean=np.mean(y)
print('The mean is',mean)
#b. Calculate the theoretical variance of above process. No need to use python.
# Show all your work in your report.
var=np.var(y)
print('The mean is',var)
#c. Calculate the first 3 theoretical ACF of y(t). No need to use python.
# Show all your work in the report.
def autocorrelation(y):
    '''It returns the autocorrelation'''
    t = len(y)
    k = range(0, t)
    meany = np.mean(y)
    tk = []
    for h in k:
        numerator = 0
        denominator = 0
        for ti in range(h, t):
            numerator += ((y[ti] - meany) * (y[ti - h] - meany))
        denominator = np.sum((y - meany) ** 2)
        tk.append(numerator / denominator)
    return tk

y_testvector=np.array([[1]])
for h in y:
       #print(i)
    y_testvector=np.append(y_testvector,[[h]],axis=0)
y_testvector=np.delete(y_testvector,0,axis=0)
    #y_testvector

res=y_testvector-y
    #res

print( " The ACF values of the residuals for this estimate are: ", autocorrelation(res)[:3])


#d. You may want to verify above results by writing a Python code but
# it is NOT required to include your code or send your code for submission.