import numpy as np
from numpy import linalg as LA
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
import math

# DATS 6450
# Homework # 6- Fitting nonlinear dataset into an Autoregressive Model
# In this homework, you will use the tute1.csv data set. An autoregressive model will be used to fit the “Sales” data, the dependent variable. The coefficients of the AR model will be estimated using the least square estimate from the following equations:

# Hint: The only challenge here is to know what the order of the AR model is. Since the GPAC concept has not covered yet, you can start the AR order from order 1 and increase it to 5.
# 1- Using Pandas library load the “Sales” column of the time series data called “tute1”.


tute1=pd.read_csv('tute1.csv')
tute1.head()
tute1.info()
y=tute1['Sales']
y=y.values
print('The number of observations for sale is: ',len(y))



#2- Pick the order of AR to be 1. Estimate the corresponding coefficient(s).

samples=len(y)
arorders=6
Results=pd.DataFrame(np.zeros((arorders-1,8)))
Results.columns=['Run #','Order#','Coefficients','MSE','Q','R','Mean_residuals','Variance_residuals']

for ars in range(1,arorders):
    Results.iloc[ars-1, 0] = ars
    Results.iloc[ars-1, 1] = ars

    def calculate_X(order, y):
        X = []
        for a, b in zip(range(1, order + 1), range(order, 0, -1)):
            X.append(y[b:(len(y) - a)])
        return (np.array(X) * -1)


    def calculate_coefficients(order, y):
        Y = np.array(y[order + 1:len(y)])
        X = (calculate_X(order, y)).T
        a = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
        return a

    coefficients=0
    coefficients = calculate_coefficients(ars,y)
    Results.iloc[ars-1, 2] = str(coefficients)
    print("\nThe coefficients for order", ars,"are: ", coefficients)

    e = np.random.randn(samples)
    t = np.zeros(len(e))

    for i in range(1,len(e)):
        sum = 0
        for k in range(0,len(coefficients)):
            sum += float(coefficients[k]) * y[i - (k + 1)] *-1
            t[i] = sum + e[i]
    y_predd=t

    res=y-y_predd
# Calculate the Mean Square Error (MSE) and display it with an appropriate message on the
# console. MSE can be calculated as follows:
# MSE = np.sqrt(np.mean(e**2))

    MSE = np.sqrt(np.mean(res ** 2))
    print("Run #", ars, "- The mean squared error for this estimate is: ", MSE)
    Results.iloc[ars-1, 3] = MSE

#5- Calculate Q and display it with an appropriate message on the console.

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

    r=autocorrelation(res)
    #r=r[1]
    r2=[]
    for h in r[1:]:
        r2.append(h**2)
    #print(r)
    print(r2)
    Q=[]

    for h in range(1,len(r2)+1):
        r22=r2[:h]
        T=len(r22)
        Q.append(T*math.fsum(r22))
    print("Run #",ars,"- The Q value for this estimate is = ",Q[-1])
    Results.iloc[ars-1,4]=Q[-1]

    #6- Plot the AFC of residuals.

    autocorr=autocorrelation(res)
    autocorr = autocorr[:20]
    autocorr1=autocorr[::-1]
    autocorr=autocorr[1:]
    autocorr=autocorr1+autocorr
    x1 = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19]
    x2=list()
    for h in range(0,len(x1)+1):
        x2.append(h)
    x=x1[::-1]+x2
    plt.stem(x,autocorr, use_line_collection=True)
    plt.title("Run #"+str(ars)+"- Autocorrelation function for the residuals of the Mean Forecasting Method",pad=40)
    plt.ylabel('Autocorrelation value')
    plt.xlabel('Lag')
    plt.figure()
    plt.show()

    #7 - Plot the histogram plot of the residuals.
    plt.hist(res)
    plt.title("Run #" + str(ars) + "- Residuals Distribution", pad=40)
    plt.xlabel('Residual value')
    plt.ylabel('Number of Observations')
    plt.figure()
    plt.show()

    #8- Calculate R-square value and display it with an appropriate message on the console.
    def correlation_coefficent_cal(x, y):
        '''It returns the correlation coefficient for two given datasets'''
        meanx = np.mean(x)
        meany = np.mean(y)
        numerator = np.sum((x - meanx) * (y - meany))
        denominator1 = np.sum((x - meanx) ** 2)
        denominator2 = np.sum((y - meany) ** 2)
        r = numerator / (np.sqrt(denominator1) * np.sqrt(denominator2))
        return r


    R2 = np.square(correlation_coefficent_cal(y, y_predd))
    print("Run #",ars,"- The R2 is:",R2)
    Results.iloc[ars-1, 5] = R2

#9- Calculate the mean of residuals and display it with an appropriate message on the console.

    meanres=np.mean(res)
    print("Run #",ars,"- The mean of the residuals for this estimate is: ",round(meanres,3))
    Results.iloc[ars-1, 6] = round(meanres,3)

# 10 - Calculate the variance of residuals and display it with an appropriate message on the console.
    varres=np.var(res)
    print("Run #",ars,"- The variance of the residuals for this estimate is: ",round(varres,3))
    Results.iloc[ars-1, 7] = varres


# 11 - Plot the estimated values versus the true values with respect to time.

    plt.scatter(range(1,len(y)+1), y, label='True values', marker="*", s=12, color="purple")
    plt.scatter(range(1,len(y)+1), y_predd, label='Predicted values', marker="o", s=7, color="green")
    plt.title("Run #" + str(ars) + "- Sales true vs predicted values", pad=20)
    plt.ylabel("Sales values")
    plt.xlabel("Lags")
    plt.legend()
    ax = plt.gca()
    ax.grid(which='major', axis='y', linestyle='-')
    plt.show()

# 12 - Change the order to 2, 3, 4 and 5 and repeat the step 2 through 11.
# 13 - Create a table and put all this information for different order number inside a table.
print(Results.iloc[:,:4])
print(Results.iloc[:,4:])
# 14 - Pick the best AR model order which represents the “Sales” dataset the best.
# You need to justify why the picked order number makes sense.
# 15 - Write the final AR model with the best order number and the corresponding coefficients as the
# model that best represents this dataset.