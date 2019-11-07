
import numpy as np
from numpy import linalg as LA
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams["figure.figsize"] = (30, 15)
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 24


# Homework # 5- Linear Regression Predictor and Least Square Estimate
# The tute1.csv data set will be used for this homework. Let the AdBudget and GDP be independent variables (regressors) and the Sales amount be the dependent variable (regressand). We would like to find a best linear regression model to fit the tute1 dataset and predict the Sales value by knowing the AdBudget and GDP
# The linear regression model is:
# 𝑦𝑡 = 𝛽0 + 𝛽1𝑥1,𝑡 + 𝛽2𝑥2,𝑡 + 𝜀𝑡
# where 𝑦𝑡 is the sales values, 𝑥1,𝑡, 𝑥2,𝑡 are AdBudget and GDP and they are all known and 𝜀𝑡 has zero 2
# 𝑇 −1 𝑇 𝛽=(𝑋𝑋) 𝑋𝑦
# where X and Y are given as :
# Matrix X has T rows and (k+1) columns where T is the number of samples and k is number of independent variable ( k=2).
# The format of the 𝑋and 𝑦 is given in page 141 of the course textbook.

# 1- Using Pandas library load the time series data called tute1 from the BB.
tute1=pd.read_csv('tute1.csv')
tute1.head()
tute1.info()


S=1

Results=pd.DataFrame(np.zeros((S,9)))
Results.columns=['Run #','𝛽0','𝛽1','𝛽2','SSE','Meanres','Stderror','R2','Q']


def correlation_coefficent_cal(x, y):
    '''It returns the correlation coefficient for two given datasets'''
    meanx = np.mean(x)
    meany = np.mean(y)
    numerator = np.sum((x - meanx) * (y - meany))
    denominator1 = np.sum((x - meanx) ** 2)
    denominator2 = np.sum((y - meany) ** 2)
    r = numerator / (np.sqrt(denominator1) * np.sqrt(denominator2))
    return r

for i in range(0,S):
    Results.iloc[i,0]=i

# 2 - Split the dataset into training set and test set.Use 80 % for training and 20 % for testing.Display the training set and testing set array as follow:
#     The training set for AdBudget(x1 - train) amount is: ________________
#     The training set for GDP(x2 - train) amount is: ________________
#     The  training set for Sales(y - train) amount is: ________________
#     The testing set for AdBudget(x1 - test) amount is: ________________
#     The testing set for GDP(x2 - test) amount is: ________________
#     The testing set for Sales(y - test) amount is: ________________
# Hint: This can be done using the following library in python.
# “from sklearn.model_selection import train_test_split”


    print ('Results for run #',i)
    X=tute1[['AdBudget','GDP']]
    y=tute1['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=i,shuffle=False)

    print("Run #",i,"- The training set for AdBudget(x1 - train) amount is: ",len(X_train.iloc[:,0]))
    print("Run #", i, "- The training set for GDP(x2 - train) amount is: ", len(X_train.iloc[:,1]))
    print("Run #",i,"- The training set for Sales(y-train) amount is: ",len(y_train))
    print("Run #",i,"- The testing set for AdBudget(x1 - test) amount is: ",len(X_test.iloc[:,0]))
    print("Run #", i, "- The testing set for GDP(x2 - test) amount is: ",len( X_test.iloc[:,1]))
    print("Run #",i,"- The testing set for Sales(y-test) amount is: ",len( y_test))


# 3- Plot the scatter plot of the Sales versus AdBudget and Sales versus GDP and calculate the correlation coefficient between them. Display the correlation coefficient on the title as it was done in previous labs. Do you think that the linear regression model could be a good estimator for this dataset? Justify your answer.


    cor_X1y = correlation_coefficent_cal(X['AdBudget'], y)

    plt.scatter(X['AdBudget'], y, color="purple", s=300)
    plt.title("Scatter plot of AdBudget(X1) and Sales(Y) with r ={}".format(round(cor_X1y, 3)), pad=30)
    plt.ylabel("Sales")
    plt.xlabel("AdBudget")
    plt.grid()
    plt.show()

    cor_X2y = correlation_coefficent_cal(X['GDP'], y)

    plt.scatter(X['GDP'], y, color="purple", s=300)
    plt.title("Scatter plot of GDP(X2) and Sales(Y) with r ={}".format(round(cor_X2y, 3)), pad=30)
    plt.ylabel("Sales")
    plt.xlabel("GDP")
    plt.grid()
    plt.show()

# 4- Construct matrix X and Y using x-train and y-train dataset and estimate the regression model coefficients 𝛽0, 𝛽1, 𝛽2 using LSE method (above equation). Display the following message :
# ̂
# 𝛽0 for this linear regression model is: _______
# ̂
# 𝛽1 for this linear regression model is: _______
# ̂
# 𝛽2 for this linear regression model is: _______

    Xmatrix=X_train.values
    Xmatrix=np.insert(Xmatrix,0,np.ones((1,len(X_train['AdBudget']))),1)

    #print(Xmatrix)

    Yvector=np.array([[1]])
    for h in y_train:
        #print(h)
        Yvector=np.append(Yvector,[[h]],axis=0)
    Yvector=np.delete(Yvector,0,axis=0)
    #Yvector
    #print(Xmatrix)
    Xtrans=Xmatrix.transpose()
    XtXinv=np.linalg.inv(np.dot(Xtrans,Xmatrix))
    B=np.dot(np.dot(XtXinv,Xtrans),Yvector)
    # print('This is B:',B)
    # print('This is B size:', B.shape)
    print("\nRun #",i,"- 𝛽0 for this linear regression model is: ",B[0])
    print("\nRun #",i,"- 𝛽1 for this linear regression model is: ",B[1])
    print("\nRun #", i,"- 𝛽2 for this linear regression model is: ", B[2])
    Results.iloc[i,1]=B[0]
    Results.iloc[i,2]=B[1]
    Results.iloc[i,3]=B[2]

    # # 5- To test the accuracy of the derived model, using the testing set ( x-test and y-test) and by knowing 𝛽̂0 and 𝛽̂1 calculate the estimated tip (𝑦̂𝑡) amount and plot the true tip and estimated tip amount in one graph. The x axis is bill amount(x-test). Add an appropriate legend and title to your plot. Write down your observation on this plot.

    Xmatrixtest = X_test.values
    Xmatrixtest = np.insert(Xmatrixtest, 0, np.ones((1, len(X_test['AdBudget']))), 1)
    #print(Xmatrixtest)

    Y_pred=np.dot(Xmatrixtest,B)
    # print(Y_pred.shape)
    # print(y_test.shape)
    # print(Xmatrixtest.T[1].shape)

    plt.scatter(Xmatrixtest.T[1], Y_pred, label='AdBudget vs predicted sales', marker="*", s=2000, color="purple")
    plt.scatter(Xmatrixtest.T[1], y_test, label='AdBudget vs true sales', marker="o", s=700, color="green")
    plt.title("Run #" + str(i) + "- Sales prediction for AdBudget using linear regression", pad=20)
    plt.ylabel("Sales")
    plt.xlabel("AdBudget")
    plt.legend()
    ax = plt.gca()
    ax.grid(which='major', axis='y', linestyle='-')
    plt.show()

    print("Run #", i, "- The predicted values of sales for the AdBudgets: ", str(X_test.values.T[0]), 'are:', str(Y_pred.T))

    plt.scatter(Xmatrixtest.T[2], Y_pred, label='GDP vs predicted sales', marker="*", s=2000, color="purple")
    plt.scatter(Xmatrixtest.T[2], y_test, label='GDP vs true sales', marker="o", s=700, color="green")
    plt.title("Run #" + str(i) + "- Sales prediction for GDP using linear regression", pad=20)
    plt.ylabel("Sales")
    plt.xlabel("GDP")
    plt.legend()
    ax = plt.gca()
    ax.grid(which='major', axis='y', linestyle='-')
    plt.show()

    print("Run #", i, "- The predicted values of sales for the GDP: ", str(X_test.values.T[1]), 'are:', str(Y_pred.T))

  # # 6- Calculate the residuals/error for this prediction. The residuals are calculated by subtracting the estimated tip (derived in previous step) from the true tip ( y-test).

    y_testvector=np.array([[1]])
    for h in y_test:
        #print(i)
        y_testvector=np.append(y_testvector,[[h]],axis=0)
    y_testvector=np.delete(y_testvector,0,axis=0)
    #y_testvector

    res=y_testvector-Y_pred
    #print(res)

    # # 7- Calculate the sum square error (SSE) of residuals (calculated in the previous step) and display the message:
    # np.sqrt(np.mean(e**2))
    # The SSE for the predictions is: ________

    # sse=0
    # SSE=[]
    # for j in res:
    #     #print(sse)
    #     sse=sse+(j**2)
    #     SSE.append(sse)
    MSE = np.sqrt(np.mean(res ** 2))
    print("Run #", i, "- The mean squared error for this estimate is: ", MSE)
    Results.iloc[i, 4] = MSE

 # # 8- Calculate the mean of the residuals and display as a message. Is this a bias estimator or an unbiased estimator? Justify your answer.

    meanres=np.mean(res)
    print("\nRun #",i,"- The mean of the residuals for this estimate is: ",round(meanres,3))
    Results.iloc[i, 5] = round(meanres,3)

    # # 9- Estimate the standard error for this predictor using the following equation and display the value as a message. What is your observation about the standard error?

    T=len(X_test)
    k=1
    c=1/(T-k-1)
    e=[]
    for h in res:
        d=h*h
        e.append(d)
    stderror=np.sqrt(c*sum(e))
    #stderror
    print("\nRun #",i,"- The standard error for this predictor is: ",stderror)
    Results.iloc[i, 6] = stderror

 # # 10- Calculate the 𝑅2 using the following equation. Write down your observation about 𝑅2 and justify the accuracy of the predictor using 𝑅2 value.


    # num=[]
    # den=[]
    # Ymean=np.mean(y_test)
    # for h in Y_pred:
    #     num.append(np.square((h-Ymean)))
    #
    # for h in y_testvector:
    #     den.append(np.square(h-Ymean))
    # num=sum(num)
    # den=sum(den)
    # R2=num/den

    R2 = np.square(correlation_coefficent_cal(y_testvector, Y_pred))
    print("\nRun #",i,"- The R2 is:",R2)
    Results.iloc[i, 7] = R2


    # # 11- Graph the ACF of the residuals and justify the predictor accuracy using AFC.


    def autocorrelation(y):
        '''It returns the autocorrelation'''
        t=len(y)
        k=range(0,t)
        meany=np.mean(y)
        tk=[]
        for h in k:
            numerator=0
            denominator=0
            for ti in range(h,t):
                numerator+=((y[ti]-meany)*(y[ti-h]-meany))
            denominator=np.sum((y-meany)**2)
            tk.append(numerator/denominator)
        return tk

    print("Run #",i,"- The ACF values of the residuals for this estimate are: ",autocorrelation(res))


    autocorr=res
    autocorr=autocorrelation(autocorr)
    autocorr1=autocorr[::-1]
    autocorr=autocorr[1:]
    autocorr=autocorr1+autocorr
    x2=list()
    for h in range(0,len(X_test)):
        x2.append(h)
    x1=[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19]
    x=x1[::-1]+x2
    plt.stem(x,autocorr, use_line_collection=True)
    plt.title("Run #"+str(i)+"- Autocorrelation function for the residuals of the Mean Forecasting Method",pad=40)
    plt.ylabel('Autocorrelation value')
    plt.xlabel('Lag')
    plt.figure()
    plt.show()


    # # 12- Plot the scatter plot between residuals and predictor variable and display the correlation coefficient between them on the title. Are they related? Justify the accuracy of this predictor by observing the correlation coefficient between them.


    cor_resX1y=correlation_coefficent_cal(res,Xmatrixtest.T[1])
    plt.scatter(Xmatrixtest.T[1],res,marker="*",s=2000,color="purple")
    plt.title("Run #"+str(i)+"- Residuals by AdBudget for the linear regression model with r ={}".format(cor_resX1y),pad=20)
    plt.ylabel("Residuals")
    plt.xlabel("AdBudget")
    ax = plt.gca()
    ax.grid(which='major', axis='y', linestyle='-')
    plt.figure()
    plt.show()

    cor_resX2y = correlation_coefficent_cal(res, Xmatrixtest.T[2])
    plt.scatter(Xmatrixtest.T[2], res, marker="*", s=2000, color="purple")
    plt.title("Run #" + str(i) + "- Residuals by GDP for the linear regression model with r ={}".format(cor_resX2y), pad=20)
    plt.ylabel("Residuals")
    plt.xlabel("GDP")
    ax = plt.gca()
    ax.grid(which='major', axis='y', linestyle='-')
    plt.figure()
    plt.show()

    # # 13- Plot the histogram for the residuals. Is it a normal distribution? Justify your answer.

    plt.hist(res)
    plt.title("Run #"+str(i)+"- Residuals Distribution",pad=40)
    plt.xlabel('Residual value')
    plt.ylabel('Number of Observations')
    plt.figure()
    plt.show()

    # # 14- Plot the scatter plot between y-test and 𝑦̂𝑡 and display the correlation coefficient between them on the title. Justify the accuracy of this predictor by observing the correlation coefficient between y-test and 𝑦̂𝑡.


    cor_YtYp=correlation_coefficent_cal(y_testvector,Y_pred)
    plt.scatter(y_testvector,Y_pred,marker="*",s=2000,color="purple")
    plt.title("Run #"+str(i)+"- Y-test by Y-predicted for the linear regression model with r ={}".format(round(cor_YtYp,3)),pad=20)
    plt.ylabel("Predicted Y")
    plt.xlabel("True Y")
    ax = plt.gca()
    ax.grid(which='major', axis='y', linestyle='-')
    plt.figure()
    plt.show()

    # # 15- Find a 95% prediction interval for this predictor using the following equation:


    X_testtrans= Xmatrixtest.transpose()
    XtXtestinv=np.linalg.inv(np.dot(X_testtrans,Xmatrixtest))
    yintpredmin=[]
    yintpredmax=[]
    for z in range(0,len(Xmatrixtest)):
        X__=Xmatrixtest[z]
        k=np.dot(np.dot(X__,XtXtestinv),X__.T)
        k=np.sqrt(1+k)
        k=1.96*stderror*k
        yintpredmin.append(Y_pred[z]-k)
        yintpredmax.append(Y_pred[z]+k)
        print("Run #",i,"- The 95% prediction interval for "+str(Y_pred[z])+' is between '+str(Y_pred[z]-k)+' and '+str(Y_pred[z]+k))



    # # 16- Calculate the Q value using the following equation:
    #
    # Where rk is the autocorrelation and h is the maximum lags. Hint: For this case with low number of data
    # samples, let h be the length of y-test.


    # As k=1 and h=1 then
    r=autocorrelation(res)
    #r=r[1]
    r2=[]
    for h in r[1:]:
        r2.append(h**2)
    #print(r)
    #print(r2)
    Q=[]

    for h in range(1,len(r2)+1):
        r22=r2[:h]
        T=len(r22)
        Q.append(T*sum(r22))
    print("\nRun #",i,"- The Q value for this estimate is = ",Q[-1])
    Results.iloc[i,8]=Q[-1]

    # # 17- You need to run your code for multiple times. Every time you run the code, you will get a different intercept and slope and consequently different predictor. Pick the predictor which has the best performance. Justify your answer why the picked predictor is the best using the statistics derived in above steps.

print(Results)




