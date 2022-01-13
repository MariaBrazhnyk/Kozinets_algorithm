import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import pandas as pd

def add_vector(beta, X, y, leng):
    sigma = np.array([[beta[3], beta[4]], [beta[5], beta[6]]])
    v, vectors = np.linalg.eig(sigma)
    eigen = []
    
    for vector in vectors.T:
        eigen.append([0,0,0, vector[0]**2, vector[1]*vector[0], vector[1]*vector[0], vector[1]**2])
    if len(X) > leng:
        X = X.iloc[:leng,:]
        y = y[:leng]

    X = X.append(pd.DataFrame(eigen, columns=X.columns), ignore_index=True)
    y = np.append(y, np.ones(2))
    return X, y
    

def classificator(beta, X, y):
    leng = len(X)
    scal = y*np.dot(X, beta)
    if (scal <= 0).any():
        correct = False
        l = X[scal <= 0].index[0]
    
    gamma = 0
        
    while not correct:
        beta = gamma*beta + (1-gamma)*y[l]*X.loc[l, :]

        X, y = add_vector(beta, X, y, leng)
        
        scal = y*np.dot(X, beta)
        if (scal<=0).any():
            correct = False
            l = X[scal <= 0].index[0]
        else:
            correct = True
            
        gamma = (np.dot(X.loc[l], X.loc[l]) - y[l]*np.dot(X.loc[l], beta))/np.dot(beta-y[l]*X.loc[l], beta-y[l]*X.loc[l])
            
    print(beta)
    return beta

def for_test(df, beta):
    the_test = pd.DataFrame({'C': 1, 'X': df.X, 'Y': df.Y, 'X^2': df.X*df.X, 'X*Y': df.X*df.Y, 'Y*X': df.Y*df.X, 'Y^2': df.Y*df.Y})

    values = []
    scal = np.dot(the_test, beta)
    for i in range (len(scal)):
        if scal[i] > 0:
            values.append(1)
        else:
            values.append(-1)

    the_test['Predicted'] = values
    return the_test

def plot(df, y, beta):
    list_in = []
    list_out = []
    for i in y.index:
        if y[i] == -1:
            list_in.append(df.loc[i])
        else:
            list_out.append(df.loc[i])

    list_in = np.array(list_in).T
    list_out = np.array(list_out).T
    
    plt.scatter(list_in[0], list_in[1], color = 'red')
    plt.scatter(list_out[0], list_out[1], color = 'blue')
    xmax, xmin = np.max(df.X), np.min(df.X)
    ymax, ymin = np.max(df.Y), np.min(df.Y)
    x = np.linspace(xmin-0.1,xmax+0.1,50)
    y1 = np.linspace(ymin-0.1,ymax+0.1,50)
    X1, Y1 = np.meshgrid(x,y1)
    Z = np.zeros((50,50))
    for i, m in enumerate([1, X1, Y1, X1*X1, X1*Y1, Y1*X1, Y1*Y1]):
        Z+=beta[i]*m
    plt.contour(X1, Y1, Z, [0])