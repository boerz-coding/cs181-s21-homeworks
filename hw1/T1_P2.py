#####################
# CS 181, Spring 2021
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c



# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']

X = X_df.values
y = y_df.values

print("y is:")
print(y)

W = np.array([[1., 0.], [0., 1.]])

def predict_kernel(alpha=0.1):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    # TODO: your code here
    W1=W*alpha
    N=np.shape(X)[0]
    y_df=np.zeros(N)
    for i in range(N):
        kernelup=0
        kerneldown=0
        for j in range(N):
            if(j!=i):
                kernelup+=np.exp(-((X[j][0]-X[i][0])**2*W1[0][0]+2*(X[j][0]-X[i][0])*(X[j][1]-X[i][1])*W1[0][1]+(X[j][1]-X[i][1])**2*W1[1][1]))*(y[j])
                kerneldown+=np.exp(-((X[j][0]-X[i][0])**2*W1[0][0]+2*(X[j][0]-X[i][0])*(X[j][1]-X[i][1])*W1[0][1]+(X[j][1]-X[i][1])**2*W1[1][1]))
        y_df[i]=kernelup/kerneldown
    return y_df

def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""
    # TODO: your code here
    W1=W
    N=np.shape(X)[0]
    dX=np.zeros(N)
    y_df=np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            if(j!=i):
                dX[j]=(X[j][0]-X[i][0])**2*W1[0][0]+2*(X[j][0]-X[i][0])*(X[j][1]-X[i][1])*W1[0][1]+(X[j][1]-X[i][1])**2*W1[1][1]
        dX[i]=9999999
        kidarray=np.argsort(dX)
        sumk=0
        for j in range(k):
            sumk+=y[kidarray[j]]
        y_df[i]=sumk/k

    return y_df

def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 

    # Saving the image to a file, and showing it as well
    plt.savefig('alpha' + str(alpha) + '.png')
    plt.show()

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 
    # Saving the image to a file, and showing it as well
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for alpha in (0.1, 3, 10):
    # TODO: Print the loss for each chart.
    plot_kernel_preds(alpha)

for k in (1, 5, len(X)-1):
    # TODO: Print the loss for each chart.
    plot_knn_preds(k)
