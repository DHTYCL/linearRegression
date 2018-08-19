#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def file(path):
    # pandas读入
    data = pd.read_csv('8.Advertising.csv')    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    return x,y


if __name__ == "__main__":
    path = '8.Advertising.csv'
    x,y=file(path)
    x = np.array(x)
    y = np.array(y)
    # # 绘制1
    # plt.plot(x[:,0], y, 'ro', label='TV')
    # plt.plot(x[:,1], y, 'g^', label='Radio')
    # plt.plot(x[:,2], y, 'mv', label='Newspaer')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()
    #
    # # 绘制2
    # plt.figure(figsize=(9,12))
    # plt.subplot(311)
    # plt.plot(x[:,0], y, 'ro')
    # plt.title('TV')
    # plt.grid()
    # plt.subplot(312)
    # plt.plot(x[:,1], y, 'g^')
    # plt.title('Radio')
    # plt.grid()
    # plt.subplot(313)
    # plt.plot(x[:,2], y, 'b*')
    # plt.title('Newspaper')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.5, random_state=1)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print (linreg.coef_)
    print (linreg.intercept_)

    y_hat = linreg.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print (mse, rmse)

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
