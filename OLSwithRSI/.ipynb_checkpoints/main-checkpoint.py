# -*- coding: utf-8 -*-
import os
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
#from sklearn.linear_model import LinearRegression

def np_ewma_vectorized(data, window):

    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev**(n-1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def RSI_np(A, window_length=14, method="WMS"):
    """
    Calculate RSI
    A: numpy array of log price
    method : "SMA": simple moving average,
            "WMS": Wilder Smoothing Method,
            "EMA": exponential moving average
    
    Return RSI for last three periods
    """
    # transform log-price to price
    A = np.exp(A)
    tmp = np.diff(A)

    gain = np.clip(tmp, a_min = 0, a_max = None)
    loss = np.abs(np.clip(tmp, a_min = None, a_max = 0))

    if method == "WMS":
        avg_gain = np_ewma_vectorized(gain, window_length)#[-10:]
        avg_loss = np_ewma_vectorized(loss, window_length)#[-10:]
    else:
        avg_gain = np_ewma_vectorized(gain, window_length)#[-10:]
        avg_loss = np_ewma_vectorized(loss, window_length)#[-10:]
    
    #rs = avg_gain / avg_loss
    rsi = 100 - (100 * avg_loss / (avg_loss + avg_gain))
    return rsi[-1]-rsi[-2], rsi[-1]


def volChanges(B, window_length=30):
    vol0 = np.mean(B[-30:])
    vol1 = np.mean(B[-60:-30])
    vol2 = np.mean(B[-90:-60])
    vol3 = np.mean(B[-120:-90])
    
    return np.array([vol2-vol3, vol1-vol2, vol0-vol1, vol0])


# z-score of log-price
def zScorePr(A):
    # Input: 1440 x 1 numpy array
    # Output: 3 dim numpy array
    # the moving average log-price of 30min, 1h, and 3h
    
    # moving averages of 30 minutes
    pr_avg_0 = uniform_filter1d(A, size=30, mode='nearest')
    
    # 1 hour (60 minutes)
    pr_avg_1 = uniform_filter1d(A, size=60, mode='nearest')
    
    # 2 hour (120 minutes)
    pr_avg_2 = uniform_filter1d(A, size=120, mode='nearest')
    
    # 3 hours (180 minutes)
    pr_avg_3 = uniform_filter1d(A, size=180, mode='nearest')
    
    z0 = (A[-1] - pr_avg_0[-1]) / np.std(pr_avg_0)
    z1 = (A[-1] - pr_avg_1[-1]) / np.std(pr_avg_1)
    z2 = (A[-1] - pr_avg_2[-1]) / np.std(pr_avg_2)
    z3 = (A[-1] - pr_avg_3[-1]) / np.std(pr_avg_3)
    return np.array([z0, z1, z2, z3])

def neg30logr(A):
    return -(A[-1] - A[-30])


def get_features(A, B):
    #m1, m2 = mov_avg(A, B)
    #macd1, _, macd = MACD(A[::30])
    return np.hstack((
        #m1, m2,
        #macd1, macd[-1],
        RSI_np(A[::30])[-1],
        volChanges(B),
        #getVolRatios(B),
        #priceVolCor(A, B),
        zScorePr(A),
        neg30logr(A)
    )).reshape((1, -1))


MODELS = []

for i in range(0,10):
    with open('model_{}.pkl'.format(i), 'rb') as f:
        model = pickle.load(f)
        MODELS.append(model)

def get_r_hat(A, B): 
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9    
        return: a numpy array of length 10, corresponding to the predictions for 
        the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """
    answer = []
    for asset in range(0,10):
        f = get_features(np.array(A)[:, asset], np.array(B)[:, asset])

        pred = MODELS[asset].predict(f) - A.iloc[-1, asset]

        answer.append(pred)
      
    answer = np.array(answer).reshape(10)
    return answer


# +

log_pr = pd.read_pickle("./log_price.df")
volu = pd.read_pickle("./volume_usd.df")

# Generate r_hat every 10 minutes

t0 = time.time()
dt = datetime.timedelta(days=1)
dt1 = datetime.timedelta(days=1/1440)
r_hat = pd.DataFrame(index=log_pr.index[1440::1440], columns=np.arange(10), dtype=np.float64)
for t in log_pr.index[1440::1440]: # compute the predictions every 10 minutes
    r_hat.loc[[t], :] = get_r_hat(log_pr.loc[(t - dt + dt1):t], volu.loc[(t - dt + dt1):t])
t_used = time.time() - t0
print(t_used)

# -



