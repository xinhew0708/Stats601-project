# -*- coding: utf-8 -*-
import os
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression


def np_ewma_vectorized(data, window, method = "WMS"):

    if method == "WMS":
        alpha = 1 / window
    elif method == "EMA":
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


def MACD(A):
    # Input: 1440 x 1 DataFrame
    # Output: two 1440 x 1 numpy array
    A = np.exp(A)
        
    # Get the 26-day EMA of the closing price
    k = np_ewma_vectorized(A, 26)
    # Get the 12-day EMA of the closing price
    d = np_ewma_vectorized(A, 12)

    # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
    macd = k - d

    #Get the 9-Day EMA of the MACD for the Trigger line
    macd_s = np.array(np_ewma_vectorized(macd, 9))

    # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
    #macd_h = macd - macd_s
    
    if abs(k[-1] - k[-2] - d[-1] + d[-2]) > 0.001:
        return ((k[-1] - k[-2] - d[-1] + d[-2]), k, d, np.array(macd))
    else:
        return (0, k, d, np.array(macd))


def RSI_np(A, window_length=14):
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

    avg_gain = np_ewma_vectorized(gain, window_length, 'WMS')#[-10:]
    avg_loss = np_ewma_vectorized(loss, window_length, 'WMS')#[-10:]
    
    rsi1 = 100 - (100 * avg_loss / (avg_loss + avg_gain))
        
    avg_gain = np_ewma_vectorized(gain, window_length, 'EMA')#[-10:]
    avg_loss = np_ewma_vectorized(loss, window_length, 'EMA')#[-10:]
    
    #rs = avg_gain / avg_loss
    rsi2 = 100 - (100 * avg_loss / (avg_loss + avg_gain))
    return rsi1[-2:], rsi2[-2:]


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

    A = np.hstack((A[0]*np.ones(90), A, A[-1]*np.ones(90)))
    
    # moving averages of 30 minutes
    N = 15
    pr_avg_0 = np.convolve(A[(90-N):-(91-N)], np.ones(2*N)/N/2, mode='valid')
    
    # 1 hour (60 minutes)
    N = 30
    pr_avg_1 = np.convolve(A[(90-N):-(91-N)], np.ones(2*N)/N/2, mode='valid')
    
    # 2 hour (120 minutes)
    N = 60
    pr_avg_2 = np.convolve(A[(90-N):-(91-N)], np.ones(2*N)/N/2, mode='valid')
    
    # 3 hours (180 minutes)
    N = 90
    pr_avg_3 = np.convolve(A[(90-N):-(91-N)], np.ones(2*N)/N/2, mode='valid')
    
    z0 = (A[-1] - pr_avg_0[-1]) / np.std(pr_avg_0)
    z1 = (A[-1] - pr_avg_1[-1]) / np.std(pr_avg_1)
    z2 = (A[-1] - pr_avg_2[-1]) / np.std(pr_avg_2)
    z3 = (A[-1] - pr_avg_3[-1]) / np.std(pr_avg_3)
    return np.array([z0, z1, z2, z3])

def neglogr(A, asset):
    params = [158, 279, 100, 15, 32, 7, 37, 6, 36, 82]
    return A[-1*params[asset]] - A[-1]


def get_features(A, B, asset):
    #m1, m2 = mov_avg(np.exp(A), B)
    indi, macd1, _, macd = MACD(A[::30])
    rsi1, rsi2 = RSI_np(A[::30])
    return np.hstack((
        0, #(m1[-1] - np.exp(A[-1])) / m1[-1], #m2,
        indi, 0, #(np.log(macd1[-1]) - A[-1]) / np.log(macd1[-1]), #macd[-1],
        rsi1, rsi2,
        volChanges(B)[2:],
        #getVolRatios(B),
        #priceVolCor(A, B),
        zScorePr(np.exp(A))[:2],
        neglogr(A, asset),
        0, #getTdist(A, asset),
        np.mean(A), np.std(A), (A[-1] - np.mean(A)) / np.std(A),
        1 #(np.mean(np.log10(B[-3:])) - volmean[asset]) / volstd[asset]
    )).reshape((1, -1))


# +
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
flist = [9,11,13]


# -

def get_r_hat(A, B): 
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9    
        return: a numpy array of length 10, corresponding to the predictions for 
        the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """
    answer = []
    for asset in range(0,10):
        f = get_features(np.array(A)[:, asset], np.array(B)[:, asset], asset)
        
        f[:, [7,8]] = f[:, [7,8]] / 10**8
        f = f[:, flist]

        pred = model.predict(f)

        answer.append(pred)
      
    answer = np.array(answer).reshape(10)
    return answer


'''
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
'''



