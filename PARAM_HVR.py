#!/usr/bin/env python
# coding: utf-8


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import numpy as np

# Remove outlier Quant_50% ±2*std values of 5-95% Quantile of RR (limits of agreement, LOA). 
# See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6111985/
# I choose quantile over mean and stv as high outlier (example 15K Milliseconds destroy results)
def artefact_correction(RR):
    quant = RR[RR.between(RR.quantile(.05),RR.quantile(.95))]
    return RR[RR.between(RR.quantile(.50) -2* quant.std(),RR.quantile(.50) +2* quant.std())]

def mean_rr(RR):
    return np.mean(RR)


def mean_hr(RR):
    # Changed into beats per minute, RR is in milliseconds
    return 60000 / mean_rr(RR)


def sdnn(RR):
    return np.std(RR, ddof=1)

def rmssd(RR):
    rr_diff = np.diff(RR)
    return np.sqrt(np.mean(rr_diff**2)) 


def sdsd(RR):
    rr_diff = np.diff(RR)
    return np.std(rr_diff)


def rmssd_sdsd(RR):
    rr_diff = np.diff(RR)
    rmssd = np.sqrt(np.mean(rr_diff**2))
    sdsd = np.sqrt(np.square(rmssd) - np.square(rr_diff.mean()))
    result = {'rmssd': rmssd, 'sdsd': sdsd}
    return result


def time_domain_analysis(RR):
    result = {'mean_rr': mean_rr(RR),'mean_hr':mean_hr(RR),'sdnn':sdnn(RR)}
    result.update(rmssd_sdsd(RR))
    return result



def poincare(RR,meta_info = False):
    # Prepare Poincaré data
    x1 = np.asarray(RR[:-1])
    x2 = np.asarray(RR[1:])

    # SD1 & SD2 Computation
    sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
    sd2 = np.std(np.add(x1, x2) / np.sqrt(2))

    # Area of ellipse
    area = np.pi * sd1 * sd2

    result = {'sd1': sd1,'sd2': sd2,'sd_quotient': sd1/sd2,'area':area}

    if meta_info:
        result['data'] = RR

    return result

def stress_index(RR=None, binsize=50, meta_info=False):

    bins = [*range(int(np.min(RR)), int(np.max(RR) + binsize), int(binsize))]

    D, bins = np.histogram(RR, bins, density=False)
    
    sumD = np.sum(D)
    if (sumD == 0):
        AMo = 0
    else:
        AMo = np.max(D)/sumD 
   
    Mo = np.median(RR)
    MxDMn = np.max(RR)-np.min(RR)

    si = (AMo*100)/(2*Mo/1000*MxDMn/1000)

    result = {'si': si}

    if meta_info:
        result['data'] = RR
        result['AMo'] = AMo
        result['Mo'] = Mo
        result['MxDMn'] = MxDMn
        result['bins'] = bins

    return result

def res_index(rmmssd):
    return np.log(rmmssd+1)

def pns_index(mean_rr,mean_rr_expact,mean_rr_sd,rmssd,rmssd_expact,rmssd_sd,sd1,sd1_expact,sd1_sd):

    mean_rr_score = (mean_rr - mean_rr_expact)/mean_rr_sd
    rmssd_score = (rmssd - rmssd_expact) / rmssd_sd
    sd1_score = (sd1 - sd1_expact) / sd1_sd

    return (mean_rr_score + rmssd_score + sd1_score)/3

def sns_index(mean_hr,mean_hr_expact,mean_hr_sd,stress_index,stress_index_expact,stress_index_sd,sd2,sd2_expact,sd2_sd):

    mean_hr_score = (mean_hr - mean_hr_expact)/mean_hr_sd
    stress_index_score = (stress_index - stress_index_expact) / stress_index_sd
    sd2_score = (sd2 - sd2_expact) / sd2_sd

    return (mean_hr_score + stress_index_score + sd2_score)/3

def iterative_sd(sd_old, n, mean_old, x_new):
    if(n<3):
        return 0
    n = n+1
    
    sd_neu_sq = (n-2) / (n-1) * (sd_old) ** 2 + (1 / n) * (x_new - mean_old) ** 2
    return np.sqrt(sd_neu_sq)

def iterative_mean(n, mean_old, x_new):
    if(n<2):
        return mean_old
    return (mean_old*n+ x_new)/(n+1)

from scipy.interpolate import UnivariateSpline
from scipy.signal import welch, periodogram
def frequency_domian(RR, method = 'fft'):
    rr_x = []
    pointer = 0
    for x in RR:
        pointer += x
        rr_x.append(pointer)

    rr_x_new = np.linspace(np.int(rr_x[0]), np.int(rr_x[-1]), np.int(rr_x[-1]))
    interpolated_func = UnivariateSpline(rr_x, RR, k=3)
    
    if method=='fft':
        datalen = len(rr_x_new)
        frq = np.fft.fftfreq(datalen, d=((1/1000.0)))
        frq = frq[range(np.int(datalen/2))]
        Y = np.fft.fft(interpolated_func(rr_x_new))/datalen
        Y = Y[range(np.int(datalen/2))]
        psd = Y**2
    elif method=='periodogram':
        frq, psd = periodogram(interpolated_func(rr_x_new), fs=1000.0)
    elif method=='welch':
        frq, psd = welch(interpolated_func(rr_x_new), fs=1000.0, nperseg=len(rr_x_new) - 1)
    else:
        raise ValueError("specified method incorrect, use 'fft', 'periodogram' or 'welch'")
    
    
    lf = np.trapz(x = frq[(frq >= 0.04) & (frq <= 0.15)], y = abs(psd[(frq >= 0.04) & (frq <= 0.15)]))
    hf = np.trapz(x = frq[(frq >= 0.16) & (frq <= 0.5)], y = abs(psd[(frq >= 0.16) & (frq <= 0.5)]))
    vlf = np.trapz(x = frq[(frq >= 0.00) & (frq <= 0.04)], y = abs(psd[(frq >= 0.00) & (frq <= 0.04)]))
    
    return {"vlf": vlf,"lf": lf, "hf":hf, "lf/hf":lf/hf, "hf/lf":hf/lf}

# Used for pNN50 and pNN20
def countPairsWithDiffK(RR, k):
    return (abs(np.diff(RR))>k).sum()/len(RR)



from scipy.stats import kurtosis, skew
def kurtose(RR):
    return kurtosis(RR)
def skewness(RR):
    return skew(RR)

# Relative RR can be used like RR. E.g. Mean, Median, SDRR, RMSDD, SDSD, KURT, SKEW etc.
def relativeRR(RR):
    return 2*(np.diff(RR)/np.sum([RR[1:], RR[:-1]], axis=0))


