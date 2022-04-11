import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import xgboost as xgb
import signal


def randomtimesignal(n, t, plot):
    def sinusoidal():
        # random sinusoidal
        a = np.random.uniform(0, 10, 1)[0]
        f = np.random.uniform(1e9, 1e10, 1)[0]
        ph = np.random.uniform(-2*np.pi, 2*np.pi, 1)[0]
        return a*np.sin(2*np.pi*f*t + ph*f)
    i = 0
    tot = 0
    while i < n:
        tot = tot + sinusoidal()
        i += 1
    if plot:
        plt.plot(t, tot)
        plt.xlabel("Time t")
        plt.ylabel("Signal e(t)")
        plt.show()
    return tot

def rts(t):
    a = np.random.uniform(0, 10, 1)[0]
    f = np.random.uniform(1e9, 1e10, 1)[0]
    ph = np.random.uniform(-2 * np.pi, 2 * np.pi, 1)[0]
    i = 0
    n = 1000
    tot = 0
    while i < n:
        tot = tot + a*np.sin(2*np.pi*f*t + ph*f)
        i += 1
    return tot


def ffts(tau, t, plot, n_samp):
    sin = randomtimesignal(10, t, plot)
    yf = fft(sin)
    xf = fftfreq(n_samp, tau)[:n_samp//10]
    if plot:
        plt.plot(xf, 2.0/n_samp * np.abs(yf[0:n_samp//10]))
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.show()
    return sin, yf, signal.find_peaks(np.abs(yf[0:n_samp//10]))[0]/10


def comparison(e_dec_t, e_t, p, t):
    k = int(len(e_t)*p)
    plt.plot(t[:k], e_dec_t[:k])
    plt.xlabel("Time t")
    plt.ylabel("Signal e(t) deconvolution")
    plt.show()
    plt.plot(t[:k], e_t[:k])
    plt.xlabel("Time t")
    plt.ylabel("Signal e(t) original")
    plt.show()
    plt.plot(t[:k], 100*abs(e_dec_t[:k] - e_t[:k])/(e_t[:k]))
    plt.xlabel("Time t")
    plt.ylabel("Percentage % error")
    plt.show()


def stat_analysis(n, func, tau, t, n_samp):
    i = 0
    ks = []
    stats1 = []
    while i < n:
        e_t, e_f, f1 = ffts(tau, t, False, n_samp)
        h_t, h_f, f2 = ffts(tau, t, False, n_samp)
        r_t = signal.convolve(e_t, h_t)
        e_dec_t, remainder = func(r_t, h_t)
        errors = 100*abs(e_dec_t - e_t)/e_t
        ks.append(np.argmax(errors > 1))
        length_min = np.min([len(f1), len(f2)])
        pearson = stats.pearsonr(f1[:length_min-1], f2[:length_min-1])[0]
        eucl = np.linalg.norm(f1[:length_min-1]-f2[:length_min-1])
        stats1.append([np.var(f1), np.var(f2), np.max(f1), np.max(f2), np.min(f1), np.min(f2), 10-len(f1),
                       10-len(f2), pearson, eucl])
        i += 1
    reg = xgb.XGBRegressor()
    reg.fit(stats1, ks)
    reg.get_booster().feature_names = ["Variance of e(t)", "Variance of h(t)", "Maximum value of e(t)",
                                       "Maximum value of h(t)", "Minimum value of e(t)", "Minimum value of h(t)",
                                       "Number of indistinct peaks of h(t)", "Number of indistinct peaks of e(t)",
                                       "Pearson Correlation", "Euclidean Distance"]
    xgb.plot_importance(reg)
    return ks, stats1
