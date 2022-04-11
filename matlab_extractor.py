import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import ifft
import signal_feature_extraction as sfe
import antropy as ant
import sklearn.metrics as sm
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")


def extract(file, bool):
    content = os.listdir(file)
    ln = len(content)
    xs = []
    ys = []
    for i in range(0,ln):
        cood = content[i][5:-4]
        dash = cood.index("_")
        xs.append(float(cood[:dash]))
        ys.append(float(cood[dash + 1:]))
    xs, ys = list(set(xs)),list(set(ys))
    xs.sort(), ys.sort()
    if bool == True:
        plt.plot(xs,ys,"o", label = "Emitters")
        plt.plot([1.2],[1.3],"o", label = "Receiver")
        plt.xticks(np.array([np.min(xs)-0.5, np.min(xs), np.mean(xs), np.max(xs), np.max(xs)+0.5]))
        plt.yticks(np.array([np.min(ys)-0.5, np.min(ys), np.mean(ys), np.max(ys), np.max(ys)+0.5]))
        plt.xlim([np.min(xs)-0.4,np.max(xs)+0.8])
        plt.ylim(([np.min(ys)-0.4,np.max(ys)+0.8]))
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.title("H_f map")
        plt.legend()
        plt.show()

    return xs,ys


data_lin = "C:/Users/16098/Documents/Research Project/MATLAB/Data_l_10_one_side"
data_chaotic = "C:/Users/16098/Documents/Research Project/MATLAB/Data_l_10_chaotic"

def opener(x,y,data):
    txt_H_f = data + "/data_" + str(x) + "_" + str(y) + ".txt"
    text_file = open(txt_H_f, "r")
    lines = text_file.read().split('\n')
    H_f = ([x.replace("i", "j") for x in lines])
    H_f = H_f[:-1]
    H_f = ([complex(x) for x in H_f])
    return H_f

def corr_abs(H_1, H_2):
    return abs(np.corrcoef(ifft(H_1), ifft(H_2))[0][1])
def corr_real(H_1, H_2):
    return np.real(np.corrcoef(ifft(H_1), ifft(H_2))[0][1])
def corr_imag(H_1, H_2):
    return np.imag(np.corrcoef(ifft(H_1), ifft(H_2))[0][1])
def ml(H_1,H_2):
    data = np.real(ifft(H_1))
    fs_1 = np.array([ant.petrosian_fd(data), ant.katz_fd(data), ant.higuchi_fd(data), sfe.sevcik(data)])
    data = np.real(ifft(H_2))
    fs_2 = np.array([ant.petrosian_fd(data), ant.katz_fd(data), ant.higuchi_fd(data), sfe.sevcik(data)])
    eucl = np.linalg.norm(fs_1 - fs_2)
    return eucl
def petrosian(H_1,H_2):
    data = np.real(ifft(H_1))
    fs_1 = np.array([ant.petrosian_fd(data)])
    data = np.real(ifft(H_2))
    fs_2 = np.array([ant.petrosian_fd(data)])
    eucl = np.linalg.norm(fs_1 - fs_2)
    return eucl
def katz(H_1,H_2):
    data = np.real(ifft(H_1))
    fs_1 = np.array([ant.katz_fd(data)])
    data = np.real(ifft(H_2))
    fs_2 = np.array([ant.katz_fd(data)])
    eucl = np.linalg.norm(fs_1 - fs_2)
    return eucl
def higuchi(H_1,H_2):
    data = np.real(ifft(H_1))
    fs_1 = np.array([ant.higuchi_fd(data)])
    data = np.real(ifft(H_2))
    fs_2 = np.array([ant.higuchi_fd(data)])
    eucl = np.linalg.norm(fs_1 - fs_2)
    return eucl
def sevcik(H_1,H_2):
    data = np.real(ifft(H_1))
    fs_1 = np.array([sfe.sevcik(data)])
    data = np.real(ifft(H_2))
    fs_2 = np.array([sfe.sevcik(data)])
    eucl = np.linalg.norm(fs_1 - fs_2)
    return eucl
def mutual_info(H_1,H_2):
    return sm.mutual_info_score(np.real(ifft(H_1)),np.real(ifft(H_2)))
def shannon_entropy(H_1,H_2):
    return entropy(np.real(H_1),np.real(H_2))
def kl_divergence(p, q):
    return np.real(sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p))))
def js_divergence(p, q):
    m = 0.5 * (np.asarray(p) + np.asarray(q))
    return abs(np.real(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)))



ext_lin = "C:/Users/16098/Documents/Research Project/Exported_Data_l_10"
ext_chaotic = "C:/Users/16098/Documents/Research Project/Exported_Data_l_10_chaotic"

def plotter(data, ext, pos, method):
    xs,ys = extract(data,"False")
    if pos == "middle":
        x,y = np.median(xs),np.median(ys)
    if pos == "far edge":
        x,y = np.min(xs),np.min(ys)
    if pos == "close edge":
        x,y = np.max(xs), np.max(ys)
    H_f_origin = opener(x,y,data)
    print(H_f_origin)
    corrs = []
    x_s = []
    y_s = []
    for i in range(0,len(xs)):
        for j in range(0,len(ys)):
            H_f_test = opener(xs[i],ys[j],data)
            corrs.append(method(H_f_origin,H_f_test))
            x_s.append(xs[i])
            y_s.append(ys[j])
    np.savetxt((ext + "/" + str(method.__name__) + "_" + str(len(xs)) + ".txt"), corrs, delimiter = ",")
    if method == corr_abs:
        plt.title("Correlation absolute")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens")
    if method == corr_real:
        plt.title("Correlation real")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens")
    if method == ml:
        plt.title("Fractal features")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens_r")
    if method == corr_imag:
        plt.title("Correlation imaginary")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens")
    if method == petrosian:
        plt.title("Petrosian")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens_r")
    if method == katz:
        plt.title("Katz")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens_r")
    if method == higuchi:
        plt.title("Higuchi")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens_r")
    if method == sevcik:
        plt.title("Sevcik")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens_r")
    if method == mutual_info:
        plt.title("Mutual Information")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens")
    if method == entropy:
        plt.title("Shannon Entropy")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens_r")
    if method == kl_divergence:
        plt.title("Kullback Leibler")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens_r")
    if method == js_divergence:
        plt.title("Jensen-Shannon Divergence")
        plt.scatter(x_s, y_s, c=corrs, s=3, cmap="Greens_r")
    plt.plot(x,y,"o",color = "red")
    plt.xticks(np.array([np.min(xs) - 0.5, np.min(xs), np.mean(xs), np.max(xs), np.max(xs) + 0.5]))
    plt.yticks(np.array([np.min(ys) - 0.5, np.min(ys), np.mean(ys), np.max(ys), np.max(ys) + 0.5]))
    plt.xlim([np.min(xs) - 0.4, np.max(xs) + 0.8])
    plt.ylim(([np.min(ys) - 0.4, np.max(ys) + 0.8]))
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")


    plt.show()
    return corrs

#corrs = plotter(data_chaotic, ext_chaotic,"middle", shannon_entropy)
H_fs = opener(0.6,0.1,data_chaotic)
def Extrapolator(H_fs,f_s,f_e,bool):
    fs = np.linspace(0,3.7477e+10,18738)
    fs_lim = np.linspace(fs[f_s],fs[f_e],(f_e-f_s)+1)
    f_peak = int((f_s+f_e)/2)
    f_med = int(18738/2)
    f_antipeak = f_med + (f_med - f_peak)
    f_s_antipeak = f_antipeak - (f_peak - f_s)
    f_e_antipeak = f_antipeak + (f_e - f_peak)
    H_fs_new = []
    for i in range(0,f_s):
        H_fs_new.append(complex(0))
    for i in range(f_s,f_e):
        H_fs_new.append(H_fs[i-f_s])
    for i in range(f_e,f_s_antipeak):
        H_fs_new.append(complex(0))
    for i in range(f_s_antipeak, f_e_antipeak):
        real = np.real(H_fs[-(i - f_s_antipeak+1)])
        imag = -np.imag(H_fs[-(i - f_s_antipeak+1)])*1j
        H_fs_new.append(real + imag)
    for i in range(f_e_antipeak, len(fs)):
        H_fs_new.append(complex(0))
    if bool == True:
        plt.plot(fs_lim,H_fs)
        plt.show()
        plt.plot(fs,H_fs_new)
        plt.show()
    return np.real_if_close(H_fs_new)


H_fs_new = Extrapolator(H_fs,1000,1400,True)
print(np.real_if_close(ifft(H_fs_new),tol = 1e-5))
print(sum(np.imag(ifft(H_fs_new)))/len(H_fs_new))