import numpy as np
import signal_feature_extraction as sfe
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core.dropout import Dropout
import os
import matplotlib.pyplot as plt
os.add_dll_directory('C:/Program Files/Graphviz/bin')


one_hot = lambda N: np.identity(N).tolist()

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(10, input_dim=n_inputs, activation='relu'))
    Dropout(0.3, noise_shape=None)
    model.add(Dense(20, input_dim=n_inputs, activation='relu'))
    Dropout(0.3, noise_shape=None)
    model.add(Dense(40, input_dim=n_inputs, activation='relu'))
    Dropout(0.3, noise_shape=None)
    model.add(Dense(80, input_dim=n_inputs, activation='relu'))
    Dropout(0.3, noise_shape=None)
    model.add(Dense(160, input_dim=n_inputs, activation='relu'))
    Dropout(0.3, noise_shape=None)
    model.add(Dense(80, input_dim=n_inputs, activation='relu'))
    Dropout(0.3, noise_shape=None)
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

N = 50
n = 5
e_ts, fs = sfe.random_fixture_sample(N)
e_ts_noisy = sfe.noiser(e_ts, n)
fs_noisy = sfe.fixture_sampler(e_ts_noisy)


X_train = np.asarray(fs).astype('float32')
X_test = np.asarray(fs_noisy).astype('float32')
Y_train = np.asarray(one_hot(N)).astype('float32')
Y_test = np.asarray(one_hot(N)).astype('float32')
print(X_train)

print(X_train)
print(Y_train)
print(X_test)
print(Y_test)


model = get_model(len(fs[0]), len(fs))
model.fit(X_train, Y_train, verbose=0, epochs=100, batch_size=16)


Y_pred = model.predict(X_test).tolist()


Y_pred_one_hot = []
for i in range(0,len(Y_pred)):
    Y_pred_one_hot.append([0]*N)
    Y_pred_one_hot[i][Y_pred[i].index(max(Y_pred[i]))] = 1


Ys  = []
for i in range(0,len(Y_pred_one_hot)):
    Ys.append(Y_pred_one_hot[i].index(max(Y_pred_one_hot[i])))


def Accuracy(N):
    SNRs = []
    error_rates = []
    for i in range(0,100):
        error_rate = 0
        print(i)
        n = (i+1)*50
        e_ts, fs = sfe.random_fixture_sample(N)
        e_ts_noisy = sfe.noiser(e_ts, n)
        fs_noisy = sfe.fixture_sampler(e_ts_noisy)
        X_test = fs_noisy
        Y_pred = model.predict(X_test).tolist()
        Y_pred_one_hot = []
        for i in range(0, len(Y_pred)):
            Y_pred_one_hot.append([0] * N)
            Y_pred_one_hot[i][Y_pred[i].index(max(Y_pred[i]))] = 1
        Ys = []
        for i in range(0, len(Y_pred_one_hot)):
            Ys.append(Y_pred_one_hot[i].index(max(Y_pred_one_hot[i])))
        for j in range(0,len(Ys)):
            bool = (Ys[j] != j)
            error_rate = error_rate + int(bool)
        S = np.mean(np.abs(e_ts))
        SNRs.append(np.log10(S) / np.log10(n))
        error_rates.append(error_rate/100)
    plt.plot(SNRs,error_rates,"o")
    plt.ylabel("Error rate")
    plt.xlabel("SNR")
    plt.title("BP")
    plt.show()

#Accuracy(50)

