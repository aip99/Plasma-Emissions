import xgboost as xgb
import signal_feature_extraction as sfe
import numpy as np
import matplotlib.pyplot as plt

one_hot = lambda N: np.identity(N).tolist()

def weights(N):
    reg = xgb.XGBRegressor(learning_rate =.1, n_estimators=1000,
                     max_depth=4, min_child_weight=5, gamma=0,
                     subsample=.8, colsample_bytree=.7, reg_alpha=1,
                     objective= 'reg:linear')
    e_ts, fs = sfe.random_fixture_sample(N)
    y = one_hot(N)
    X = fs
    reg.fit(X, y)
    reg.get_booster().feature_names = ["Petrosian", "Katz", "Higuchi", "Detrend", "Sevcik"]
    xgb.plot_importance(reg)
    plt.show()
    pass

weights(1000)
