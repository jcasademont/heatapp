import lib.utils as utils
import lib.layouts as layouts
from lib.gmrf import GMRF
import os
import numpy as np

K = list(layouts.datacenter_layout.keys())
datacenter = utils.prep_dataframe(keep=K)

datacenter_shifted = utils.create_shifted_features(datacenter)
datacenter = datacenter.join(datacenter_shifted, how="outer")
datacenter = datacenter.dropna()

datacenter_layout = layouts.datacenter_layout

data_folder = os.path.join(os.path.dirname(__file__), './data/')
gmrf = GMRF(variables_names=datacenter.columns.values)
gmrf.precision_ = np.load(data_folder + "heatapp_prec.npy")
gmrf.mean_ = np.load(data_folder + "heatapp_mean.npy")

def build_vector(data):
    m = datacenter.shape[1]
    x = np.zeros((1, m))

    cols = datacenter.columns.values

    for k in data.keys():
        i = np.where(cols == k)[0][0]
        x[0, i] = data[k]

    return x
