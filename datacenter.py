import os
import numpy as np

import lib.utils as utils
import lib.layouts as layouts

from lib.gaussian.hrf import HRF
from lib.gaussian.gbn import GBN
from lib.gaussian.gmrf import GMRF

K = list(layouts.datacenter_layout.keys())
datacenter = utils.prep_dataframe(keep=K)

datacenter_shifted = utils.create_shifted_features(datacenter)
datacenter = datacenter.join(datacenter_shifted, how="outer")
datacenter = datacenter.dropna()

datacenter_layout = layouts.datacenter_layout

data_folder = os.path.join(os.path.dirname(__file__), './data/')
gmrf = GMRF(variables_names=datacenter.columns.values)
gmrf.precision_ = np.load(data_folder + "gmrf_prec.npy")
gmrf.mean_ = np.load(data_folder + "gmrf_mean.npy")

hrf = HRF(5, 10, variables_names=datacenter.columns.values)
BNs = np.load(data_folder + "hybrid_bns.npy")

hrf.bns = np.empty(BNs.shape[0], dtype=object)
for i in range(BNs.shape[0]):
    gbn = GBN(variables_names=BNs[i][0])
    gbn.nodes = BNs[i][1]
    gbn.edges = BNs[i][2]
    gbn.compute_mean_cov_matrix()
    hrf.bns[i] = gbn

def build_vector(data):
    m = datacenter.shape[1]
    x = np.zeros((1, m))

    cols = datacenter.columns.values

    for k in data.keys():
        i = np.where(cols == k)[0][0]
        x[0, i] = data[k]

    return x
