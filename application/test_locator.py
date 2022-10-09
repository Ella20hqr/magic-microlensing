import os, sys, re
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
import torchcde
import h5py
from scipy.signal import find_peaks
import MulensModel as mm
import corner
from model.utils import getfsfb
from matplotlib.offsetbox import AnchoredText

from scipy.signal import medfilt
from PyAstronomy.pyasl import binningx0dt

from model.locator import Locator
from model.cde_mdn import CDE_MDN

torch.random.manual_seed(42)
np.random.seed(42)
plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rc('font', size=14)

gap = False
t_start = 8180
t_end = 8380
eventname = 'KB180800'

### preprocess
if gap:# create some gap
    tel = pd.read_csv('./KMT/%s/%s_aligned.csv'%(eventname, eventname), skiprows=0, usecols=[0, 2, 5, 6])
    tel = tel[(tel['Tel'] != 'KMTS01') & (tel['Tel'] != 'KMTS41')]
    x_orig = tel[['HJD', 'mag_aligned', 'e_mag_aligned']].to_numpy()
else:
    x_orig = np.loadtxt('./KMT/%s/%s_aligned.csv'%(eventname, eventname), delimiter=',', skiprows=1, usecols=(2, 5, 6))
order = np.argsort(x_orig[:, 0])
x_orig = x_orig[order]
err = x_orig[:, -1]
x = x_orig[:, :-1]
err = err[x[:, 0] > t_start]
x = x[x[:, 0] > t_start]
if gap:
    err = err[x[:, 0] < 8650]
    x = x[x[:, 0] < 8650]
else:
    err = err[x[:, 0] < t_end]
    x = x[x[:, 0] < t_end]


fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(311)
ax1.errorbar(x[:, 0], x[:, 1], yerr=err, marker='o', linestyle='none', markersize=2)
ax1.invert_yaxis()
print(f'Length of the original data:{len(x)}')


# x[:, 1] = medfilt(x[:, 1], 7)
threshold = 0.1
ind = err < threshold
x = x[ind]
err = err[ind]
x[:, 1] -= 5

ax2 = fig1.add_subplot(312)
ax2.errorbar(x[:, 0], x[:, 1], yerr=err, marker='o', linestyle='none', markersize=2)
ax2.invert_yaxis()
print(f'Length of the good data:{len(x)}')


# x, dt = binningx0dt(x[:, 0], x[:, 1], err, reduceBy=5, yvalFunc=np.median) # old
x, dt = binningx0dt(x[:, 0], x[:, 1], err, dt=0.125, yvalFunc=np.median, useBinCenter=True) # new
print(f'Length of the binned data:{len(x)}')

ax3 = fig1.add_subplot(313)
ax3.errorbar(x[:, 0], x[:, 1], yerr=x[:, 2], marker='o', linestyle='none', markersize=2)
ax3.invert_yaxis()
plt.show()

x = x[:, :2]

### predict
train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(torch.tensor(x).unsqueeze(0))
k = 2; method='avg'
checkpt = torch.load('experiments/locator/locator_k_%s.ckpt'%(k), map_location='cpu')
# checkpt = torch.load('/work/huqr/experiments/locator/experiment_54125.ckpt', map_location='cpu')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

ckpt_args = checkpt['args']
state_dict = checkpt['state_dict']

model = Locator(device, k=k, method=method).to(device)
model_dict = model.state_dict()

# 1. filter out unnecessary keys
state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(state_dict) 
# 3. load the new state dict
model.load_state_dict(state_dict)
model.to(device)

with torch.no_grad():
    model.eval()
    model.animate = True
    model.soft_threshold = False
    res = model(train_coeffs.float().to(device), torch.zeros((1, 2)).to(device))
    pred_t = res[0].detach().cpu().numpy()

print(pred_t)
mask = res[2].detach().cpu().numpy()[0]

ind = mask[:,1]==1
mask = mask[ind]


fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.axvline(mask[0][0], color='black')
ax.axvline(mask[-1][0], color='black')
ax.axvline(pred_t[0][0], color='black', linestyle='-.')
ax.scatter(x[:, 0], x[:, 1], s=5)
ax.invert_yaxis()
plt.show()