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

### feed estimator the parm from locator ###
t0, tE = [8280.102654, 40.558097]
# t0, tE = pred_t[0]

### preprocess ###
if gap:
    tel = pd.read_csv('./KMT/%s/%s_aligned.csv'%(eventname, eventname), skiprows=0, usecols=[0, 2, 5, 6])
    tel = tel[(tel['Tel'] != 'KMTS01') & (tel['Tel'] != 'KMTS41')]
    x_orig = tel[['HJD', 'mag_aligned', 'e_mag_aligned']].to_numpy()
else:
    x_orig = np.loadtxt('./KMT/%s/%s_aligned.csv'%(eventname, eventname), delimiter=',', skiprows=1, usecols=(2, 5, 6))
order = np.argsort(x_orig[:, 0])
x_orig = x_orig[order]
err = x_orig[:, -1]
x_orig = x_orig[:, :-1]


x_orig[:, 0] = (x_orig[:, 0] - t0)/tE
## -2 < t < 2
ind = (x_orig[:, 0] > -2) * (x_orig[:, 0] < 2)
mbase = x_orig[~ind, 1].mean()
x_orig = x_orig[ind]
err = err[ind]

fig, ax = plt.subplots(1, 1)
ax.errorbar(x_orig[:, 0], x_orig[:, 1], err, marker='o', markersize=2)
ax.invert_yaxis()
plt.show()

x = x_orig.copy()
## Perform a median filter on an N-dimensional array.
print(x[:, 1])
if gap:
    x[:, 1] = medfilt(x[:, 1], 7)
else:
    x[:, 1] = medfilt(x[:, 1], 5)

## err < 0.06
threshold = 0.06
print(f'len of good data:{len(x[err<threshold, 0])}')
ind = err < threshold
x = x[ind]
err = err[ind]

## align magnitude
print(f'mbase:{mbase}')
x[:, 1] = (x[:, 1] - mbase) / 0.2

fig, ax  = plt.subplots(1, 1)
ax.scatter(x[:, 0], x[:, 1], s=1)
ax.invert_yaxis()


### predict ###
data = torch.tensor(x.reshape(1, *x.shape))
# depth = 3; window_length = 4
depth = 3; window_length = 8
data = torchcde.logsig_windows(data, depth, window_length=window_length)
train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data)


checkpt = torch.load('experiments/estimator/estimator_l32nG12diag.ckpt', map_location='cpu')
ckpt_args = checkpt['args']
state_dict = checkpt['state_dict']
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

output_dim = 5
input_dim = data.shape[-1]
latent_dim = ckpt_args.latents

model = CDE_MDN(input_dim, latent_dim, output_dim).to(device)
model_dict = model.state_dict()

# 1. filter out unnecessary keys
state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(state_dict) 
# 3. load the new state dict
model.load_state_dict(state_dict)
model.to(device)

with torch.no_grad():
    pi, normal = model(train_coeffs.float().to(device))


n = int(1e6)

pi_ = pi.probs.cpu(); loc_ = normal.loc.cpu(); scale_ = normal.scale.cpu()
pi_ = torch.tile(pi_, (n, 1)); loc_ = torch.tile(loc_, (n, 1, 1)); scale_ = torch.tile(scale_, (n, 1, 1))
normal_dist = torch.distributions.Normal(loc_, scale_)
pi_dist = torch.distributions.OneHotCategorical(probs=pi_)
sample = model.sample(pi_dist, normal_dist).numpy()


test_p = [[0.140, np.log10(6.46)-3, np.log10(0.079), np.log10(0.826), 2-1.542/np.pi, 0],
        [0.141, np.log10(6.99)-3, np.log10(0.123), np.log10(1.575), 2-1.544/np.pi, 0]]
fs, fb = getfsfb(x[:, 0], 10**(x[:, 1]/5/(-2.5)), err/(2.5*np.log(10))*10**(x[:, 1]/5/(-2.5)), 0, 1, *test_p[0][:-1])[1:3]
test_p[0][-1] = np.log10(fs / (fs + fb))
fs, fb = getfsfb(x[:, 0], 10**(x[:, 1]/5/(-2.5)), err/(2.5*np.log(10))*10**(x[:, 1]/5/(-2.5)), 0, 1, *test_p[1][:-1])[1:3]
test_p[1][-1] = np.log10(fs / (fs + fb))
truths_full = np.array(test_p)
truths = np.delete(truths_full, 1, axis=-1)

range_p = [(0, 1), (-4, 0), (-0.6, 0.6), (0, 2), (-1, 0)]
# range_p = None
sigma_level = 1-np.exp(-0.5)
fig = corner.corner(sample, labels=[r"$u_0$", r"$\lg q$", r"$\lg s$", r"$\alpha/180$", r"$\lg f_s$"],
            smooth=1,
            bins=50,
            range=range_p,
            show_titles=True, title_kwargs={"fontsize": 12},
            truths=truths[0], truth_color='C1', 
            fill_contours=False, color='blue', no_fill_contours=True,
            plot_datapoints=False, plot_density=False,
            )
corner.overplot_lines(fig, truths[1], color="C2")
corner.overplot_points(fig, truths[1][None], marker="s", color="C2")
plt.show()


from scipy.optimize import fmin
import VBBinaryLensing

def get_fsfb(amp, flux, ferr):
    sig2 = ferr**2
    wght = flux/sig2
    d = np.ones(2)
    d[0] = np.sum(wght*amp)
    d[1] = np.sum(wght)
    b = np.zeros((2,2))
    b[0,0] = np.sum(amp**2/sig2)
    b[0,1] = np.sum(amp/sig2)
    b[1,0] = b[0,1]
    b[1,1] = np.sum(1./sig2)
    c = np.linalg.inv(b)
    fs = np.sum(c[0]*d)
    fb = np.sum(c[1]*d)
    fserr = np.sqrt(c[0,0])
    fberr = np.sqrt(c[1,1])
    fmod = fs*amp+fb
    chi2 = np.sum((flux-fmod)**2/sig2)
    return chi2,fs,fb,fserr,fberr

def compute_model_lc(time_array, fitting_parameters, VBBL, rho=None):
    u0, lgq, lgs, ad180 = fitting_parameters[:4]
    q, s = 10**lgq, 10**lgs
    alpha = ad180 * np.pi # convert to radian
    t0, te = 0, 1
    if rho == None:
        rho = 1e-3
    if len(fitting_parameters) == 5:
        rho = 10**fitting_parameters[-1]
    if len(fitting_parameters) == 6:
        t0 = fitting_parameters[-2]
        te = fitting_parameters[-1]
    if len(fitting_parameters) == 7:
        rho = 10**fitting_parameters[-3]
        t0 = fitting_parameters[-2]
        te = fitting_parameters[-1]
    tau = (time_array-t0)/te
    xs = tau*np.cos(alpha) - u0*np.sin(alpha)
    ys = tau*np.sin(alpha) + u0*np.cos(alpha)
    magnifications = np.array([VBBL.BinaryMag2(s, q, xs[i], ys[i], rho) for i in range(len(xs))])
    return magnifications

def compute_chisq(fitting_parameters, time, flux, ferr, VBBL, return_model=False, rho=None, return_lc=False):
        if len(fitting_parameters) == 5:
            rho = 10**fitting_parameters[-1]
            fitting_parameters = fitting_parameters[:-1]
        magnifications = compute_model_lc(time, fitting_parameters, VBBL, rho)
        chi2, fs, fb, fserr, fberr = get_fsfb(magnifications, flux, ferr)
        if return_lc:
            time_model = np.arange(time.min(), time.max(), 0.001)
            magnifications = compute_model_lc(time_model, fitting_parameters, VBBL, rho)
            mag_model = 18 - 2.5*np.log10(magnifications*fs + fb)
            model = np.vstack((time_model, mag_model))
            return chi2, fs, fb, model
        if return_model:
            return chi2, fs, fb
        return chi2

def perform_optimization(time, flux, ferr, para_initial, verbose=True):
    VBBL = VBBinaryLensing.VBBinaryLensing()

    para_best, chi2_min, iter, funcalls, warnflag, allevcs = fmin(compute_chisq, para_initial, args=(time, flux, ferr, VBBL), full_output=True, retall=True, maxiter=1000, maxfun=5000, disp=verbose)

    chi2_min, fs, fb = compute_chisq(para_initial, time, flux, ferr, VBBL, return_model=True)
    if verbose:
        print('initial chisq: ', chi2_min)
    chi2_min, fs, fb = compute_chisq(para_best, time, flux, ferr, VBBL, return_model=True)
    if verbose:
        print('best chisq & (fs, fb): ', chi2_min, fs, fb)
    time_model = np.arange(time.min(), time.max(), 0.001)
    magnifications = compute_model_lc(time_model, para_best, VBBL)
    mag_model = 18 - 2.5*np.log10(magnifications*fs + fb)
    model = np.vstack((time_model, mag_model))
    return para_best, chi2_min, model, warnflag

def prepare_lc_mdn(X, pis, locs):
    mag = X[:, :, 1] / 5 + 18
    flux = 10 ** (0.4 * (18 - mag))
    merr = torch.ones_like(mag) * 0.033
    ferr = merr*flux*np.log(10)/2.5
    # times, mag, flux, ferr
    lc = torch.stack([X[:, :, 0], mag, flux, ferr], dim=-1)
    first_indices = torch.arange(len(pis))[:, None]
    order = torch.argsort(pis, dim=-1, descending=True)
    pis = pis[first_indices, order]
    locs = locs[first_indices, order]
    return lc.numpy(), pis.numpy(), locs.numpy()


# lc = torch.tensor(x_orig.reshape(1, *x_orig.shape))
# lc[:, :, 1] = (lc[:, :, 1] - mbase) / 0.2

if gap:
    tel = pd.read_csv('./KMT/%s/%s_aligned.csv'%(eventname, eventname), skiprows=0, usecols=[0, 2, 5, 6])
    tel = tel[(tel['Tel'] != 'KMTS01') & (tel['Tel'] != 'KMTS41')]
    lc = tel[['HJD', 'mag_aligned', 'e_mag_aligned']].to_numpy()[:, :-1].reshape(1, -1, 2)
    lc[:, :, 1] = (lc[:, :, 1] - mbase) / 0.2
else:
    lc = np.loadtxt('./KMT/%s/%s_aligned.csv'%(eventname, eventname), delimiter=',', skiprows=1, usecols=(2, 5)).reshape(1, -1, 2)
    lc[:, :, 1] = (lc[:, :, 1] - mbase) / 0.2
plt.scatter(lc[0, :, 0], lc[0, :, 1])

lc, pis_sort, locs_sort = prepare_lc_mdn(torch.tensor(lc), pi.probs.cpu(), normal.loc.cpu())

plt.scatter(lc[0, :, 0], lc[0, :, 1]-18)
plt.scatter(data[0, :, 0] * tE + t0, data[0, :, 1]/5)

dof = lc.shape[1]

i = 0
verbose = True
fig = plt.figure(1, (12, 12))
ax_lc = plt.subplot2grid(shape=(3, 3), loc=(0, 0), rowspan=1, colspan=3)
plt.xlabel(r'HJD-2450000', fontsize=20)
plt.ylabel(r'$m$', fontsize=20)

best_parameters = []
colors = ['orange', 'green', 'black']
labels = ['close', 'wide', 'black']
linestyle = ['-', 'dashed', '-']
minimum_chi2 = np.inf
for index in tqdm(range(2)):
    para_initial = locs_sort[i, index, :-1]
    para_initial = para_initial.tolist()
    if index == 0 and gap == True:
        para_initial.insert(4, -1.7) # lgrho
    else:
        para_initial.insert(4, -2)
    para_initial.insert(5, t0) # t0
    para_initial.insert(6, tE) # te
    if verbose:
        print(para_initial)
    para_best, chi2_min, model, warnflag = perform_optimization(lc[i, :, 0], lc[i, :, 2], lc[i, :, 3], para_initial, verbose=verbose)
    chi2_min, fs, fb = compute_chisq(para_best, lc[i, :, 0], lc[i, :, 2], lc[i, :, 3], VBBinaryLensing.VBBinaryLensing(), return_model=True)
    print('fs', fs, 'fb', fb)
    if chi2_min < minimum_chi2:
        minimum_chi2 = chi2_min
    if len(para_best) == 5:
        lgrho = para_best[-1]
        para_best = np.delete(para_best, -1).tolist()
        para_best.insert(1, lgrho)
    if len(para_best) == 7:
        lgrho = para_best[-3]
        para_best = np.delete(para_best, -3).tolist()
        para_best.insert(1, lgrho)
    best_param = np.hstack((chi2_min, warnflag, para_best, np.array([np.log10(fs/(fs+fb))])))
    best_parameters.append(best_param)
    # plt.plot(model[0] * pred_t[0][1] + pred_t[0][0], model[1] - 18 + mbase, label=r'%s, $\chi^2$=%.1f'%(labels[index], chi2_min), color=colors[index], linestyle=linestyle[index])
    plt.plot(model[0], model[1] - 18 + mbase, label=r'%s, $\chi^2$=%.1f'%(labels[index], chi2_min / minimum_chi2 * dof), color=colors[index], linestyle=linestyle[index])

print('best params:', best_parameters)


tru_wide = truths_full[1].tolist()
delta = 10**truths_full[1][2]/(1+10**truths_full[1][2])*(10**truths_full[1][3]-1/10**truths_full[1][3])
tru_wide[0] -= delta * np.cos(tru_wide[-2]*np.pi-3*np.pi/2)


tru_close = truths_full[0].tolist()
tru_time_close = (lc[i, :, 0] * tE + t0 - 8592.392) / 6.50
tru_time_wide = (lc[i, :, 0] * tE + t0 - 8592.391) / 6.57

# chi2_min, fs, fb, model = compute_chisq(np.array(tru_close)[[0, 2, 3, 4]], tru_time_close, lc[i, :, 2], lc[i, :, 3], VBBinaryLensing.VBBinaryLensing(), rho=10**tru_close[1], return_model=True, return_lc=True)
# truths[0][-1] = np.log10(fs/(fs+fb))
# plt.plot(model[0], model[1], label=r'close, $\chi^2$=%.1f'%(chi2_min), color='red', linestyle='dashed')

# chi2_min, fs, fb, model = compute_chisq(np.array(tru_wide)[[0, 2, 3, 4]], tru_time_wide, lc[i, :, 2], lc[i, :, 3], VBBinaryLensing.VBBinaryLensing(), rho=10**tru_wide[1], return_model=True, return_lc=True)
# truths[1][-1] = np.log10(fs/(fs+fb))
# plt.plot(model[0], model[1], label=r'wide, $\chi^2$=%.1f'%(chi2_min), color='blue', linestyle='dashed')


# plt.scatter(lc[0, :, 0] * pred_t[0][1] + pred_t[0][0], lc[0, :, 1] - 18 + mbase, color='black', alpha=0.5, rasterized=True)
plt.scatter(lc[0, :, 0], lc[0, :, 1] - 18 + mbase, color='black', alpha=0.5, rasterized=True)
# plt.xlim(-0.3 * pred_t[0][1] + pred_t[0][0], 0.3 * pred_t[0][1] + pred_t[0][0])
plt.xlim(t_start, t_end)
plt.ylim(16, 18)
plt.legend(loc='upper right', prop={'size': 16})
plt.gca().invert_yaxis()

param_list = [tru_close, best_parameters[0][2:]]
traj_color = ['red', 'orange']
cau_color = traj_color

ax_geo = plt.subplot2grid(shape=(3, 3), loc=(1, 1), rowspan=1, colspan=2)
plt.axis('equal')
plt.xlim(-1, 1)
plt.ylim(-0.3, 0.3)
plt.yticks([-0.5, 0, 0.5])
for j, params in enumerate(param_list):
    if j != 0:
        print(params)
        if len(params) == 5:
            u_0, lgq, lgs, alpha_180, lgfs = params
            lgrho = -3
            t0, te = 0, 1
        if len(params) == 8:
            u_0, lgrho, lgq, lgs, alpha_180, t0, te, lgfs = params
        else:
            u_0, lgrho, lgq, lgs, alpha_180, lgfs = params
            t0, te = 0, 1
        parameters = {
                    't_0': t0,
                    't_E': te,
                    'u_0': u_0,
                    'rho': 10**lgrho, 
                    'q': 10**lgq, 
                    's': 10**lgs, 
                    'alpha': alpha_180*180,
                }
        modelmm = mm.Model(parameters, coords=None)
        if j == 0 or j == 1:
            modelmm.plot_trajectory(t_range=(t0-2*te, t0+2*te), caustics=False, arrow=True, color=traj_color[j], linestyle='dashed')
        else:
            modelmm.plot_trajectory(t_range=(t0-2*te, t0+2*te), caustics=False, arrow=False, color=traj_color[j])
        modelmm.plot_caustics(color=cau_color[j], s=3)

param_list = [tru_wide, best_parameters[1][2:]]
traj_color = ['blue', 'green']
cau_color = traj_color


ax_geo = plt.subplot2grid(shape=(3, 3), loc=(2, 1), rowspan=1, colspan=2)
plt.axis('equal')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.yticks([-0.5, 0, 0.5])
for j, params in enumerate(param_list):
    if j != 0:
        print(params)
        if len(params) == 5:
            u_0, lgq, lgs, alpha_180, lgfs = params
            lgrho = -3
            t0, te = 0, 1
        if len(params) == 8:
            u_0, lgrho, lgq, lgs, alpha_180, t0, te, lgfs = params
        else:
            u_0, lgrho, lgq, lgs, alpha_180, lgfs = params
            t0, te = 0, 1
        parameters = {
                    't_0': t0,
                    't_E': te,
                    'u_0': u_0,
                    'rho': 10**lgrho, 
                    'q': 10**lgq, 
                    's': 10**lgs, 
                    'alpha': alpha_180*180,
                }
        modelmm = mm.Model(parameters, coords=None)
        if j == 0 or j == 1:
            modelmm.plot_trajectory(t_range=(t0-2*te, t0+2*te), caustics=False, arrow=True, color=traj_color[j], linestyle='dashed')
        else:
            modelmm.plot_trajectory(t_range=(t0-2*te, t0+2*te), caustics=False, arrow=False, color=traj_color[j])
        modelmm.plot_caustics(color=cau_color[j], s=3)

# ax = plt.subplot2grid((3, 3), (2, 1), rowspan=1, colspan=2)
# ax.axis('off')
# plt.text(-0.5, 0.7, r'Truth close:' '\n' r'$u_0=%.3f$, $\lg q=%.3f$, $\lg s=%.3f$,' '\n' r'$\lg\rho=%.2f$, $\alpha=%.3f\degree$, $\lg f_{\rm S}=%.3f$' % (
#     tru_close[0], tru_close[2], tru_close[3], tru_close[1], tru_close[4]*180, tru_close[5]
# ), fontsize=20, color='red')
# plt.text(0.2, 0.7, r'Truth wide:' '\n' r'$u_0=%.3f$, $\lg q=%.3f$, $\lg s=%.3f$,' '\n' r'$\lg\rho=%.2f$, $\alpha=%.3f\degree$, $\lg f_{\rm S}=%.3f$' % (
#     tru_wide[0], tru_wide[2], tru_wide[3], tru_wide[1], tru_wide[4]*180, tru_wide[5]
# ), fontsize=20, color='blue')
# # plt.text(0.2, 0.5, r'Closest Peak:' '\n' r'$u_0=%.3f$, $\lg q=%.3f$, $\lg s=%.3f$,' '\n' r'$\lg\rho=%d$, $\alpha=%.3f\degree$, $\lg f_{\rm S}=%.3f$' % (
# #     param_pred_gap[0], param_pred_gap[2], param_pred_gap[3], param_pred_gap[1], param_pred_gap[4]*180, param_pred_gap[5]
# # ), fontsize=20)
# # plt.text(0.2, 0.1, r'Global Peak:' '\n' r'$u_0=%.3f$, $\lg q=%.3f$, $\lg s=%.3f$,' '\n' r'$\lg\rho=%d$, $\alpha=%.3f\degree$, $\lg f_{\rm S}=%.3f$' % (
# #     param_pred_gap_g[0], param_pred_gap_g[2], param_pred_gap_g[3], param_pred_gap_g[1], param_pred_gap_g[4]*180, param_pred_gap_g[5]
# # ), fontsize=20)
# plt.text(-0.5, 0.3, r'Predicted close:' '\n' r'$u_0=%.3f$, $\lg q=%.3f$, $\lg s=%.3f$,' '\n' r'$\lg\rho=%d$, $\alpha=%.3f\degree$, $\lg f_{\rm S}=%.3f$' % (
#     best_parameters[0][2:][0], best_parameters[0][2:][1], best_parameters[0][2:][2], -3, best_parameters[0][2:][3]*180, best_parameters[0][2:][4]
# ), fontsize=20, color='orange')
# plt.text(0.2, 0.3, r'Predicted wide:' '\n' r'$u_0=%.3f$, $\lg q=%.3f$, $\lg s=%.3f$,' '\n' r'$\lg\rho=%d$, $\alpha=%.3f\degree$, $\lg f_{\rm S}=%.3f$' % (
#     best_parameters[1][2:][0], best_parameters[1][2:][1], best_parameters[1][2:][2], -3, best_parameters[1][2:][3]*180, best_parameters[1][2:][4]
# ), fontsize=20, color='green')
plt.tight_layout()
if gap:
    plt.savefig('./KMT/%s/lc_kmt_gap_%s.png'%(eventname,0))
else:
    plt.savefig('./KMT/%s/lc_kmt_%s.png'%(eventname,0))
plt.show()


checkpt = torch.load('experiments/estimator/estimator_l32nG12diag.ckpt', map_location='cpu')
ckpt_args = checkpt['args']
state_dict = checkpt['state_dict']

output_dim = 5
input_dim = data.shape[-1]
latent_dim = ckpt_args.latents

model = CDE_MDN(input_dim, latent_dim, output_dim).to(device)
model_dict = model.state_dict()

# 1. filter out unnecessary keys
state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(state_dict) 
# 3. load the new state dict
model.load_state_dict(state_dict)
model.to(device)
with torch.no_grad():
    pi, normal = model(train_coeffs.float().to(device))
plt.rc('font', size=12)

import plot_triangle
## first sort the gaussians according to the weight ##
i = 0
pi = pi.probs.cpu().numpy()[i]
mu = normal.loc.cpu().numpy()[i]
scale = normal.scale.cpu().numpy()[i]
order = np.argsort(pi)[::-1]
pi = pi[order][:4]
mu = mu[order][:4][:, :-1]
scale = scale[order][:4][:, :-1]
print(pi)
## the ground truth ##
# param_true = [0.34033019, -0.35541759, 0.01886, 1.18859352, -0.8133]
param_true = truths[:, :-1]
print(param_true)
#param_true = [0.247, -0.074, 0.227, 291.236/180, -0.094]
param_pred = np.zeros_like(param_true)
norm_fac = np.zeros_like(param_true)

param_best = best_parameters.copy()
for i in range(len(param_best)):
    param_best[i] = np.delete(param_best[i][2:], 1)

n_param = len(mu[0])
# labels = [r'$u_0$', r'$\lg q$', r'$\lg s$', r'$\alpha/180\degree$', r'$\lg f_{\rm s}$']
labels = [r'$u_0$', r'$\lg q$', r'$\lg s$', r'$\alpha/180\degree$']

# param_ranges = [[0, 0.3], [-1.55, -0.8], [-0.2, 0.3], [1.4, 1.6], [-0.2, 0.01]]
param_ranges = [[0, 0.3], [-1.55, -0.8], [-0.2, 0.3], [1.4, 1.6]]
for j_gauss in range(len(pi)):
    weight = pi[j_gauss]/pi[0] # normalize the weight by the highest value, so that the color appears better
    cov_mat = np.zeros((n_param, n_param))
    for i in range(len(mu[j_gauss])):
        cov_mat[i, i] = scale[j_gauss, i]**2
    norm_fac += pi[j_gauss] #/np.linalg.det(cov_mat)
    param_pred += mu[j_gauss] * pi[j_gauss] #/np.linalg.det(cov_mat) * mu[j_gauss]
    if j_gauss == 0:
        fig, axes = plot_triangle.plot_covariance(mu[j_gauss], labels, cov_mat, weight=weight, ground_truth_color='red', ground_truth_ls='dashed')
        # fig, axes = plot_triangle.plot_covariance(mu[j_gauss], labels, cov_mat, weight=weight, ground_truth=param_true[0], ground_truth_color='red', ground_truth_ls='dashed')
        # plot_triangle.plot_covariance(mu[j_gauss], labels, cov_mat, fig=fig, axes=axes, weight=weight, ground_truth=param_true[1], ground_truth_color='blue', ground_truth_ls='dashed')
    if j_gauss == len(pi)-1:
        param_pred /= norm_fac
        # plot_triangle.plot_covariance(mu[j_gauss], labels, cov_mat, fig=fig, axes=axes, weight=weight)
        plot_triangle.plot_covariance(mu[j_gauss], labels, cov_mat, fig=fig, axes=axes, weight=weight, extents=param_ranges, ground_truth=np.array(param_best)[0], ground_truth_color='orange', ground_truth_ls='-')
        plot_triangle.plot_covariance(mu[j_gauss], labels, cov_mat, fig=fig, axes=axes, weight=weight, extents=param_ranges, ground_truth=np.array(param_best)[1], ground_truth_color='green', ground_truth_ls='-')
    else:
        plot_triangle.plot_covariance(mu[j_gauss], labels, cov_mat, fig=fig, axes=axes, weight=weight)

# mu_truth = param_true
# cov_mat_truth = [np.diag((0.007, 0.002/0.079/np.log(10), 0.011/0.826/np.log(10), 0.008/np.pi, 0.492/0.190/np.log(10)*np.sqrt(3)*0.01))**2, 
#                 np.diag((0.007, 0.003/0.079/np.log(10), 0.014/0.826/np.log(10), 0.007/np.pi, 0.535/0.206/np.log(10)*np.sqrt(3)*0.01))**2]
# plot_triangle.plot_covariance(mu_truth[0], labels, cov_mat_truth[0], fig=fig, axes=axes, weight=0.5, extents=param_ranges)
# plot_triangle.plot_covariance(mu_truth[1], labels, cov_mat_truth[1], fig=fig, axes=axes, weight=0.5, extents=param_ranges)
if gap:
    plt.savefig('./KMT/%s/corner_kmt_gap_%s.png'%(eventname,k))
else:
    plt.savefig('./KMT/%s/corner_kmt_%s.png'%(eventname,k))
plt.show()

