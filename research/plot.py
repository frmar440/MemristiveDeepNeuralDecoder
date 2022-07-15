import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import pandas as pd
from scipy.optimize import curve_fit

import torch
from models import MDND

from aihwkit.simulator.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.configs.utils import MappingParameter
from aihwkit.inference import RRAMLikeNoiseModel

# mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams["errorbar.capsize"] = 2


def sig_prog():

    G_T = np.linspace(0, 1, 100)
    S_RRAM = 0.0771 + 1.036*G_T
    S_PCM = 0.2635 + 1.965*G_T - 1.1731*G_T**2

    fig, ax = plt.subplots()

    ax.plot(G_T, S_RRAM, 'b-', label=r'RRAM (3iT)')
    ax.plot(G_T, S_PCM, 'r-', label=r'PCM (IBM)')
    ax.set_xlabel(r'$g_T = g / g_{max}$ [-]')
    ax.set_ylabel(r'$\sigma_{PROG}$ [$\mu$S]')
    ax.tick_params(direction='in', which='both')
    ax.legend()

    plt.savefig('research/plots/sig_prog.pdf')

def gaussian_noise_plot():

    G_T = np.linspace(0.3, 1.0, 100)
    G = G_T*200 # G_MAX = 200 \mu S
    S_RRAM = 0.0771 + 1.036*G_T

    fig, ax = plt.subplots()

    ax.plot(G_T, S_RRAM/G, 'b-', label=r'RRAM (3iT)')
    ax.axhline(0.006, color="black", linestyle="--")
    ax.annotate("Relative training noise: 0.6%", xy=(.5, .5), xycoords="axes fraction")
    ax.set_xlabel(r'$g_T = g / g_{max}$ [-]')
    ax.set_ylabel(r'$\sigma_{PROG} / g$ [-]')
    ax.tick_params(direction='in', which='both')
    ax.legend()

    plt.savefig('research/plots/gaussian_noise.pdf')

def sig_read():

    G_T = np.linspace(0, 1, 100)
    S_300K = 0.5577 + 0.11922*G_T
    S_4K = 0.3127 + 0.13698*G_T

    fig, ax = plt.subplots()

    ax.plot(G_T, S_300K, 'b-', label='300K')
    ax.plot(G_T, S_4K, 'r-', label='4K')
    ax.set_xlabel(r'$g_T = g / g_{max}$ [-]')
    ax.set_ylabel(r'$\sigma_{READ}$ [$\mu$S]')
    ax.tick_params(direction='in', which='both')
    ax.legend()

    plt.savefig('research/plots/sig_read.pdf')


def dac_adc_resolution_plot():

    decoder_performance = pd.read_pickle('research/experiments/results/decoder_performance.pkl')
    baseline = decoder_performance["baseline", "mean"]
    df = pd.read_pickle('research/experiments/results/dac_adc_resolution.pkl')

    resolutions = df.index.to_numpy()

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    
    axs[0, 0].axhline(baseline[0.15]*100, color='k', linestyle='--')
    axs[0, 0].annotate(f'Baseline: {baseline[0.15]*100:>.2f}%\n'+r'$p = 0.15\%$',
                        xy=(.45, .5), xycoords='axes fraction')     
    axs[0, 0].errorbar(resolutions, df["p0015", "mean"].to_numpy()*100, yerr=df["p0015", "std"].to_numpy()*100,
                 marker='s', color='orange')

    axs[0, 1].axhline(baseline[0.5]*100, color='k', linestyle='--')
    axs[0, 1].annotate(f'Baseline: {baseline[0.5]*100:>.2f}%\n'+r'$p = 0.50\%$',
                        xy=(.45, .5), xycoords='axes fraction')  
    axs[0, 1].errorbar(resolutions, df["p005", "mean"].to_numpy()*100, yerr=df["p005", "std"].to_numpy()*100,
                 marker='s', color='orange')

    axs[1, 0].axhline(baseline[0.7]*100, color='k', linestyle='--')
    axs[1, 0].annotate(f'Baseline: {baseline[0.7]*100:>.2f}%\n'+r'$p = 0.70\%$',
                        xy=(.45, .5), xycoords='axes fraction')    
    axs[1, 0].errorbar(resolutions, df["p007", "mean"].to_numpy()*100, yerr=df["p007", "std"].to_numpy()*100,
                 marker='s', color='orange')
    
    axs[1, 1].axhline(baseline[1.0]*100, color='k', linestyle='--')
    axs[1, 1].annotate(f'Baseline: {baseline[1.0]*100:>.2f}%\n'+r'$p = 1.00\%$',
                        xy=(.45, .5), xycoords='axes fraction')       
    axs[1, 1].errorbar(resolutions, df["p01", "mean"].to_numpy()*100, yerr=df["p01", "std"].to_numpy()*100,
                 marker='s', color='orange')

    for ax in axs.flat:
        ax.tick_params(direction='in', which='both')
        ax.set_xticks(resolutions)
        ax.label_outer()
    
    fig.supxlabel('DAC/ADC resolution [bit]')
    fig.supylabel('Decoder test accuracy [%]')

    plt.savefig('research/plots/dac_adc_resolution.pdf')


def prog_noise_scale_plot():

    decoder_performance = pd.read_pickle('research/experiments/results/decoder_performance.pkl')
    baseline = decoder_performance["baseline", "mean"]
    df = pd.read_pickle('research/experiments/results/prog_noise_scale.pkl')

    noise_scales = df.index.to_numpy()

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    
    axs[0, 0].axhline(baseline[0.15]*100, color='k', linestyle='--')
    axs[0, 0].annotate(f'Baseline: {baseline[0.15]*100:>.2f}%\n'+r'$p = 0.15\%$',
                        xy=(.1, .1), xycoords='axes fraction')
    mean, std = df["p0015", "mean"].to_numpy()*100, df["p0015", "std"].to_numpy()*100     
    axs[0, 0].plot(noise_scales, mean, color='purple')
    axs[0, 0].fill_between(noise_scales, mean-std, mean+std, facecolor='violet', alpha=0.3)

    axs[0, 1].axhline(baseline[0.5]*100, color='k', linestyle='--')
    axs[0, 1].annotate(f'Baseline: {baseline[0.5]*100:>.2f}%\n'+r'$p = 0.50\%$',
                        xy=(.1, .1), xycoords='axes fraction')  
    mean, std = df["p005", "mean"].to_numpy()*100, df["p005", "std"].to_numpy()*100     
    axs[0, 1].plot(noise_scales, mean, color='purple')
    axs[0, 1].fill_between(noise_scales, mean-std, mean+std, facecolor='violet', alpha=0.3)

    axs[1, 0].axhline(baseline[0.7]*100, color='k', linestyle='--')
    axs[1, 0].annotate(f'Baseline: {baseline[0.7]*100:>.2f}%\n'+r'$p = 0.70\%$',
                        xy=(.1, .1), xycoords='axes fraction')    
    mean, std = df["p007", "mean"].to_numpy()*100, df["p007", "std"].to_numpy()*100     
    axs[1, 0].plot(noise_scales, mean, color='purple')
    axs[1, 0].fill_between(noise_scales, mean-std, mean+std, facecolor='violet', alpha=0.3)
    
    axs[1, 1].axhline(baseline[1.0]*100, color='k', linestyle='--')
    axs[1, 1].annotate(f'Baseline: {baseline[1.0]*100:>.2f}%\n'+r'$p = 1.00\%$',
                        xy=(.1, .1), xycoords='axes fraction')       
    mean, std = df["p01", "mean"].to_numpy()*100, df["p01", "std"].to_numpy()*100     
    axs[1, 1].plot(noise_scales, mean, color='purple')
    axs[1, 1].fill_between(noise_scales, mean-std, mean+std, facecolor='violet', alpha=0.3)

    for ax in axs.flat:
        ax.tick_params(direction='in', which='both')
        ax.label_outer()
    
    fig.supxlabel('Inference noise scale [-]')
    fig.supylabel('Decoder test accuracy [%]')

    plt.savefig('research/plots/prog_noise_scale.pdf')

def pdrop_plot():

    decoder_performance = pd.read_pickle('research/experiments/results/decoder_performance.pkl')
    baseline = decoder_performance["baseline", "mean"]
    df = pd.read_pickle('research/experiments/results/pdrop.pkl')

    pdrops = df.index.to_numpy()

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    
    axs[0, 0].axhline(baseline[0.15]*100, color='k', linestyle='--')
    axs[0, 0].annotate(f'Baseline: {baseline[0.15]*100:>.2f}%\n'+r'$p = 0.15\%$',
                        xy=(.45, .5), xycoords='axes fraction')
    mean, std = df["p0015", "mean"].to_numpy()*100, df["p0015", "std"].to_numpy()*100     
    axs[0, 0].plot(pdrops, mean, color='red')
    axs[0, 0].fill_between(pdrops, mean-std, mean+std, facecolor='red', alpha=0.2)

    axs[0, 1].axhline(baseline[0.5]*100, color='k', linestyle='--')
    axs[0, 1].annotate(f'Baseline: {baseline[0.5]*100:>.2f}%\n'+r'$p = 0.50\%$',
                        xy=(.45, .5), xycoords='axes fraction')  
    mean, std = df["p005", "mean"].to_numpy()*100, df["p005", "std"].to_numpy()*100     
    axs[0, 1].plot(pdrops, mean, color='red')
    axs[0, 1].fill_between(pdrops, mean-std, mean+std, facecolor='red', alpha=0.2)

    axs[1, 0].axhline(baseline[0.7]*100, color='k', linestyle='--')
    axs[1, 0].annotate(f'Baseline: {baseline[0.7]*100:>.2f}%\n'+r'$p = 0.70\%$',
                        xy=(.45, .5), xycoords='axes fraction')    
    mean, std = df["p007", "mean"].to_numpy()*100, df["p007", "std"].to_numpy()*100     
    axs[1, 0].plot(pdrops, mean, color='red')
    axs[1, 0].fill_between(pdrops, mean-std, mean+std, facecolor='red', alpha=0.2)
    
    axs[1, 1].axhline(baseline[1.0]*100, color='k', linestyle='--')
    axs[1, 1].annotate(f'Baseline: {baseline[1.0]*100:>.2f}%\n'+r'$p = 1.00\%$',
                        xy=(.45, .5), xycoords='axes fraction')       
    mean, std = df["p01", "mean"].to_numpy()*100, df["p01", "std"].to_numpy()*100     
    axs[1, 1].plot(pdrops, mean, color='red')
    axs[1, 1].fill_between(pdrops, mean-std, mean+std, facecolor='red', alpha=0.2)

    for ax in axs.flat:
        ax.tick_params(direction='in', which='both')
        ax.label_outer()
    
    fig.supxlabel('Defective device probability [-]')
    fig.supylabel('Decoder test accuracy [%]')

    plt.savefig('research/plots/pdrop.pdf')

def decoder_performance_plot():

    df = pd.read_pickle('research/experiments/results/decoder_performance.pkl')
    df2 = pd.read_pickle('research/experiments/results/naive_performance.pkl')

    pfr = df.index.to_numpy()
    pfr_linspace = np.linspace(0.15, 1, 100)

    naive = df2["naive"].to_numpy()*100
    baseline_mean = df["baseline", "mean"].to_numpy()*100
    baseline_std = df["baseline", "std"].to_numpy()*100
    fp_mdnd_mean = df["fp-mdnd", "mean"].to_numpy()*100
    fp_mdnd_std = df["fp-mdnd", "std"].to_numpy()*100

    def monomial(x, a, b):
        return a * x**b

    naive_popt, _ = curve_fit(monomial, pfr, 100-naive)
    baseline_popt, _ = curve_fit(monomial, pfr, 100-baseline_mean)
    fp_mdnd_popt, _ = curve_fit(monomial, pfr, 100-fp_mdnd_mean)

    fig, ax = plt.subplots()

    ax.plot(pfr, 100-naive, marker='s', linestyle='', color='red')
    ax.plot(pfr_linspace, monomial(pfr_linspace, *naive_popt), linestyle='-', color='red',
            label=f'Naive decoder: {naive_popt[0]:.2f}*p^{naive_popt[1]:.2f}')

    ax.errorbar(pfr, 100-baseline_mean, yerr=baseline_std, marker='s', linestyle='', color='limegreen')
    ax.plot(pfr_linspace, monomial(pfr_linspace, *baseline_popt), linestyle='-', color='limegreen',
            label=f'Baseline: {baseline_popt[0]:.2f}*p^{baseline_popt[1]:.2f}')
    
    ax.errorbar(pfr, 100-fp_mdnd_mean, yerr=fp_mdnd_std, marker='s', linestyle='', color='dodgerblue')
    ax.plot(pfr_linspace, monomial(pfr_linspace, *fp_mdnd_popt), linestyle='-', color='dodgerblue',
            label=f'FP-MDND: {fp_mdnd_popt[0]:.2f}*p^{fp_mdnd_popt[1]:.2f}')

    ax.set_xlabel('Physical fault rate [%]')
    ax.set_ylabel('Logical fault rate [%]')
    ax.tick_params(direction='in', which='both')
    ax.legend()

    plt.savefig('research/plots/decoder_performance.pdf')

def weight_distribution_plot():

    MDND_LOAD_PATH = 'research/saves/hwa-mdnd/hwa_trained_mdnd_model_d3_p01_nU16_pdrop0.000-2022-07-13 04:48:07.121802.pth'

    # resistive processing unit
    rpu_config = InferenceRPUConfig()
    rpu_config.mapping = MappingParameter(digital_bias=False) # bias term is handled by the analog tile (crossbar)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # memristive deep neural decoder
    model = MDND(
        input_size=4,
        hidden_size=16,
        output_size=2,
        rpu_config=rpu_config
    ).to(device)
    # load weights (but use the current RPU config)
    model.load_state_dict(torch.load(MDND_LOAD_PATH), load_rpu_config=False)

    weights = model.get_weights()
    weights = np.concatenate([t.numpy() for t in weights], axis=None)
    std = np.std(weights)

    fig, ax = plt.subplots()
    ax.hist(weights, bins=20, density=True, histtype='bar',
            color='black')
    ax.axvline(-std, color="r", linestyle="--", label=r"$\sigma$")
    ax.axvline(std, color="r", linestyle="--")
    ax.axvline(-2*std, color="g", linestyle="--", label=r"2$\sigma$")
    ax.axvline(2*std, color="g", linestyle="--")
    ax.axvline(-3*std, color="b", linestyle="--", label=r"3$\sigma$")
    ax.axvline(3*std, color="b", linestyle="--")

    ax.set_xlabel('Weight value [-]')
    ax.set_ylabel('Normalized counts [-]')
    ax.tick_params(direction='in')
    ax.legend()

    plt.show()

def weight_clip_plot():

    MDND_LOAD_PATHS = [
        'research/saves/fp-mdnd/fp_trained_mdnd_model_d3_p0015_nU16_nR3-2022-07-11 06:06:33.570522.pth',
        'research/saves/fp-mdnd/fp_trained_mdnd_model_d3_p005_nU16_nR3-2022-07-11 07:55:53.452814.pth',
        'research/saves/fp-mdnd/fp_trained_mdnd_model_d3_p007_nU16_nR3-2022-07-11 10:43:28.346008.pth',
        'research/saves/fp-mdnd/fp_trained_mdnd_model_d3_p01_nU16_nR3-2022-07-11 12:25:29.338821.pth'
    ]

    # resistive processing unit
    rpu_config = InferenceRPUConfig()
    rpu_config.mapping = MappingParameter(digital_bias=False) # bias term is handled by the analog tile (crossbar)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # memristive deep neural decoder
    model = MDND(
        input_size=4,
        hidden_size=16,
        output_size=2,
        rpu_config=rpu_config
    ).to(device)

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # load weights (but use the current RPU config)
    model.load_state_dict(torch.load(MDND_LOAD_PATHS[0]), load_rpu_config=False)
    weights = model.get_weights()
    weights = np.concatenate([t.numpy() for t in weights], axis=None)
    std = np.std(weights)

    axs[0, 0].hist(weights, bins=20, density=True, histtype='bar',
                   color='black')
    axs[0, 0].axvline(-2.5*std, color="black", linestyle="--")
    axs[0, 0].axvline(2.5*std, color="black", linestyle="--")
    axs[0, 0].annotate(r'$p = 0.15\%$', xy=(.08, .8), xycoords='axes fraction') 

    # load weights (but use the current RPU config)
    model.load_state_dict(torch.load(MDND_LOAD_PATHS[1]), load_rpu_config=False)
    weights = model.get_weights()
    weights = np.concatenate([t.numpy() for t in weights], axis=None)
    std = np.std(weights)

    axs[0, 1].hist(weights, bins=20, density=True, histtype='bar',
                   color='black')
    axs[0, 1].axvline(-2.5*std, color="black", linestyle="--")
    axs[0, 1].axvline(2.5*std, color="black", linestyle="--")
    axs[0, 1].annotate(r'$p = 0.50\%$', xy=(.08, .8), xycoords='axes fraction')

    # load weights (but use the current RPU config)
    model.load_state_dict(torch.load(MDND_LOAD_PATHS[2]), load_rpu_config=False)
    weights = model.get_weights()
    weights = np.concatenate([t.numpy() for t in weights], axis=None)
    std = np.std(weights)

    axs[1, 0].hist(weights, bins=20, density=True, histtype='bar',
                   color='black')
    axs[1, 0].axvline(-2.5*std, color="black", linestyle="--")
    axs[1, 0].axvline(2.5*std, color="black", linestyle="--")
    axs[1, 0].annotate(r'$p = 0.70\%$', xy=(.08, .8), xycoords='axes fraction') 

    # load weights (but use the current RPU config)
    model.load_state_dict(torch.load(MDND_LOAD_PATHS[3]), load_rpu_config=False)
    weights = model.get_weights()
    weights = np.concatenate([t.numpy() for t in weights], axis=None)
    std = np.std(weights)

    axs[1, 1].hist(weights, bins=20, density=True, histtype='bar',
                   color='black')
    axs[1, 1].axvline(-2.5*std, color="black", linestyle="--")
    axs[1, 1].axvline(2.5*std, color="black", linestyle="--")
    axs[1, 1].annotate(r'$p = 1.00\%$', xy=(.08, .8), xycoords='axes fraction') 

    for ax in axs.flat:
        ax.tick_params(direction='in', which='both')
        ax.label_outer()
    
    fig.supxlabel('Weight value [-]')
    fig.supylabel('Normalized counts [-]')

    plt.savefig('research/plots/weight_clip.pdf')

def training_noise_plot():

    df = pd.read_pickle('research/experiments/results/training_noise.pkl')
    std_dev = df.index.to_numpy()*100

    decoder_performance = pd.read_pickle('research/experiments/results/decoder_performance.pkl')
    baseline = decoder_performance["baseline", "mean"]
    fp_mdnd = decoder_performance["fp-mdnd", "mean"]

    fig, ax = plt.subplots()

    ax.axhline(baseline[1.0]*100, color='k', linestyle='--')
    ax.axhline(fp_mdnd[1.0]*100, color='k', linestyle=':')
    ax.annotate(f'Baseline: {baseline[1.0]*100:>.2f}%\n'
                f'FP-MDND: {fp_mdnd[1.0]*100:>.2f}%\n'
                r'$p = 1.00\%$',
                xy=(.45, .5), xycoords='axes fraction')
    mean, std = df["p01", "mean"].to_numpy()*100, df["p01", "std"].to_numpy()*100     
    ax.plot(std_dev, mean, color='purple')
    ax.fill_between(std_dev, mean-std, mean+std, facecolor='violet', alpha=0.3)

    ax.set_xlabel('Relative training noise [%]')
    ax.set_ylabel('Decoder test accuracy [%]')
    ax.tick_params(direction='in')

    plt.show()

def training_pdrop_plot():

    df = pd.read_pickle('research/experiments/results/training_pdrop.pkl')
    pdrop = df.index.to_numpy()

    decoder_performance = pd.read_pickle('research/experiments/results/decoder_performance.pkl')
    baseline = decoder_performance["baseline", "mean"]
    fp_mdnd = decoder_performance["fp-mdnd", "mean"]

    fig, ax = plt.subplots()

    ax.axhline(baseline[1.0]*100, color='k', linestyle='--')
    ax.axhline(fp_mdnd[1.0]*100, color='k', linestyle=':')
    ax.annotate(f'Baseline: {baseline[1.0]*100:>.2f}%\n'
                f'FP-MDND: {fp_mdnd[1.0]*100:>.2f}%\n'
                r'$p = 1.00\%$',
                xy=(.45, .5), xycoords='axes fraction')
    mean, std = df["p01", "mean"].to_numpy()*100, df["p01", "std"].to_numpy()*100     
    ax.plot(pdrop, mean, color='red')
    ax.fill_between(pdrop, mean-std, mean+std, facecolor='red', alpha=0.2)

    ax.set_xlabel('Training drop connections [-]')
    ax.set_ylabel('Decoder test accuracy [%]')
    ax.tick_params(direction='in')

    plt.savefig('research/plots/training_pdrop_plot1.pdf')

def weight_hwaclip_plot():
    
    MDND_LOAD_PATHS = [
        'research/saves/hwa-mdnd/hwa_trained_mdnd_model_d3_p01_nU16_pdrop0.000-2022-07-13 04:48:07.121802.pth'
    ]

    # resistive processing unit
    rpu_config = InferenceRPUConfig()
    rpu_config.mapping = MappingParameter(digital_bias=False) # bias term is handled by the analog tile (crossbar)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # memristive deep neural decoder
    model = MDND(
        input_size=4,
        hidden_size=16,
        output_size=2,
        rpu_config=rpu_config
    ).to(device)

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # load weights (but use the current RPU config)
    model.load_state_dict(torch.load(MDND_LOAD_PATHS[0]), load_rpu_config=False)
    weights = model.get_weights()
    weights = np.concatenate([t.numpy() for t in weights], axis=None)
    std = np.std(weights)

    axs[0, 0].hist(weights, bins=20, density=True, histtype='bar',
                   color='black')
    axs[0, 0].axvline(-2.5*std, color="black", linestyle="--")
    axs[0, 0].axvline(2.5*std, color="black", linestyle="--")
    axs[0, 0].annotate(r'$p = 0.15\%$', xy=(.08, .8), xycoords='axes fraction') 

    # load weights (but use the current RPU config)
    model.load_state_dict(torch.load(MDND_LOAD_PATHS[1]), load_rpu_config=False)
    weights = model.get_weights()
    weights = np.concatenate([t.numpy() for t in weights], axis=None)
    std = np.std(weights)

    axs[0, 1].hist(weights, bins=20, density=True, histtype='bar',
                   color='black')
    axs[0, 1].axvline(-2.5*std, color="black", linestyle="--")
    axs[0, 1].axvline(2.5*std, color="black", linestyle="--")
    axs[0, 1].annotate(r'$p = 0.50\%$', xy=(.08, .8), xycoords='axes fraction')

    # load weights (but use the current RPU config)
    model.load_state_dict(torch.load(MDND_LOAD_PATHS[2]), load_rpu_config=False)
    weights = model.get_weights()
    weights = np.concatenate([t.numpy() for t in weights], axis=None)
    std = np.std(weights)

    axs[1, 0].hist(weights, bins=20, density=True, histtype='bar',
                   color='black')
    axs[1, 0].axvline(-2.5*std, color="black", linestyle="--")
    axs[1, 0].axvline(2.5*std, color="black", linestyle="--")
    axs[1, 0].annotate(r'$p = 0.70\%$', xy=(.08, .8), xycoords='axes fraction') 

    # load weights (but use the current RPU config)
    model.load_state_dict(torch.load(MDND_LOAD_PATHS[3]), load_rpu_config=False)
    weights = model.get_weights()
    weights = np.concatenate([t.numpy() for t in weights], axis=None)
    std = np.std(weights)

    axs[1, 1].hist(weights, bins=20, density=True, histtype='bar',
                   color='black')
    axs[1, 1].axvline(-2.5*std, color="black", linestyle="--")
    axs[1, 1].axvline(2.5*std, color="black", linestyle="--")
    axs[1, 1].annotate(r'$p = 1.00\%$', xy=(.08, .8), xycoords='axes fraction') 

    for ax in axs.flat:
        ax.tick_params(direction='in', which='both')
        ax.label_outer()
    
    fig.supxlabel('Weight value [-]')
    fig.supylabel('Normalized counts [-]')

    plt.savefig('research/plots/weight_hwaclip.pdf')

def conductances_plot():
    
    MDND_LOAD_PATH = 'research/saves/fp-mdnd/fp_trained_mdnd_model_d3_p01_nU16_nR3-2022-07-11 12:25:29.338821.pth'

    # model parameters
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    HIDDEN_SIZE = 16

    # resistive processing unit
    rpu_config = InferenceRPUConfig()
    rpu_config.mapping = MappingParameter(digital_bias=False) # bias term is handled by the analog tile (crossbar)
    rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=60.0, prog_noise_scale=1.) # rram noise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # memristive deep neural decoder
    analog_model = MDND(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        rpu_config=rpu_config
    ).to(device)
    # load weights (but use the current RPU config)
    analog_model.load_state_dict(torch.load(MDND_LOAD_PATH), load_rpu_config=False)

    conductances = analog_model.get_conductances()
    rnn_Gpos = conductances[0][0].numpy().T
    rnn_Gneg = conductances[0][1].numpy().T
    y, x = rnn_Gpos.shape
    vmin = min(rnn_Gpos.min(), rnn_Gneg.min())
    vmax = max(rnn_Gpos.max(), rnn_Gneg.max())

    fig, axs = plt.subplots(1, 2, sharey=True, constrained_layout=True)

    pcm0 = axs[0].pcolormesh(np.arange(x), np.arange(y), rnn_Gpos, cmap='Blues',
                            vmin=vmin, vmax=vmax)
    cbar0 = fig.colorbar(pcm0, ax=axs[0])
    axs[0].set_xlabel(r'$j$')
    axs[0].set_ylabel(r'$i$')
    cbar0.set_label(r'$g_{ij}^{+}$ [$\mu$S]')

    pcm1 = axs[1].pcolormesh(np.arange(x), np.arange(y), rnn_Gneg, cmap='Reds',
                            vmin=vmin, vmax=vmax)
    cbar1 = fig.colorbar(pcm1, ax=axs[1])
    axs[1].set_xlabel(r'$j$')
    cbar1.set_label(r'$g_{ij}^{-}$ [$\mu$S]')

    plt.savefig('research/plots/fp-conductances.pdf')

def hwa_lr_losses_plot():

    fig, ax = plt.subplots()

    df = pd.read_pickle('research/experiments/results/training/hwa_lr_losses.pkl')
    train_batches = df.index.to_numpy()

    for lr in [1e-5, 1e-4, 1e-3, 1e-2]:
        train_losses = df[lr].to_numpy()
        ax.plot(train_batches[::10], train_losses[::10], label=f"lr = {lr:.0e}")

    ax.set_xlabel("Batch [-]")
    ax.set_ylabel("Loss [-]")
    ax.legend()

    plt.show()

# dac_adc_resolution_plot()
# prog_noise_scale_plot()
# pdrop_plot()
# decoder_performance_plot()

# weight_clip_plot()

# conductances_plot()

# training_noise_plot()
training_pdrop_plot()

# weight_distribution_plot()

# hwa_lr_losses_plot()