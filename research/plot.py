import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import pandas as pd
from scipy.optimize import curve_fit

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
    
    axs[0, 0].axhline(baseline["p0015"]*100, color='k', linestyle='--')
    axs[0, 0].annotate(f'Baseline: {baseline["p0015"]*100:>.2f}%\n'+r'$p = 0.15\%$', xy=(15, 80))     
    axs[0, 0].errorbar(resolutions, df["p0015", "mean"].to_numpy()*100, yerr=df["p0015", "std"].to_numpy()*100,
                 marker='s', color='orange')

    axs[0, 1].axhline(baseline["p005"]*100, color='k', linestyle='--')
    axs[0, 1].annotate(f'Baseline: {baseline["p005"]*100:>.2f}%\n'+r'$p = 0.50\%$', xy=(15, 80))  
    axs[0, 1].errorbar(resolutions, df["p005", "mean"].to_numpy()*100, yerr=df["p005", "std"].to_numpy()*100,
                 marker='s', color='orange')

    axs[1, 0].axhline(baseline["p007"]*100, color='k', linestyle='--')
    axs[1, 0].annotate(f'Baseline: {baseline["p007"]*100:>.2f}%\n'+r'$p = 0.70\%$', xy=(15, 80))    
    axs[1, 0].errorbar(resolutions, df["p007", "mean"].to_numpy()*100, yerr=df["p007", "std"].to_numpy()*100,
                 marker='s', color='orange')
    
    axs[1, 1].axhline(baseline["p01"]*100, color='k', linestyle='--')
    axs[1, 1].annotate(f'Baseline: {baseline["p01"]*100:>.2f}%\n'+r'$p = 1.00\%$', xy=(15, 80))       
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
    
    axs[0, 0].axhline(baseline["p0015"]*100, color='k', linestyle='--')
    axs[0, 0].annotate(f'Baseline: {baseline["p0015"]*100:>.2f}%\n'+r'$p = 0.15\%$', xy=(1, 40))
    mean, std = df["p0015", "mean"].to_numpy()*100, df["p0015", "std"].to_numpy()*100     
    axs[0, 0].plot(noise_scales, mean, color='purple')
    axs[0, 0].fill_between(noise_scales, mean-std, mean+std, facecolor='violet', alpha=0.3)

    axs[0, 1].axhline(baseline["p005"]*100, color='k', linestyle='--')
    axs[0, 1].annotate(f'Baseline: {baseline["p005"]*100:>.2f}%\n'+r'$p = 0.50\%$', xy=(1, 40))  
    mean, std = df["p005", "mean"].to_numpy()*100, df["p005", "std"].to_numpy()*100     
    axs[0, 1].plot(noise_scales, mean, color='purple')
    axs[0, 1].fill_between(noise_scales, mean-std, mean+std, facecolor='violet', alpha=0.3)

    axs[1, 0].axhline(baseline["p007"]*100, color='k', linestyle='--')
    axs[1, 0].annotate(f'Baseline: {baseline["p007"]*100:>.2f}%\n'+r'$p = 0.70\%$', xy=(1, 40))    
    mean, std = df["p007", "mean"].to_numpy()*100, df["p007", "std"].to_numpy()*100     
    axs[1, 0].plot(noise_scales, mean, color='purple')
    axs[1, 0].fill_between(noise_scales, mean-std, mean+std, facecolor='violet', alpha=0.3)
    
    axs[1, 1].axhline(baseline["p01"]*100, color='k', linestyle='--')
    axs[1, 1].annotate(f'Baseline: {baseline["p01"]*100:>.2f}%\n'+r'$p = 1.00\%$', xy=(1, 40))       
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
    
    axs[0, 0].axhline(baseline["p0015"]*100, color='k', linestyle='--')
    axs[0, 0].annotate(f'Baseline: {baseline["p0015"]*100:>.2f}%\n'+r'$p = 0.15\%$', xy=(.25, 80))
    mean, std = df["p0015", "mean"].to_numpy()*100, df["p0015", "std"].to_numpy()*100     
    axs[0, 0].plot(pdrops, mean, color='red')
    axs[0, 0].fill_between(pdrops, mean-std, mean+std, facecolor='red', alpha=0.2)

    axs[0, 1].axhline(baseline["p005"]*100, color='k', linestyle='--')
    axs[0, 1].annotate(f'Baseline: {baseline["p005"]*100:>.2f}%\n'+r'$p = 0.50\%$', xy=(.25, 80))  
    mean, std = df["p005", "mean"].to_numpy()*100, df["p005", "std"].to_numpy()*100     
    axs[0, 1].plot(pdrops, mean, color='red')
    axs[0, 1].fill_between(pdrops, mean-std, mean+std, facecolor='red', alpha=0.2)

    axs[1, 0].axhline(baseline["p007"]*100, color='k', linestyle='--')
    axs[1, 0].annotate(f'Baseline: {baseline["p007"]*100:>.2f}%\n'+r'$p = 0.70\%$', xy=(.25, 80))    
    mean, std = df["p007", "mean"].to_numpy()*100, df["p007", "std"].to_numpy()*100     
    axs[1, 0].plot(pdrops, mean, color='red')
    axs[1, 0].fill_between(pdrops, mean-std, mean+std, facecolor='red', alpha=0.2)
    
    axs[1, 1].axhline(baseline["p01"]*100, color='k', linestyle='--')
    axs[1, 1].annotate(f'Baseline: {baseline["p01"]*100:>.2f}%\n'+r'$p = 1.00\%$', xy=(.25, 80))       
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

    pfr = df.index.to_numpy()
    pfr_linspace = np.linspace(0.15, 1, 100)

    baseline_mean = df["baseline", "mean"].to_numpy()*100
    baseline_std = df["baseline", "std"].to_numpy()*100
    fp_mdnd_mean = df["fp-mdnd", "mean"].to_numpy()*100
    fp_mdnd_std = df["fp-mdnd", "std"].to_numpy()*100

    def monomial(x, a, b):
        return a * x**b

    baseline_popt, _ = curve_fit(monomial, pfr, 100-baseline_mean)
    fp_mdnd_popt, _ = curve_fit(monomial, pfr, 100-fp_mdnd_mean)

    fig, ax = plt.subplots()

    ax.errorbar(pfr, 100-baseline_mean, yerr=baseline_std, marker='s', linestyle='', color='limegreen')
    ax.plot(pfr_linspace, monomial(pfr_linspace, *baseline_popt), linestyle='-', color='limegreen',
            label=f'Baseline: {baseline_popt[0]:.2f}*p^{baseline_popt[1]:.2f}')
    
    ax.errorbar(pfr, 100-fp_mdnd_mean, yerr=fp_mdnd_std, marker='s', linestyle='', color='dodgerblue')
    ax.plot(pfr_linspace, monomial(pfr_linspace, *fp_mdnd_popt), linestyle='-', color='dodgerblue',
            label=f'FP-MDND: {fp_mdnd_popt[0]:.2f}*p^{fp_mdnd_popt[1]:.2f}')

    ax.set_xlabel('Physical fault rate [%]')
    ax.set_xticks(pfr)
    ax.set_ylabel('Logical fault rate [%]')
    ax.tick_params(direction='in', which='both')
    ax.legend()

    plt.savefig('research/plots/decoder_performance.pdf')

def weight_distribution_plot():

    pass


dac_adc_resolution_plot()
prog_noise_scale_plot()
pdrop_plot()
decoder_performance_plot()
