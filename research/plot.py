import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import pandas as pd

# mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams["errorbar.capsize"] = 1


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


def dac_adc_resolution():

    se = pd.read_pickle('research/experiments/results/baseline.pkl')
    df = pd.read_pickle('research/experiments/results/dac_adc_resolution.pkl')

    resolutions = df.index.to_numpy()[1:]

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    
    axs[0, 0].axhline(se["p0035"]*100, color='k', linestyle='--')
    axs[0, 0].annotate(f'Baseline: {se["p0035"]*100:>.2f}%\n'+r'$p = 0.35\%$', xy=(15, 80))     
    axs[0, 0].errorbar(resolutions, df["p0035", "mean"].to_numpy()[1:]*100, yerr=df["p0035", "std"].to_numpy()[1:]*100,
                 marker='s', color='orange')

    axs[0, 1].axhline(se["p006"]*100, color='k', linestyle='--')
    axs[0, 1].annotate(f'Baseline: {se["p006"]*100:>.2f}%\n'+r'$p = 0.60\%$', xy=(15, 80))  
    axs[0, 1].errorbar(resolutions, df["p006", "mean"].to_numpy()[1:]*100, yerr=df["p006", "std"].to_numpy()[1:]*100,
                 marker='s', color='orange')

    axs[1, 0].axhline(se["p007"]*100, color='k', linestyle='--')
    axs[1, 0].annotate(f'Baseline: {se["p007"]*100:>.2f}%\n'+r'$p = 0.70\%$', xy=(15, 80))    
    axs[1, 0].errorbar(resolutions, df["p007", "mean"].to_numpy()[1:]*100, yerr=df["p007", "std"].to_numpy()[1:]*100,
                 marker='s', color='orange')
    
    axs[1, 1].axhline(se["p01"]*100, color='k', linestyle='--')
    axs[1, 1].annotate(f'Baseline: {se["p01"]*100:>.2f}%\n'+r'$p = 1.00\%$', xy=(15, 80))       
    axs[1, 1].errorbar(resolutions, df["p01", "mean"].to_numpy()[1:]*100, yerr=df["p01", "std"].to_numpy()[1:]*100,
                 marker='s', color='orange')

    for ax in axs.flat:
        ax.tick_params(direction='in', which='both')
        ax.set_xticks(resolutions)
        ax.label_outer()
    
    fig.supxlabel('DAC/ADC resolution [bit]')
    fig.supylabel('Decoder test accuracy [%]')

    plt.savefig('research/plots/dac_adc_resolution.pdf')

dac_adc_resolution()