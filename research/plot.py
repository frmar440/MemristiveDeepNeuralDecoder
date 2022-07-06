import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import pandas as pd

mpl.rcParams['axes.labelsize'] = 12


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

