import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from scipy.signal import find_peaks
from scipy import stats

from utility import get_apd90

import numpy as np


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)


def figure_heterogeneity():
    fig = plt.figure(figsize=(6.5, 2.75))

    fig.subplots_adjust(.1, .15, .96, .96)

    grid = fig.add_gridspec(1, 3, hspace=.8, wspace=.4)

    axs = []

    #Panel A
    subgrid = grid[0].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_flat_spont(ax)
    axs.append(ax)

    #Panel B
    subgrid = grid[1].subgridspec(1, 4)
    ax = fig.add_subplot(subgrid[0:3])
    ax_out = fig.add_subplot(subgrid[3])
    plot_apd_vs_mdp([ax, ax_out])
    axs.append(ax)

    #Panel C
    subgrid = grid[2].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_cm_vs_gin(ax)
    axs.append(ax)


    alphas = ['A', 'B', 'C']
    for i, ax in enumerate(axs):
        if i == 1:
            ax.set_title(alphas[i], y=.94, x=-.4)
        else:
            ax.set_title(alphas[i], y=.94, x=-.2)

    plt.savefig('./figure-pdfs/f7.pdf')
    plt.show()


#Panels, Heterogeneity
def plot_flat_spont(ax):
    spont_file = '4_021921_1_alex_control'
    flat_file = '5_031121_3_alex_control'
    spont_long = '6_033021_4_alex_control'

    labels = ['Cell 1', 'Cell 2', 'Cell 3']
    st = ['grey', 'k', 'lightsteelblue']

    dvdt_to_zero = [1138, 1114, 1138]

    for i, f in enumerate([flat_file, spont_file, spont_long]):
        curr_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_spont.csv')

        times = curr_dat['Time (s)'].values[:25000] * 1000 - dvdt_to_zero[i] 
        voltages = curr_dat['Voltage (V)'].values[:25000] * 1000


        ax.plot(times, voltages, st[i], label=labels[i])

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlim(-500, 750)


def plot_apd_vs_mdp(axs):
    ax = axs[0]
    ax_out = axs[1]

    all_cells = listdir('./data/cells')

    mdps = []
    apd90s = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue

        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        if not (
            ((ap_dat['Voltage (V)'].max() - ap_dat['Voltage (V)'].min()) > .03)
            and
            (ap_dat['Voltage (V)'].max() > .01)):
            continue

        apd90 = get_apd90(ap_dat)
        
        mdps.append(ap_dat['Voltage (V)'].min()*1000)
        apd90s.append(apd90)

    print(f'Number of cells in APD vs MDP is: {len(apd90s)}')
    print(f'Average APD90: {np.mean(apd90s)}')
    print(f'Average MDP: {np.mean(mdps)}')

    ax.scatter(apd90s, mdps, color='k')
    ax_out.scatter(apd90s, mdps, color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$APD_{90}$ (ms)') 
    ax.set_ylabel('MP (mV)') 
    print(f'APD vs MDP plot includes {len(mdps)} cells')

    ax_out.spines['right'].set_visible(False)
    ax_out.spines['left'].set_visible(False)
    ax_out.spines['top'].set_visible(False)
    ax_out.set_yticklabels([])
    ax_out.set_yticks([])

    ax.set_xlim(45, 250)
    ax_out.set_xlim(395, 460)

    ax_out.set_ylim(-75, -40)
    ax.set_ylim(-75, -40)

    d = .03
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax_out.plot((1.06 - d, 1.06 + d), (-d, +d), **kwargs)

    ax.xaxis.set_label_coords(.7, -.12)

    print('MDP vs APD')
    print('\t')
    print('\n')

    slope, intercept, r_value, p_value, std_err = stats.linregress(
                mdps, apd90s)
    print('MDP vs APD90')
    print(f'p value for Gin vs MDP is {p_value}')
    print(f'R value for Gin vs MDP is {r_value}')


def plot_cm_vs_gin(ax):
    all_cells = listdir('./data/cells')

    gins = []
    cms = []
    is_ap = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')

        if not (
            ((ap_dat['Voltage (V)'].max() - ap_dat['Voltage (V)'].min()) > .03)
            and
            (ap_dat['Voltage (V)'].max() > .01)):
            is_ap.append(False)
        else:
            is_ap.append(True)

        cm = cell_params['Cm'].values[0]
        rm = cell_params['Rm'].values[0]

        cms.append(cm)
        gins.append(1/rm*1000)

    [ax.scatter(cms[i], gins[i], color='k') for i in range(0, len(is_ap)) if is_ap[i]]
    [ax.scatter(cms[i], gins[i], color='k', marker='s') for i in range(0, len(is_ap)) if not is_ap[i]]
    ax.scatter(cms, gins, color='k')
    slope, intercept, r_value, p_value, std_err = stats.linregress(cms, gins)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$C_m$ (pF)') 
    ax.set_ylabel(r'$g_{in}$ (nS)') 

    print(f'Cm vs Rm P-value is: {p_value}')
    print(f'{len(cms)} Cells were included') 


def main():
    figure_heterogeneity()


if __name__ == "__main__":
    main()
