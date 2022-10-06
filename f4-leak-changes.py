import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from scipy.signal import find_peaks
import matplotlib
from seaborn import histplot
import numpy as np

import myokit

from utility import VCSegment, VCProtocol


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10 
plt.rcParams['axes.labelsize'] = 10 
plt.rc('legend', fontsize = 8)


def plot_figure_gin_change():
    fig = plt.figure(figsize=(6.5, 2.75))
    fig.subplots_adjust(.07, .15, .95, .95)

    grid = fig.add_gridspec(1, 2, hspace=.2, wspace=0.3)

    #panel 1
    plot_gin_hist(fig, grid[0])

    #panel 2
    plot_gin_vs_t(fig, grid[1])

    plt.savefig('./figure-pdfs/f4.pdf')
    plt.show()


def plot_gin_hist(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    ax = fig.add_subplot(subgrid[0]) 
    ax.set_title('A', y=.94, x=-.15)

    all_cells = listdir('./data/cells')

    all_gin = []
    all_gin2 = []
    all_rin = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')
        rm = cell_params['Rm'].values[0]

        all_gin.append(1/rm*1000)
        all_gin2.append(1/cell_params['Rm'].values[1]*1000)
        all_rin.append(rm)

    print(len(all_gin))
    histplot(all_gin, ax=ax, bins=12, color='k', alpha=.5, binwidth=.5)

    ax.set_xlabel(r'$g_{in} (nS)$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    print(f'Mean: {np.mean(all_gin)}')
    print(f'Median: {np.median(all_gin)}')
    print(f'Std: {np.std(all_gin)}')
    print(f'Min: {np.min(all_gin)}')
    print(f'Max: {np.max(all_gin)}')


def plot_gin_vs_t(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    ax = fig.add_subplot(subgrid[0]) 
    ax.set_title('B', y=.94, x=-.2)

    all_cells = listdir('./data/cells')

    delta_gin = []
    delta_t = []

    for i, cell in enumerate(all_cells):
        if 'DS_Store' in cell:
            continue
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')
        rm_spont = cell_params['Rm'].values[0]
        rm_vc = cell_params['Rm'].values[1]

        st = cell_params['param_time'].values[0].minute
        end = cell_params['param_time'].values[1].minute

        minute_diff = end - st

        if minute_diff == 0:
            continue

        if minute_diff < 0:
            minute_diff = 60 - st + end

        gin_change = (1/rm_vc - 1/rm_spont) / (1/rm_spont)

        delta_gin.append(gin_change)
        delta_t.append(minute_diff)

    ax.scatter(delta_t, 100*np.array(delta_gin), color='k', marker='o')

    ax.set_xlabel(r'$\Delta Time$ (min)')
    ax.set_ylabel(r'$g_{in}$ Change (%)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    print(f'Average time change: {np.mean(delta_gin)}')
    print(f'Median time change: {np.median(delta_gin)}')
    print(f'Std time change: {np.std(delta_gin)}')
    print(f'Includes {len(delta_gin)} Cells')

    print(f'Abs Average time change: {np.mean(np.abs(delta_gin))}')
    print(f'Median time change: {np.median(np.abs(delta_gin))}')
    print(f'Std time change: {np.std(np.abs(delta_gin))}')
    print(f'Includes {len(delta_gin)} Cells')


def main():
    plot_figure_gin_change()
        

if __name__ == "__main__":
    main()
