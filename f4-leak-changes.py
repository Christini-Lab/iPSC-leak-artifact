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
    fig.subplots_adjust(.07, .2, .95, .95)

    grid = fig.add_gridspec(1, 2, hspace=.2, wspace=0.3)

    #panel 1
    plot_gin_hist(fig, grid[0])
    print('one')

    #panel 2
    plot_gin_vs_t(fig, grid[1])
    print('two')

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
        all_rin.append(rm/1000)

    #histplot(all_gin, ax=ax, bins=12, color='k', alpha=.5, binwidth=.5)
    #hist, bins = np.histogram(all_rin, bins=8)
    #logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    #ax.hist(all_rin, bins=logbins)
    histplot(all_rin, ax=ax, bins=8, color='k', alpha=.5,
                                            log_scale=True)
    #ax.hist(all_rin, bins=logbins, color='k', alpha=.5, binwidth=.5)
    #ax.set_xscale('log')

    #ax.set_xlabel(r'$g_{in} (nS)$')
    ax.set_xlabel(r'$R_{in} (G\Omega)$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    labs = [1]
    labs += [None for i in range(2, 10)]
    labs += [10]

    #ax.set_xticks([0, 1, 10], labels=['0', '1', '10'])

    print(f'Mean: {np.mean(all_rin)}')
    print(f'Median: {np.median(all_rin)}')
    print(f'Std: {np.std(all_rin)}')
    print(f'Min: {np.min(all_rin)}')
    print(f'Max: {np.max(all_rin)}')


def plot_gin_vs_t(fig, grid_box):
    subgrid = grid_box.subgridspec(4, 1, wspace=.9, hspace=.1, )
    ax_out = fig.add_subplot(subgrid[0]) 
    ax = fig.add_subplot(subgrid[1:]) 
    ax_out.set_title('B', y=.7, x=-.2)

    all_cells = listdir('./data/cells')

    delta_gin = []
    delta_t = []

    j = 0

    for i, cell in enumerate(all_cells):
        if 'DS_Store' in cell:
            continue

        #print(f'Cell {j}')
        j = j + 1

        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')
        rm_spont = cell_params['Rm'].values[0]
        rm_vc = cell_params['Rm'].values[1]

        st = cell_params['param_time'].values[0].minute
        end = cell_params['param_time'].values[1].minute

        minute_diff = end - st

        #if minute_diff == 0:
        #    import pdb
        #    pdb.set_trace()
        #    continue

        if minute_diff < 0:
            minute_diff = 60 - st + end

        gin_change = (rm_vc - rm_spont) / (rm_spont)

        if gin_change > 6:
            max_gin_change = gin_change
            max_gin_t = minute_diff 

        delta_gin.append(gin_change)
        delta_t.append(minute_diff)

    ax.scatter(delta_t, 100*np.array(delta_gin), color='k', marker='o')

    ax_out.scatter(max_gin_t, 100*max_gin_change, color='k', marker='o')
    ax_out.spines['right'].set_visible(False)
    ax_out.spines['bottom'].set_visible(False)
    ax_out.spines['top'].set_visible(False)
    ax_out.set_xticklabels([])
    ax_out.set_xticks([])
    ax_out.set_ylim(1150, 1350)

    ax.set_ylim(-120, 380)

    d = .03
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax_out.plot((-d, +d), (1.03 - d, 1.03 + d), **kwargs)

    #ax.yaxis.set_label_coords(.7, -.12)


    ax.set_xlabel(r'$\Delta Time$ (min)')
    ax.set_ylabel(r'$R_{in}$ Change (%)')

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
