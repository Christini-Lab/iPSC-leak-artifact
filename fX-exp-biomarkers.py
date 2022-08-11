import matplotlib.pyplot as plt
import pandas as pd
from os import listdir

from seaborn import histplot, regplot, pointplot, swarmplot
import numpy as np



def plot_figure():
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(.07, .10, .95, .98)

    #plot_flat_spont(axs[0][0])

    #plot_cm_vs_mdp(axs[0][1])

    plot_rm_change_time(axs[1][0])
    plot_rm_change_hist(axs[1][1])

    for row in axs:
        for ax in row:
            ax.xaxis.get_label().set_fontsize(14)
            ax.yaxis.get_label().set_fontsize(14)
            ax.tick_params(axis='both', which='major', labelsize=12)

    plt.savefig('./figure-pdfs/f3-exp-cm-rm.pdf')
    plt.show()


def plot_flat_spont(ax):
    spont_file = '4_021921_1_alex_control'
    flat_file = '5_031121_3_alex_control'

    labels = ['Cell 1', 'Cell 2']
    st = ['grey', 'k']

    for i, f in enumerate([flat_file, spont_file]):
        curr_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_spont.csv')

        times = curr_dat['Time (s)'].values[:20000] * 1000
        voltages = curr_dat['Voltage (V)'].values[:20000] * 1000


        ax.plot(times, voltages, st[i], label=labels[i])

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.legend(loc=1)


def plot_rm_change_time(ax):
    all_cells = listdir('./data/cells')

    delta_rm = []

    for i, cell in enumerate(all_cells):
        if 'DS_Store' in cell:
            continue
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')
        rm_spont = cell_params['Rm'].values[0]
        rm_vc = cell_params['Rm'].values[1]

        if np.mod(i, 2) == 0:
            print('hi')
            continue
        
        if ((rm_spont > 1500) or (rm_vc > 1500)):
            continue
        
        st = cell_params['param_time'].values[0].minute 
        end = cell_params['param_time'].values[1].minute
        minute_diff = end - st

        if minute_diff == 0:
            continue
        
        if minute_diff < 0:
            minute_diff = 60 - st + end

        ax.plot([0, minute_diff], [rm_spont, rm_vc], color='k', marker='o', alpha=.4)

    ax.set_xlabel('Time (min)')
    ax.set_ylabel(r'$\Delta R_m (M \Omega)$') 


def plot_rm_change_hist(ax):
    all_cells = listdir('./data/cells')

    delta_rm = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')
        rm_spont = cell_params['Rm'].values[0]
        rm_vc = cell_params['Rm'].values[1]

        
        if ((rm_spont > 2500) or (rm_vc > 2500)):
            continue

        rm_change = rm_vc - rm_spont

        delta_rm.append(rm_change)

    print(len(delta_rm))
    histplot(delta_rm, ax=ax, color='k')
    #ax.axvline(0, c='grey', alpha=.2)
    ax.axvline(np.average(delta_rm), c='grey', alpha=.9, label='Average')
    ax.axvline(np.median(delta_rm), c='grey', linestyle='--',
            alpha=.9, label='Median')

    ax.set_xlabel(r'$\Delta R_m (M \Omega)$') 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()





def main():
    plot_figure()


if __name__ == "__main__":
    main()
