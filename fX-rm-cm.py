import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from scipy.signal import find_peaks

from seaborn import histplot, regplot, pointplot, swarmplot
import numpy as np



def plot_figure():
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(.07, .10, .95, .98)

    plot_flat_spont(axs[0][0])

    #plot_rm_change(axs[1][0])
    plot_apd_vs_mdp(axs[0][1])

    plot_cm_vs_mdp(axs[1][1])

    plot_cm_flat_vs_spont(axs[1][0])

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

    ax.set_xlim(-500, 1000)

    ax.legend(loc=1)


def plot_apd_vs_mdp(ax):
    all_cells = listdir('./data/cells')

    mdps = []
    apd90s = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        apd90 = get_apd90(ap_dat)

        
        if apd90 is not None:
            if apd90 > 300:
                continue
            mdps.append(ap_dat['Voltage (V)'].min()*1000)
            apd90s.append(apd90)

    ax.scatter(apd90s, mdps, color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$APD_{90}$') 
    ax.set_ylabel('MDP') 


def plot_rm_change(ax):
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


def plot_cm_vs_mdp(ax):
    all_cells = listdir('./data/cells')

    mdps= []
    cms= []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        mdps.append(ap_dat['Voltage (V)'].min()*1000)
        cms.append(cell_params['Cm'].values[0])



    #ax.scatter(cms_flat, mdps_flat, c='k', label='Flat')
    #ax.scatter(cms, mdps, c='grey', marker='^', label='Spont')
    regplot(cms, mdps, color='k', ax=ax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$C_m$') 
    ax.set_ylabel('MDP') 


def plot_cm_flat_vs_spont(ax):
    all_cells = listdir('./data/cells')

    cms = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        if (
            ((ap_dat['Voltage (V)'].max() - ap_dat['Voltage (V)'].min()) > .03)
            and
            (ap_dat['Voltage (V)'].max() > 0)):
            cc_behavior = 'Spont'
        else:
            cc_behavior = 'Flat'

        cms.append([cc_behavior, cell_params['Cm'].values[0]])

    all_cms = pd.DataFrame(cms, columns=['type', 'Cm'])
            
    swarmplot(x='type', y='Cm', data=all_cms, size=9, color='grey', ax=ax, zorder=1)
    pointplot(x='type', y='Cm', data=all_cms, join=False, capsize=.05, markers='_', ax=ax, color='k', ci='sd')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('') 
    ax.set_ylabel('Cm') 
    ax.set_ylim(0, 105)


#Utility function
def get_apd90(ap_dat):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000
    
    if ((v.max() - v.min()) < 20):
        return None
    if v.max() < 0:
        return None

    kernel_size = 100
    kernel = np.ones(kernel_size) / kernel_size
    v_smooth = np.convolve(v, kernel, mode='same')

    #plt.plot(t, v)
    #plt.plot(t, v_smooth)
    #plt.plot(np.diff(v_smooth))

    peak_idxs = find_peaks(np.diff(v_smooth), height=.1, distance=1000)[0]

    if len(peak_idxs) < 2:
        return None

    min_v = np.min(v[peak_idxs[0]:peak_idxs[1]])
    min_idx = np.argmin(v[peak_idxs[0]:peak_idxs[1]])
    search_space = [peak_idxs[0], peak_idxs[0] + min_idx]
    amplitude = np.max(v[search_space[0]:search_space[1]]) - min_v
    v_90 = min_v + amplitude * .1
    idx_apd90 = np.argmin(np.abs(v[search_space[0]:search_space[1]] - v_90))

    #plt.axvline(t[idx_apd90+search_space[0]])
    

    #plt.plot(t[search_space[0]:search_space[1]], v[search_space[0]:search_space[1]])

    return idx_apd90 / 10



def main():
    plot_figure()


if __name__ == "__main__":
    main()
