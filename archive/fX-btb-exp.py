import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from scipy.signal import find_peaks

from seaborn import regplot
import numpy as np


def plot_figure():
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(.07, .10, .95, .98)

    grid = fig.add_gridspec(2, 1, hspace=.2, wspace=0.2)

    #fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    plot_biom_over_time(fig, grid[0])

    subgrid = grid[1].subgridspec(1, 2, wspace=.2, hspace=.1)
    
    plot_cm_vs_varmdp(fig.add_subplot(subgrid[0]))

    plot_cm_vs_varapd(fig.add_subplot(subgrid[1]))

    #plot_cm_vs_mdp(axs[1][1])

    #plot_cm_flat_vs_spont(axs[1][0])

    #for row in axs:
    #    for ax in row:
    #        ax.xaxis.get_label().set_fontsize(14)
    #        ax.yaxis.get_label().set_fontsize(14)
    #        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.savefig('./figure-pdfs/f8-btb-exp.pdf')
    plt.show()


def plot_biom_over_time(fig, grid_box):
    subgrid = grid_box.subgridspec(2, 1, wspace=.2, hspace=.1)
    ax_apd = fig.add_subplot(subgrid[0])
    ax_mdp = fig.add_subplot(subgrid[1])

    all_cells = listdir('./data/cells')

    apd90s = []
    mdps = []

    cell = '7_042921_1_alex_quinine'

    cell = '5_031221_2_alex_control'
                    #'5_031821_2_alex_verapamil',
                    #'6_033021_4_alex_control',
                    #'6_033021_5_alex_control',
                    #'6_033121_2_alex_control',
                    #'6_040821_2_alex_quinidine',
                    #'7_042621_6_alex_quinine',
                    #'7_042721_4_alex_quinine',
                    #'7_042921_1_alex_quinine']


    ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
    cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

    biomarkers = get_biomarkers(ap_dat)

    ax_apd.plot(biomarkers['apd'][0], biomarkers['apd'][1], 'k', marker='o')
    ax_mdp.plot(biomarkers['mdp'][0], biomarkers['mdp'][1], 'k', marker='o')

    ax_apd.set_ylabel(r'$APD_{90}$')
    ax_apd.set_xticklabels([])

    ax_mdp.set_xlabel('Time (ms)')
    ax_mdp.set_ylabel('MDP (mV)')

    for ax in [ax_apd, ax_mdp]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


def plot_cm_vs_varmdp(ax):
    all_cells = listdir('./data/cells')

    apd90s = []
    mdps = []

    all_cms = []
    var_mdps = []

    all_cells = ['4_021921_1_alex_control', 
                    '4_021921_2_alex_control',
                    '4_022521_2_alex_cisapride',
                    '5_031221_2_alex_control',
                    '5_031821_2_alex_verapamil',
                    '6_033021_4_alex_control',
                    '6_033021_5_alex_control',
                    '6_033121_2_alex_control',
                    '6_040821_2_alex_quinidine',
                    '7_042621_6_alex_quinine',
                    '7_042721_4_alex_quinine',
                    '7_042921_1_alex_quinine']

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        if cell == '5_031921_3_alex_verapamil':
            continue
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        biomarkers = get_biomarkers(ap_dat)

        if biomarkers is None:
            continue
        
        if len(biomarkers['apd'][0]) < 4:
            continue

        all_mdp = biomarkers['mdp'][1]

        #if np.var(all_mdp) > .3:
        #    continue
        
        norm_mdp = [100*(1 - m/all_mdp[0]) for m in all_mdp]

        all_cms.append(cell_params['Cm'].values[1])
        #var_mdps.append(np.var(all_mdp))
        var_mdps.append(np.var(norm_mdp))

    regplot(all_cms, var_mdps, color='k', ax=ax)

    ax.set_xlabel('Cm (pF)')
    ax.set_ylabel('Var(MDP) (mV)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_cm_vs_varapd(ax):
    all_cells = listdir('./data/cells')

    all_cms = []
    var_apd = []

    all_cells = ['4_021921_1_alex_control', 
                    '4_021921_2_alex_control',
                    '4_022521_2_alex_cisapride',
                    '5_031221_2_alex_control',
                    '5_031821_2_alex_verapamil',
                    '6_033021_4_alex_control',
                    '6_033021_5_alex_control',
                    '6_033121_2_alex_control',
                    '6_040821_2_alex_quinidine',
                    '7_042621_6_alex_quinine',
                    '7_042721_4_alex_quinine',
                    '7_042921_1_alex_quinine']

    for i, cell in enumerate(all_cells):
        if 'DS_Store' in cell:
            continue

        if cell == '5_031921_3_alex_verapamil':
            continue

        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        biomarkers = get_biomarkers(ap_dat)

        if biomarkers is None:
            continue
        
        all_apd = biomarkers['apd'][1]

        if len(biomarkers['apd'][0]) < 4:
            continue

        print(f'Cell {i}') 
        
        if cell_params['Cm'].values[1] < 30:
            print(cell)

        norm_apd = [100*(1 - m/all_apd[0]) for m in all_apd]


        #if np.var(all_apd) > 5000:
        #    import pdb
        #    pdb.set_trace()
        #    continue

        all_cms.append(cell_params['Cm'].values[1])
        #var_apd.append(np.var(all_apd))
        var_apd.append(np.var(norm_apd))

    regplot(all_cms, var_apd, color='k', ax=ax)

    ax.set_xlabel('Cm (pF)')
    ax.set_ylabel('Var(APD) (ms)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)









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
def get_biomarkers(ap_dat):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000
    
    if ((v.max() - v.min()) < 20):
        return None #non-spont if max-min is less than 20

    if v.max() < 0:
        return None # non-spont if peak is less than 0

    kernel_size = 100
    kernel = np.ones(kernel_size) / kernel_size
    v_smooth = np.convolve(v, kernel, mode='same')


    peak_idxs = find_peaks(np.diff(v_smooth), height=.1, distance=1000)[0]

    if len(peak_idxs) < 2:
        return None
    
    all_apd = []
    apd_times = []
    all_mdp = []
    mdp_times = []
    
    for i, pk_idx in enumerate(peak_idxs[0:-1]):
        next_idx = peak_idxs[i+1] 
        min_v = np.min(v[pk_idx:next_idx])
        min_idx = np.argmin(v[pk_idx:next_idx])
        search_space = [pk_idx, pk_idx + min_idx]
        amplitude = np.max(v[search_space[0]:search_space[1]]) - min_v
        v_90 = min_v + amplitude * .1
        idx_apd90 = np.argmin(np.abs(v[search_space[0]:search_space[1]] - v_90))

        all_mdp.append(min_v)
        mdp_times.append((pk_idx + min_idx)/10)
        all_apd.append(idx_apd90/10)
        apd_times.append((idx_apd90 + pk_idx)/10)

    return {'apd': [apd_times, all_apd],
            'mdp': [mdp_times, all_mdp]}


def main():
    plot_figure()


if __name__ == "__main__":
    main()
