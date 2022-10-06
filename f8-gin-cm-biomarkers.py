import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from scipy import stats
from seaborn import regplot
import numpy as np

from utility import get_apd90, get_cl, get_dvdt

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)


def figure_gin_cm_morph():
    fig = plt.figure(figsize=(6.5, 5))

    fig.subplots_adjust(.1, .1, .96, .96)

    grid = fig.add_gridspec(2, 2, hspace=.3, wspace=.25)

    axs = []

    #Panel A
    subgrid = grid[0, 0].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_gin_vs_mdp(ax)
    axs.append(ax)

    #Panel B
    subgrid = grid[0, 1].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_gin_vs_apd(ax)
    axs.append(ax)

    #Panel C
    subgrid = grid[1, 0].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_gin_vs_cl(ax)
    axs.append(ax)

    #Panel D
    subgrid = grid[1, 1].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_gin_vs_dvdt(ax)
    axs.append(ax)

    alphas = ['A', 'B', 'C', 'D']
    for i, ax in enumerate(axs):
        ax.set_title(alphas[i], y=.94, x=-.2)
        ax.set_xlim(-5, 120)

    plt.savefig('./figure-pdfs/f8.pdf')
    plt.show()


#Panels Gin vs Morph
def plot_gin_vs_mdp(ax):
    all_cells = listdir('./data/cells')

    mdp_spont = []
    mdp_flat = []
    gin_spont = []
    gin_flat = []
    cms_spont = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        mdp = ap_dat['Voltage (V)'].min()*1000
        rm = cell_params['Rm'].values[0]

        if (
            ((ap_dat['Voltage (V)'].max() - ap_dat['Voltage (V)'].min()) > .03)
            and
            (ap_dat['Voltage (V)'].max() > 0)):
            mdp_spont.append(mdp)
            gin_spont.append(1/rm*1E6/cell_params['Cm'].values[0])
        else:
            mdp_flat.append(mdp)
            gin_flat.append(1/rm*1E6/cell_params['Cm'].values[0])

    regplot(gin_flat,mdp_flat, color='goldenrod', ax=ax, ci=None)
    ax.lines[0].set_linestyle('dotted')
    regplot(gin_spont,mdp_spont, color='k', ax=ax, ci=None)
    ax.lines[1].set_linestyle('--')

    ax.scatter(gin_spont, mdp_spont, c='k', label='Spont')
    ax.scatter(gin_flat, mdp_flat, c='goldenrod', marker='s', label='Flat')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$g_{in}$/$C_m$ (pS/pF)') 
    ax.set_ylabel('MP (mV)') 
    ax.legend()

    slope, intercept, r_value, p_value, std_err = stats.linregress(
                gin_flat,mdp_flat)
    print('Gin/Cm vs MDP')
    print(f'\tFlat MDP: {np.mean(mdp_flat)} +/- {np.std(mdp_flat)}')
    print(f'Flat: p value for Gin vs MDP is {p_value}')
    print(f'Flat: R value for Gin vs MDP is {r_value}')
    print(f'Flat: Includes {len(gin_flat)} Cells')

    slope, intercept, r_value, p_value, std_err = stats.linregress(
                gin_spont, mdp_spont)
    print(f'\tSpont MDP: {np.mean(mdp_spont)} +/- {np.std(mdp_spont)}')
    print(f'Spont: p value for Gin vs MDP is {p_value}')
    print(f'Spont: R value for Gin vs MDP is {r_value}')
    print(f'Spont: Includes {len(gin_spont)} Cells')

    print('MDP vs Gin/Cm')
    print('\t')
    print('\n')


def plot_gin_vs_apd(ax):
    all_cells = listdir('./data/cells')

    apds = []
    gins = []

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

        rm = cell_params['Rm'].values[0]
        apd = get_apd90(ap_dat)

        apds.append(apd)
        gins.append(1/rm*1E6/cell_params['Cm'].values[0])

    ax.scatter(gins, apds, color='k')

    vals_df = pd.DataFrame({'gins': gins, 'apds': apds})
    vals_df = vals_df[vals_df['gins'] < 3]

    slope, intercept, r_value, p_value, std_err = stats.linregress(
            vals_df['gins'],vals_df['apds'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$g_{in}/C_m$ (pS/pF)') 
    ax.set_ylabel(r'$APD_{90}$ (ms)') 

    print('Gin/Cm vs APD90')
    print(f'APD90 average is {np.mean(apds)} +/- {np.std(apds)}')
    print(f'p value for Gin vs APD is {p_value}')
    print(f'Includes {len(gins)} Cells')


def plot_gin_vs_cl(ax):
    all_cells = listdir('./data/cells')

    cls = []
    gins = []

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

        rm = cell_params['Rm'].values[0]
        cl = get_cl(ap_dat)

        cls.append(cl)
        gins.append(1/rm*1E6/cell_params['Cm'].values[0])

    ax.scatter(gins, cls, color='k')

    slope, intercept, r_value, p_value, std_err = stats.linregress(gins,cls)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$g_{in}$ (nS)') 
    ax.set_xlabel(r'$g_{in}/C_m$ (pS/pF)') 
    ax.set_ylabel(r'$CL$ (ms)') 

    print(f'p value for Gin vs CL is {p_value}')
    print(f'Includes {len(gins)} Cells')


def plot_gin_vs_dvdt(ax):
    all_cells = listdir('./data/cells')

    dvdts = []
    gins = []

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

        rm = cell_params['Rm'].values[0]
        dvdt = get_dvdt(ap_dat)

        dvdts.append(dvdt)
        gins.append(1/rm*1E6/cell_params['Cm'].values[0])

    ax.scatter(gins, dvdts, color='k')

    slope, intercept, r_value, p_value, std_err = stats.linregress(gins,dvdts)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$g_{in}$ (nS)') 
    ax.set_xlabel(r'$g_{in}/C_m$ (pS/pF)') 
    ax.set_ylabel(r'$dV/dt_{max}$ (V/s)') 

    print(f'p value for Rm vs dVdt is {p_value}')
    print(f'Includes {len(gins)} Cells')


def main():
    figure_gin_cm_morph()


if __name__ == "__main__":
    main()
