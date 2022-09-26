import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from scipy.signal import find_peaks
from scipy import stats

from seaborn import histplot, regplot, pointplot, swarmplot
import numpy as np


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)


def figure_all():
    fig = plt.figure(figsize=(6.5, 7.5))

    fig.subplots_adjust(.1, .08, .96, .96)

    grid = fig.add_gridspec(3, 1, hspace=.4)

    axs = []

    #Panel A
    subgrid = grid[0].subgridspec(1,2, wspace=.3)
    ax = fig.add_subplot(subgrid[0])
    plot_flat_spont(ax)
    axs.append(ax)

    #Panel B
    ax = fig.add_subplot(subgrid[1])
    plot_cm_vs_rm(ax)
    axs.append(ax)


    #Panel C
    subgrid = grid[1].subgridspec(1, 4, wspace=.6)
    #ax = fig.add_subplot(subgrid[0])
    #plot_rm_flat_vs_spont(ax)
    #axs.append(ax)

    #Panel D
    ax = fig.add_subplot(subgrid[0])
    plot_rm_vs_mdp(ax)
    axs.append(ax)

    #Panel E
    ax = fig.add_subplot(subgrid[1])
    plot_rm_vs_apd(ax)
    axs.append(ax)

    #Panel F
    #subgrid = grid[2].subgridspec(1, 3, wspace=.4)
    #ax = fig.add_subplot(subgrid[0])
    #plot_cm_flat_vs_spont(ax)
    #axs.append(ax)

    #Panel G
    ax = fig.add_subplot(subgrid[2])
    plot_cm_vs_mdp(ax)
    axs.append(ax)

    #Panel H
    ax = fig.add_subplot(subgrid[3])
    plot_cm_vs_apd(ax)
    axs.append(ax)


    alphas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for i, ax in enumerate(axs):
        ax.set_title(alphas[i], y=.94, x=-.2)

    plt.savefig('./figure-pdfs/f-exp-cm-rmv2.pdf')
    plt.show()


def figure_cm_rm():
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
    subgrid = grid[1].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_apd_vs_mdp(ax)
    axs.append(ax)

    #Panel C
    subgrid = grid[2].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_cm_vs_gin(ax)
    axs.append(ax)


    alphas = ['A', 'B', 'C']
    for i, ax in enumerate(axs):
        ax.set_title(alphas[i], y=.94, x=-.2)

    plt.savefig('./figure-pdfs/f-exp-cm-rm-corr.pdf')
    plt.show()


def figure_rm_morph():
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

    plt.savefig('./figure-pdfs/f-rm-vs-morph.pdf')
    plt.show()


def figure_cm_morph():
    fig = plt.figure(figsize=(6.5, 5))

    fig.subplots_adjust(.1, .1, .96, .96)

    grid = fig.add_gridspec(2, 2, hspace=.3, wspace=.25)

    axs = []

    #Panel A
    subgrid = grid[0, 0].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_cm_vs_mdp(ax)
    axs.append(ax)

    #Panel B
    subgrid = grid[0, 1].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_cm_vs_apd(ax)
    axs.append(ax)

    #Panel C
    subgrid = grid[1, 0].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_cm_vs_cl(ax)
    axs.append(ax)

    #Panel D
    subgrid = grid[1, 1].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_cm_vs_dvdt(ax)
    axs.append(ax)

    alphas = ['A', 'B', 'C', 'D']
    for i, ax in enumerate(axs):
        ax.set_title(alphas[i], y=.94, x=-.2)

    plt.savefig('./figure-pdfs/f-cm-vs-morph.pdf')
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

    ax.set_xlim(-500, 1000)

    ax.legend(loc=1)


def plot_apd_vs_mdp(ax):
    all_cells = listdir('./data/cells')

    mdps = []
    apd90s = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue

        if '6_033021_4_alex_control' == cell:
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

    ax.scatter(apd90s, mdps, color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$APD_{90}$ (ms)') 
    ax.set_ylabel('MDP (mV)') 
    print(f'APD vs MDP plot includes {len(mdps)} cells')


def plot_cm_vs_gin(ax):
    all_cells = listdir('./data/cells')

    gins = []
    cms = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        if '6_033021_4_alex_control' == cell:
            continue
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        cm = cell_params['Cm'].values[0]
        rm = cell_params['Rm'].values[0]

        cms.append(cm)
        gins.append(1/rm*1000)

    ax.scatter(cms, gins, color='k')
    slope, intercept, r_value, p_value, std_err = stats.linregress(cms, gins)
    #regplot(cms, gins, color='k', ax=ax, line_kws={'label':f'$R$={round(r_value, 3)}; p={round(p_value, 2)}'})
    #regplot(cms, rms, color='k', ax=ax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$C_m$ (pF)') 
    ax.set_ylabel(r'$g_{in}$ (nS)') 
    #ax.legend()

    print(f'Cm vs Rm P-value is: {p_value}')
    print(f'{len(cms)} Cells were included') 


#Panels Gin vs Morph
def plot_gin_vs_mdp(ax):
    all_cells = listdir('./data/cells')

    mdp_spont = []
    mdp_flat = []
    gin_spont = []
    gin_flat = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        if '6_033021_4_alex_control' == cell:
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
            gin_spont.append(1/rm*1000)
        else:
            mdp_flat.append(mdp)
            gin_flat.append(1/rm*1000)

    gins = np.concatenate([gin_spont, gin_flat])
    mdps = mdp_spont + mdp_flat 

    vals_df = pd.DataFrame({'gins': gins, 'mdps': mdps})
    vals_df = vals_df[vals_df['gins'] < 3]

    #print(f'')

    slope, intercept, r_value, p_value, std_err = stats.linregress(
                vals_df['gins'],vals_df['mdps'])
    #regplot(vals_df['gins'],vals_df['mdps'], color='k', ax=ax,
    #        line_kws={'label':f'$R$={round(r_value, 3)}; p={round(p_value, 2)}'})
    regplot(vals_df['gins'],vals_df['mdps'], color='k', ax=ax, ci=None)
    ax.lines[0].set_linestyle('--')

    ax.scatter(gin_spont, mdp_spont, c='k', label='Spont')
    ax.scatter(gin_flat, mdp_flat, c='goldenrod', marker='s', label='Flat')
    ax.axvspan(0, vals_df['gins'].max(), color='grey', alpha=.1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$g_{in}$ (nS)') 
    ax.set_ylabel('MDP (mV)') 
    ax.legend()
    print(f'p value for Gin vs MDP is {p_value}')
    print(f'R value for Gin vs MDP is {r_value}')
    print(f'Includes {len(gins)} Cells')


def plot_gin_vs_apd(ax):
    all_cells = listdir('./data/cells')

    apds = []
    gins = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        if '6_033021_4_alex_control' == cell:
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

        if apd is not None:
            if apd > 300:
                continue

        apds.append(apd)
        gins.append(1/rm*1000)

    ax.scatter(gins, apds, color='k')

    vals_df = pd.DataFrame({'gins': gins, 'apds': apds})
    vals_df = vals_df[vals_df['gins'] < 3]

    slope, intercept, r_value, p_value, std_err = stats.linregress(
            vals_df['gins'],vals_df['apds'])
    #regplot(gins, apds, color='k', ax=ax,
    #        line_kws={'label':f'$R$={round(r_value, 3)}; p={round(p_value, 2)}'})
    #regplot(vals_df['gins'],vals_df['apds'], color='k', ax=ax)#, ci=None)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$g_{in}$ (nS)') 
    ax.set_ylabel(r'$APD_{90}$ (ms)') 

    #ax.axvspan(0, vals_df['gins'].max(), color='grey', alpha=.1)

    print(f'p value for Gin vs APD is {p_value}')
    print(f'Includes {len(gins)} Cells')


def plot_gin_vs_cl(ax):
    all_cells = listdir('./data/cells')

    cls = []
    gins = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        if '6_033021_4_alex_control' == cell:
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
        gins.append(1/rm*1000)

    ax.scatter(gins, cls, color='k')

    slope, intercept, r_value, p_value, std_err = stats.linregress(gins,cls)
    #regplot(gins, cls, color='k', ax=ax,
    #        line_kws={'label':f'$R$={round(r_value, 3)}; p={round(p_value, 2)}'})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$g_{in}$ (nS)') 
    ax.set_ylabel(r'$CL$ (ms)') 

    #ax.legend()

    print(f'p value for Gin vs CL is {p_value}')
    print(f'Includes {len(gins)} Cells')


def plot_gin_vs_dvdt(ax):
    all_cells = listdir('./data/cells')

    dvdts = []
    gins = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        if '6_033021_4_alex_control' == cell:
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
        gins.append(1/rm*1000)

    ax.scatter(gins, dvdts, color='k')

    slope, intercept, r_value, p_value, std_err = stats.linregress(gins,dvdts)
    #regplot(gins, dvdts, color='k', ax=ax,
    #        line_kws={'label':f'$R$={round(r_value, 3)}; p={round(p_value, 2)}'})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$g_{in}$ (nS)') 
    ax.set_ylabel(r'$dV/dt_{max}$ (V/s)') 

    #ax.legend()

    print(f'p value for Rm vs dVdt is {p_value}')
    print(f'Includes {len(gins)} Cells')


#Panels Cm vs Morph
def plot_cm_vs_mdp(ax):
    all_cells = listdir('./data/cells')

    mdps_spont = []
    cms_spont = []
    mdps_flat = []
    cms_flat = []
    gin_spont = []
    gin_flat = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        if '6_033021_4_alex_control' == cell:
            continue
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        rm = cell_params['Rm'].values[0]

        if (
            ((ap_dat['Voltage (V)'].max() - ap_dat['Voltage (V)'].min()) > .03)
            and
            (ap_dat['Voltage (V)'].max() > 0)):
            mdps_spont.append(ap_dat['Voltage (V)'].min()*1000)
            cms_spont.append(cell_params['Cm'].values[0])
            gin_spont.append(1/rm*1000)
        else:
            mdps_flat.append(ap_dat['Voltage (V)'].min()*1000)
            cms_flat.append(cell_params['Cm'].values[0])
            gin_flat.append(1/rm*1000)


    cms = cms_spont + cms_flat
    mdps = mdps_spont + mdps_flat 
    gins = np.concatenate([gin_spont, gin_flat])

    vals_df = pd.DataFrame({'gins': gins, 'cms': cms, 'mdps': mdps})
    vals_df = vals_df[vals_df['gins'] < 3]

    slope, intercept, r_value, p_value, std_err = stats.linregress(
            vals_df['cms'],vals_df['mdps'])
    regplot(vals_df['cms'],vals_df['mdps'], color='k', ax=ax, ci=None)
    ax.lines[0].set_linestyle('--')
    #regplot(cms, mdps, color='k', ax=ax)
    ax.scatter(cms_spont, mdps_spont, c='k', label='Spont')
    ax.scatter(cms_flat, mdps_flat, c='goldenrod', marker='s', label='Flat')

    cms = cms_spont + cms_flat
    mdps = mdps_spont + mdps_flat 
    gins = np.concatenate([gin_spont, gin_flat])
    vals_df = pd.DataFrame({'gins': gins, 'cms': cms, 'mdps': mdps})
    vals_df = vals_df[vals_df['gins'] > 3]
    ax.scatter(vals_df['cms'], vals_df['mdps'], c='r', marker='x', s=35, label='Excluded', linewidths=2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$C_m (pF)$') 
    ax.set_ylabel('MDP (mV)') 
    ax.legend()

    print(f'p value for Cm vs MDP is {p_value}')
    print(f'R value for Cm vs MDP is {r_value}')


def plot_cm_vs_apd(ax):
    all_cells = listdir('./data/cells')

    apds = []
    cms = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        if '6_033021_4_alex_control' == cell:
            continue
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        if not (
            ((ap_dat['Voltage (V)'].max() - ap_dat['Voltage (V)'].min()) > .03)
            and
            (ap_dat['Voltage (V)'].max() > .01)):
            continue

        cm = cell_params['Cm'].values[0]

        apd = get_apd90(ap_dat)

        if apd is not None:
            if apd > 300:
                continue

        apds.append(apd)
        cms.append(cm)

    ax.scatter(cms, apds, color='k')
    slope, intercept, r_value, p_value, std_err = stats.linregress(cms,apds)
    #regplot(cms, apds, color='k', ax=ax,
    #        line_kws={'label':f'$R$={round(r_value, 3)}; p={round(p_value, 2)}'})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$C_m$ (pF)') 
    ax.set_ylabel(r'$APD_{90}$ (ms)') 
    print(f'p value for Cm vs APD is {p_value}')


def plot_cm_vs_cl(ax):
    all_cells = listdir('./data/cells')

    cls = []
    cms = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        if '6_033021_4_alex_control' == cell:
            continue
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        if not (
            ((ap_dat['Voltage (V)'].max() - ap_dat['Voltage (V)'].min()) > .03)
            and
            (ap_dat['Voltage (V)'].max() > .01)):
            continue

        cm = cell_params['Cm'].values[0]

        cl = get_cl(ap_dat)

        cls.append(cl)
        cms.append(cm)

    ax.scatter(cms, cls, color='k')
    slope, intercept, r_value, p_value, std_err = stats.linregress(cms,cls)
    #regplot(cms, cls, color='k', ax=ax,
    #        line_kws={'label':f'$R$={round(r_value, 3)}; p={round(p_value, 2)}'})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$C_m$ (pF)') 
    ax.set_ylabel(r'CL (ms)') 
    print(f'p value for Cm vs CL is {p_value}')


def plot_cm_vs_dvdt(ax):
    all_cells = listdir('./data/cells')

    dvdts = []
    cms = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        if '6_033021_4_alex_control' == cell:
            continue
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        if not (
            ((ap_dat['Voltage (V)'].max() - ap_dat['Voltage (V)'].min()) > .03)
            and
            (ap_dat['Voltage (V)'].max() > .01)):
            continue

        cm = cell_params['Cm'].values[0]

        dvdt = get_dvdt(ap_dat)

        dvdts.append(dvdt)
        cms.append(cm)

    ax.scatter(cms, dvdts, color='k')
    slope, intercept, r_value, p_value, std_err = stats.linregress(cms,dvdts)
    #regplot(cms, dvdts, color='k', ax=ax,
    #        line_kws={'label':f'$R$={round(r_value, 3)}; p={round(p_value, 2)}'})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'$C_m$ (pF)') 
    ax.set_ylabel(r'$dV/dt_{max}$ (V/s)') 
    print(f'p value for Cm vs dV/dt is {p_value}')


#Spont vs Flat
def plot_cm_flat_vs_spont(ax):

    all_cells = listdir('./data/cells')

    cms = []

    cm_flat = []
    cm_spont = []

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
            cm_spont.append(cell_params['Cm'].values[0])
        else:
            cc_behavior = 'Flat'
            cm_flat.append(cell_params['Cm'].values[0])

        cms.append([cc_behavior, cell_params['Cm'].values[0]])

    all_cms = pd.DataFrame(cms, columns=['type', 'Cm'])
            
    swarmplot(x='type', y='Cm', data=all_cms, color='grey', ax=ax, zorder=1)
    pointplot(x='type', y='Cm', data=all_cms, join=False, capsize=.05, markers='_', ax=ax, color='k', ci='sd')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('') 
    ax.set_ylabel(r'$C_m (pF)$') 
    ax.set_ylim(0, 105)


def plot_rm_flat_vs_spont(ax):
    all_cells = listdir('./data/cells')

    cms = []

    rm_flat = []
    rm_spont = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        ap_dat = pd.read_csv(f'./data/cells/{cell}/Pre-drug_spont.csv')
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')

        rm = cell_params['Rm'].values[0]

        if rm > 4000:
            continue

        if (
            ((ap_dat['Voltage (V)'].max() - ap_dat['Voltage (V)'].min()) > .03)
            and
            (ap_dat['Voltage (V)'].max() > 0)):
            cc_behavior = 'Spont'
            rm_spont.append(cell_params['Rm'].values[0])
        else:
            cc_behavior = 'Flat'
            rm_flat.append(cell_params['Rm'].values[0])

        cms.append([cc_behavior, cell_params['Rm'].values[0]])

    all_cms = pd.DataFrame(cms, columns=['type', 'Rm'])
            
    swarmplot(x='type', y='Rm', data=all_cms, color='grey', ax=ax, zorder=1)
    pointplot(x='type', y='Rm', data=all_cms, join=False, capsize=.05, markers='_', ax=ax, color='k', ci='sd')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('') 
    ax.set_ylabel(r'$R_m (G\Omega)$') 

    print(f'Spontaneous average is {np.average(rm_spont)}+/-{np.std(rm_spont)}')
    print(f'Flat average is {np.average(rm_flat)}+/-{np.std(rm_flat)}')
    print(f'Rm Spont vs Flat ttest: {stats.ttest_ind(rm_spont, rm_flat)}')


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


def get_cl(ap_dat):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000

    peak_pts = find_peaks(v, 10, distance=1000, width=200)[0]

    average_cl = np.mean(np.diff(peak_pts)) / 10

    return average_cl


def get_dvdt(ap_dat):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000


    #plt.plot(t, v)
    
    peak_pts = find_peaks(v, 10, distance=1000, width=200)[0]

    #plt.plot(t, v)

    new_v = moving_average(v, 10)
    new_t = moving_average(t, 10)

    #plt.plot(np.diff(new_v))

    v_diff = np.diff(new_v)

    dvdt_maxs = []

    for peak_pt in peak_pts:
        start_pt = int(peak_pt/10-50)
        end_pt = int(peak_pt/10)
        dvdt_maxs.append(np.max(v_diff[start_pt:end_pt]))

        #plt.axvline(peak_pt/10, -50, 20, c='c')
        #plt.axvline(peak_pt/10-50, -50, 20, c='r')

    average_dvdt = np.mean(dvdt_maxs)

    return average_dvdt 


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs]
    return np.array(new_vals)


def main():
    #plot_figure()
    #figure_cm_rm()
    #figure_rm_morph()
    figure_cm_morph()


if __name__ == "__main__":
    main()
