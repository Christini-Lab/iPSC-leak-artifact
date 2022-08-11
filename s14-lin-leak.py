#!/usr/bin/env python3
#
# TODO 
#
import myokit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import methods
from methods import post, plots
from scipy.signal import find_peaks

import matplotlib

import methods.data
import methods.plots
import methods.post


def lin_leak_mod_biomarkers():
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(.07, .10, .95, .98)

    grid = fig.add_gridspec(2, 1, hspace=.1, wspace=0.2)

    indices = [0, 3, 4, 7, 10, 11, 14, 15, 17]
    norm = np.array([500, 40, 110, 155, 35, 15, 5, 30, 10, -50, -55, -40, -30, 25, 10])

    all_axs = []
    all_subgrids = []

    fs = 16

    all_biomarkers = {0: [], 1:[]}

    for i, mfile in enumerate(['./mmt/kernik_2019_mc.mmt', './mmt/paci-2013-ventricular.mmt']):
        kernik_base, p, x = myokit.load(mfile)
        s_base = myokit.Simulation(kernik_base)
        s_base.pre(15000)
        
        res_base = s_base.run(10000)

        subgrid = grid[i, 0].subgridspec(3, 6, wspace=0.7, hspace=.4)
        all_subgrids.append(subgrid)

        ax = fig.add_subplot(subgrid[0:3, 0:3])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == 1:
            ax.set_xlabel('Time (ms)', fontsize=fs)

        ax.set_ylabel('Voltage (mV)', fontsize=fs)
        all_axs.append(ax)
        t = res_base.time()
        v = res_base['membrane.V']

        single_t, single_v = get_single_ap(t, v)

        ax.plot(single_t, single_v, c=(1, .2, .2), linestyle='--')

        #GET BIOMARKERS
        trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]

        biomarkers = post.biomarkers(*trace, denoise=False)
        #ax = fig.add_subplot(subgrid[0, 6])

        biomarkers = biomarkers.mean(1)
        biomarkers = np.append(biomarkers, 0)

        all_biomarkers[i].append(biomarkers)

        #t, r = methods.plots.biomarker_polar(biomarkers[:, 0][indices], norm[indices])

        ## Biomarker plot
        #ax = fig.add_subplot(subgrid[0, 3], projection='polar', frameon=False)
        #ax.set_xticks(t[:-1])
        #ax.set_xticklabels(np.array(methods.plots.BIOM_NAMES)[indices])
        #ax.set_yticklabels([])
        #ax.set_ylim(0, 1.5)

        #ax.plot(t, r, 'r--', clip_on=False, )
        #all_axs.append(ax)


    #all_subgrids = []

    cmap = matplotlib.cm.get_cmap('viridis')
    #cmap = matplotlib.cm.get_cmap('cividis')

    all_cols = [(1, .2, .2)]

    all_leaks = np.linspace(.1, 1.5, 15)

    for num, leak in enumerate(all_leaks):
        print(leak)
        for i, modname in enumerate(['./mmt/kernik_leak.mmt', './mmt/paci-2013-ventricular-leak.mmt']):
            mod, p, x = myokit.load(modname)
            mod['membrane']['gLeak'].set_rhs(leak)
            if 'kernik' in modname:
                mod['geom']['Cm'].set_rhs(98.7)
            s_base = myokit.Simulation(mod)
            s_base.pre(15000)

            cols = cmap(num*14)
            if i == 0:
                all_cols.append(cols)

            res_base = s_base.run(10000)

            t = res_base.time()
            v = res_base['membrane.V']

            t = np.asarray(t) - 6000
            arg_t = np.argmin(np.abs(t))

            ax = all_axs[i]
                
            single_t, single_v = get_single_ap(np.array(res_base.time()), 
                                               np.array(res_base['membrane.V']))

            if ((leak >.96) and (leak < 1.02)):
                ax.plot(single_t, single_v, c='k', linestyle='--') 
            else:
                ax.plot(single_t, single_v, c=cols)

            trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]

            biomarkers = post.biomarkers(*trace, denoise=False)

            biomarkers = biomarkers.mean(1)
            biomarkers = np.append(biomarkers, 1/leak)

            all_biomarkers[i].append(biomarkers)

            #t, r = methods.plots.biomarker_polar(biomarkers[:, 0][indices], norm[indices])

            ## Biomarker plot
            #ax = all_axs[i*2+1]

            #ax.plot(t, r, c=cols, clip_on=False)


    biomarker_names = np.array(methods.plots.BIOM_NAMES)
    biomarker_names = np.append(biomarker_names, r'$R_{leak}$')[indices]

    patterns = ['-' for i in range(0, (len(all_cols)-1))]
    patterns = [''] + patterns

    for k, v in all_biomarkers.items():
        subgrid = all_subgrids[k]
        mod_biomarkers = np.array(v)
        for i in range(0, 9):
            biom_name = biomarker_names[i]
            ax = fig.add_subplot(subgrid[int(i/3), 3+i%3])
            curr_bioms = mod_biomarkers[:, indices[i]]

            bars = ax.bar(list(range(0, len(curr_bioms))), curr_bioms, color=all_cols)
            bars[0].set(hatch='-', edgecolor='white')
            #ax.bar(list(range(0, len(curr_bioms))), curr_bioms, color=all_cols)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xticklabels([])
            #ax.set_title(biom_name)
            ax.set_xlabel(biom_name)
            

    plt.show()


def lin_leak_comparison():
    all_leaks = np.linspace(.1, 2.0, 20)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(.15, .15, .95, .95, wspace=.2)
    fs = 16

    axs[0].set_xlabel(r'$APD_{90} (ms)$', fontsize=fs)
    axs[1].set_xlabel(r'$APD_{90} (ms)$', fontsize=fs)
    axs[2].set_xlabel(r'$CL$ (ms)', fontsize=fs)

    axs[0].set_ylabel(r'$Amplitude (mV)$', fontsize=fs)
    axs[1].set_ylabel(r'MDP (mV)', fontsize=fs)
    axs[2].set_ylabel(r'dV/dt (mV/ms)', fontsize=fs)
    #indices = [0, 3, 4, 7, 10, 11, 14, 15, 16]

    cell_bioms = np.array(get_cell_biomarkers())

    axs[0].scatter(cell_bioms[:, 15], cell_bioms[:, 16], color='k', marker=',')
    axs[1].scatter(cell_bioms[:, 15], cell_bioms[:, 10], color='k', marker=',')
    axs[2].scatter(cell_bioms[:, 0], cell_bioms[:, 14], color='k', marker=',')

    pub_bioms = get_pub_biomarkers()
    #axs[0].scatter(x=pub_bioms['apd90'], y=pub_bioms['apa'], marker='^')
    axs[0].errorbar(x=pub_bioms['apd90'], y=pub_bioms['apa'],
            xerr=pub_bioms['apd90_sem'], yerr=pub_bioms['apa_sem'], fmt='o')
    #axs[1].scatter(pub_bioms['bpm'], pub_bioms['dvdt'], marker='^')
    axs[1].errorbar(pub_bioms['apd90'], pub_bioms['mdp'],
            xerr=pub_bioms['apd90_sem'], yerr=pub_bioms['mdp_sem'], fmt='o')

    #axs[2].scatter(pub_bioms['apd90'], pub_bioms['mdp'], marker='^')
    cl_err = 60/(pub_bioms['bpm'] - pub_bioms['bpm_sem']) - 60/pub_bioms['bpm']
    axs[2].errorbar(1000*60/pub_bioms['bpm'], pub_bioms['dvdt'],
            xerr=cl_err*1000, yerr=pub_bioms['dvdt_sem'], fmt='o')

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    fs = 16

    all_biomarkers = {0: [], 1:[]}

    for i, mfile in enumerate(['./mmt/kernik-2019.mmt', './mmt/paci-2013-ventricular.mmt']):
        kernik_base, p, x = myokit.load(mfile)
        s_base = myokit.Simulation(kernik_base)
        s_base.pre(15000)
        
        res_base = s_base.run(10000)

        #GET BIOMARKERS
        trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]

        biomarkers = post.biomarkers(*trace, denoise=False)

        biomarkers = biomarkers.mean(1)
        biomarkers = np.append(biomarkers, 0)

        all_biomarkers[i].append(biomarkers)


    cmap = matplotlib.cm.get_cmap('viridis')


    for num, leak in enumerate(all_leaks):
        for i, modname in enumerate(['./mmt/kernik-2019-leak.mmt', './mmt/paci-2013-ventricular-leak.mmt']):
            mod, p, x = myokit.load(modname)
            mod['membrane']['g_leak'].set_rhs(leak)
            if 'kernik' in modname:
                mod['geom']['Cm'].set_rhs(98.7)
            s_base = myokit.Simulation(mod)
            s_base.pre(15000)

            cols = cmap(num*10)

            res_base = s_base.run(10000)

            trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]

            biomarkers = post.biomarkers(*trace, denoise=False)

            biomarkers = biomarkers.mean(1)

            if biomarkers[-1] < 5:
                continue

            biomarkers = np.append(biomarkers, 1/leak)

            all_biomarkers[i].append(biomarkers)


    kernik_res = np.array(all_biomarkers[0])
    paci_res = np.array(all_biomarkers[1])

    all_biomarkers = np.array(all_biomarkers)

    cols_kernik = [cmap(x*12) for x in range(0, kernik_res.shape[0]) ]
    cols_paci = [cmap(x*12) for x in range(0, paci_res.shape[0]) ]

    #axs[0].arrow(paci_res[4, 15], paci_res[4, 16],
    #                (paci_res[5, 15]-paci_res[4, 15])/2,
    #                (paci_res[5, 16]-paci_res[4, 16])/2,
    #                head_width=2,
    #                head_length=4,
    #                color='k',
    #                lw=3)
    axs[0].plot(kernik_res[:, 15], kernik_res[:, 16], 'grey', alpha=.3)
    axs[0].scatter(kernik_res[:, 15], kernik_res[:, 16], color=cols_kernik)
    axs[0].plot(paci_res[:, 15], paci_res[:, 16],
            'grey', linestyle='--', alpha=.3)
    axs[0].scatter(paci_res[:, 15], paci_res[:, 16], color=cols_paci)

    axs[1].scatter(kernik_res[:, 15], kernik_res[:, 10], color=cols_kernik)
    axs[1].plot(kernik_res[:, 15], kernik_res[:, 10], 'grey', alpha=.3)
    axs[1].scatter(paci_res[:, 15], paci_res[:, 10], color=cols_paci)
    axs[1].plot(paci_res[:, 15], paci_res[:, 10],
            'grey', linestyle='--', alpha=.3)

    axs[2].scatter(kernik_res[:, 0], kernik_res[:, 14], color=cols_kernik)
    axs[2].plot(kernik_res[:, 0], kernik_res[:, 14], 'grey', alpha=.3, label='Kernik')
    axs[2].scatter(paci_res[:, 0], paci_res[:, 14], color=cols_paci)
    axs[2].plot(paci_res[:, 0], paci_res[:, 14],
            'grey', linestyle='--', alpha=.3, label='Paci')

    plt.legend()

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Get biomarkers for all cells and plot on same axes


    plt.show()

    import pdb
    pdb.set_trace()


def comp_cond_mod_biomarkers(leak=.5):
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(.07, .10, .95, .98)

    grid = fig.add_gridspec(2, 1, hspace=.1, wspace=0.2)

    indices = [0, 3, 4, 7, 10, 11, 14, 15, 17]
    norm = np.array([500, 40, 110, 155, 35, 15, 5, 30, 10, -50, -55, -40, -30, 25, 10])

    all_axs = []
    all_subgrids = []

    fs = 16

    all_biomarkers = {0: [], 1:[]}

    for i, mfile in enumerate(['./mmt/kernik-2019.mmt', './mmt/paci-2013-ventricular.mmt']):
        kernik_base, p, x = myokit.load(mfile)
        s_base = myokit.Simulation(kernik_base)
        s_base.pre(15000)
        
        res_base = s_base.run(10000)

        subgrid = grid[i, 0].subgridspec(3, 6, wspace=0.7, hspace=.4)
        all_subgrids.append(subgrid)

        ax = fig.add_subplot(subgrid[0:3, 0:3])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == 1:
            ax.set_xlabel('Time (ms)', fontsize=fs)

        ax.set_ylabel('Voltage (mV)', fontsize=fs)
        all_axs.append(ax)
        t = res_base.time()
        v = res_base['membrane.V']

        single_t, single_v = get_single_ap(t, v)

        ax.plot(single_t, single_v, c=(1, .2, .2), linestyle='--')

        #GET BIOMARKERS
        trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]

        biomarkers = post.biomarkers(*trace, denoise=False)
        #ax = fig.add_subplot(subgrid[0, 6])

        biomarkers = biomarkers.mean(1)
        biomarkers = np.append(biomarkers, 0)

        all_biomarkers[i].append(biomarkers)

    cmap = matplotlib.cm.get_cmap('viridis')

    all_cols = [(1, .2, .2)]

    g_name = 'ik1'
    k_param_dict = {'ik1': 'ik1.g_K1',
                    'if': 'ifunny.g_f'}
    p_param_dict = {'ik1': 'ik1.g',
                    'if': 'if.g'}

    for num, cond in enumerate(np.linspace(.1, 3, 20)):
        for i, modname in enumerate(['./mmt/kernik-2019-leak.mmt', './mmt/paci-2013-ventricular-leak.mmt']):
            mod, p, x = myokit.load(modname)
            mod['membrane']['g_leak'].set_rhs(leak)

            if 'kernik' in modname:
                mod['geom']['Cm'].set_rhs(98.7)
                group, param = k_param_dict[g_name].split('.')
            else:
                group, param = p_param_dict[g_name].split('.')

            val = mod[group][param].value()
            mod[group][param].set_rhs(val*g_val)

            s_base = myokit.Simulation(mod)
            s_base.pre(15000)
            res_base = s_base.run(10000)

            trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]

            biomarkers = post.biomarkers(*trace, denoise=False)
            biomarkers = biomarkers.mean(1)

            if biomarkers[-1] < 5:
                continue

            biomarkers = np.append(biomarkers, 1/leak)
            all_biomarkers[i].append(biomarkers)

            cols = cmap(num*14)
            if i == 0:
                all_cols.append(cols)



            #---
            #t = res_base.time()
            #v = res_base['membrane.V']

            #t = np.asarray(t) - 6000
            #arg_t = np.argmin(np.abs(t))

            #ax = all_axs[i]
            #    
            #single_t, single_v = get_single_ap(np.array(res_base.time()), 
            #                                   np.array(res_base['membrane.V']))

            #ax.plot(single_t, single_v, c=cols)

            #trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]

            #biomarkers = post.biomarkers(*trace, denoise=False)

            #biomarkers = biomarkers.mean(1)
            #biomarkers = np.append(biomarkers, 1/leak)

            #all_biomarkers[i].append(biomarkers)

            ##t, r = methods.plots.biomarker_polar(biomarkers[:, 0][indices], norm[indices])

            ### Biomarker plot
            ##ax = all_axs[i*2+1]

            ##ax.plot(t, r, c=cols, clip_on=False)


    biomarker_names = np.array(methods.plots.BIOM_NAMES)
    biomarker_names = np.append(biomarker_names, r'$R_{leak}$')[indices]

    patterns = ['-' for i in range(0, (len(all_cols)-1))]
    patterns = [''] + patterns

    for k, v in all_biomarkers.items():
        subgrid = all_subgrids[k]
        mod_biomarkers = np.array(v)
        for i in range(0, 9):
            biom_name = biomarker_names[i]
            ax = fig.add_subplot(subgrid[int(i/3), 3+i%3])
            curr_bioms = mod_biomarkers[:, indices[i]]

            bars = ax.bar(list(range(0, len(curr_bioms))), curr_bioms, color=all_cols)
            bars[0].set(hatch='-', edgecolor='white')
            #ax.bar(list(range(0, len(curr_bioms))), curr_bioms, color=all_cols)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xticklabels([])
            #ax.set_title(biom_name)
            ax.set_xlabel(biom_name)
            

    plt.show()


def mod_curr_comparison(leak=.5):
    g_name = 'ik1'
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(.15, .15, .95, .95, wspace=.2)
    fs = 16

    axs[0].set_xlabel(r'$APD_{90} (ms)$', fontsize=fs)
    axs[1].set_xlabel(r'$APD_{90} (ms)$', fontsize=fs)
    axs[2].set_xlabel(r'$CL$ (ms)', fontsize=fs)

    axs[0].set_ylabel(r'$Amplitude (mV)$', fontsize=fs)
    axs[1].set_ylabel(r'MDP (mV)', fontsize=fs)
    axs[2].set_ylabel(r'dV/dt (mV/ms)', fontsize=fs)
    #indices = [0, 3, 4, 7, 10, 11, 14, 15, 16]

    cell_bioms = np.array(get_cell_biomarkers())

    axs[0].scatter(cell_bioms[:, 15], cell_bioms[:, 16], color='k', marker=',')
    axs[1].scatter(cell_bioms[:, 15], cell_bioms[:, 10], color='k', marker=',')
    axs[2].scatter(cell_bioms[:, 0], cell_bioms[:, 14], color='k', marker=',')

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    fs = 16

    all_biomarkers = {0: [], 1:[]}

    cmap = matplotlib.cm.get_cmap('viridis')
    k_param_dict = {'ik1': 'ik1.g_K1',
                    'if': 'ifunny.g_f'}
    p_param_dict = {'ik1': 'ik1.g',
                    'if': 'if.g'}

    for num, g_val in enumerate(np.linspace(.1, 3, 20)):
        for i, modname in enumerate(['./mmt/kernik-2019-leak.mmt', './mmt/paci-2013-ventricular-leak.mmt']):
            mod, p, x = myokit.load(modname)
            mod['membrane']['g_leak'].set_rhs(leak)

            if 'kernik' in modname:
                mod['geom']['Cm'].set_rhs(98.7)
                group, param = k_param_dict[g_name].split('.')
            else:
                group, param = p_param_dict[g_name].split('.')

            val = mod[group][param].value()
            mod[group][param].set_rhs(val*g_val)

            s_base = myokit.Simulation(mod)
            s_base.pre(15000)
            res_base = s_base.run(10000)

            trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]

            biomarkers = post.biomarkers(*trace, denoise=False)
            biomarkers = biomarkers.mean(1)

            if biomarkers[-1] < 5:
                continue

            biomarkers = np.append(biomarkers, 1/leak)
            all_biomarkers[i].append(biomarkers)

    kernik_res = np.array(all_biomarkers[0])
    paci_res = np.array(all_biomarkers[1])

    all_biomarkers = np.array(all_biomarkers)

    cols_kernik = [cmap(x*12) for x in range(0, kernik_res.shape[0]) ]
    cols_paci = [cmap(x*12) for x in range(0, paci_res.shape[0]) ]

    axs[0].plot(kernik_res[:, 15], kernik_res[:, 16], 'grey', alpha=.3)
    axs[0].scatter(kernik_res[:, 15], kernik_res[:, 16], color=cols_kernik)
    axs[0].plot(paci_res[:, 15], paci_res[:, 16],
            'grey', linestyle='--', alpha=.3)
    axs[0].scatter(paci_res[:, 15], paci_res[:, 16], color=cols_paci)

    axs[1].scatter(kernik_res[:, 15], kernik_res[:, 10], color=cols_kernik)
    axs[1].plot(kernik_res[:, 15], kernik_res[:, 10], 'grey', alpha=.3)
    axs[1].scatter(paci_res[:, 15], paci_res[:, 10], color=cols_paci)
    axs[1].plot(paci_res[:, 15], paci_res[:, 10],
            'grey', linestyle='--', alpha=.3)

    axs[2].scatter(kernik_res[:, 0], kernik_res[:, 14], color=cols_kernik)
    axs[2].plot(kernik_res[:, 0], kernik_res[:, 14], 'grey', alpha=.3, label='Kernik')
    axs[2].scatter(paci_res[:, 0], paci_res[:, 14], color=cols_paci)
    axs[2].plot(paci_res[:, 0], paci_res[:, 14],
            'grey', linestyle='--', alpha=.3, label='Paci')

    plt.legend()

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Get biomarkers for all cells and plot on same axes


    plt.show()

    import pdb
    pdb.set_trace()


def get_cell_biomarkers():
    cells = [f'beat{1 + i}' for i in range(9)]
    n = len(cells)

    # Load data
    print(f'Loading data for {n} cells...')
    traces = [methods.data.load_spont(cell) for cell in cells]

    # Get overlaid APs
    tmin, tmax = -80, 950
    aps = [methods.post.aps(t, v, tmin, tmax) for t, v in traces]

    # Get biomarkers – saved as list of cells with 2d array consisting of 17 biomarkers X # of APs
    biomarkers = [np.mean(methods.post.biomarkers(*trace), 1) for trace in traces]

    return biomarkers


def get_pub_biomarkers():
    authors = ['wu', 'ma11', 'doss_a', 'doss_b', 'cordiero', 'es-salah', 'ma15']
    apd90 = [367, 414.7, 324, 379.3, 277.3, 269.4, 372.9] 
    apd90_sem = [19.6, 21.8, 123.7, 145.1, 9, 30.1, 14.2]
    bpm = [44.6, 35.3, 54.4, 51.3, 60.4, 46.1, 71]
    bpm_sem = [3.9, 2.2, 30, 21, 2.6, 8.86, 5.2]

    apa = [111.26, 104, 103.3, 102.7, 101.8, 101.6, 99.1]
    apa_sem = [.99, 1.1, 12.7, 13.1, .9, 3.1, 2.6]

    mdp = [-62, -75.6, -66, -64.4, -67, -58.6, -59.8]
    mdp_sem = [3.53, 1.2, 9.4, 8.7, .8, 2.1, .7]

    dvdt = [21.7, 27.8, 24, 28.5, 29.3, 10.9, np.nan]
    dvdt_sem = [1.56, 4.8, 13.8, 17.9, 1.7, 1.2, np.nan]
    all_dat = np.array([authors, apd90, apd90_sem,
            bpm, bpm_sem, apa, apa_sem, mdp, mdp_sem, dvdt, dvdt_sem])
    cols = ['authors', 'apd90', 'apd90_sem', 'bpm',
                'bpm_sem', 'apa', 'apa_sem', 'mdp',
                'mdp_sem', 'dvdt', 'dvdt_sem']
    df = pd.DataFrame(data=np.transpose(all_dat),
        columns=cols)
    all_types = [str] + [float for x in range(0, 10)]
    convert_dict = dict(zip(cols, all_types))

    df = df.astype(convert_dict)
    return df


def get_single_ap(t, v):
    t = np.array(t)
    max_t = t[-1]

    min_v, max_v = np.min(v), np.max(v)

    if (max_v - min_v) < 10:
        sample_end = np.argmin(np.abs(t - 1100))
        new_t = t[0:sample_end] -100
        new_t = [-100, 1000]
        new_v = [v[0], v[0]]
        return new_t, new_v

    dvdt_peaks = find_peaks(np.diff(v)/np.diff(t), distance=200, width=10, prominence=.3)[0]
    start_idx = dvdt_peaks[int(len(dvdt_peaks) / 2)]
    ap_start = np.argmin(np.abs(t - t[start_idx]+100))
    ap_end = np.argmin(np.abs(t - t[start_idx]-1000))

    new_t = t[ap_start:ap_end] - t[start_idx]
    new_v = v[ap_start:ap_end]

    return new_t, new_v 




lin_leak_mod_biomarkers()

#lin_leak_comparison()

#comp_cond_mod_biomarkers()
#mod_curr_comparison()
