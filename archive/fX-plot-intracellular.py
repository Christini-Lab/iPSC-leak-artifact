import random
from math import log10
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import pickle

from multiprocessing import Pool
from deap import base, creator, tools # pip install deap
import myokit

random.seed(1)

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10 
plt.rcParams['axes.labelsize'] = 10 
plt.rc('legend', fontsize = 8)


def get_kernik_ap(param_updates={}):
    mfile = './mmt/kernik_2019_mc.mmt'
    k_mod, p, x = myokit.load(mfile)

    for param, val in param_updates.items():
        group, key = param.split('.')
        k_mod[group][key].set_rhs(val*k_mod[group][key].value())

    s_base = myokit.Simulation(k_mod)

    t_max = 100000

    times = np.arange(0, t_max, .5)

    res_base = s_base.run(t_max, log_times=times)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    return (single_t, single_v)


def get_target_ap(param_updates={'membrane.gLeak': .2}):
    mfile = './mmt/kernik_leak.mmt'
    k_mod, p, x = myokit.load(mfile)
    
    for param, val in param_updates.items():
        group, key = param.split('.')
        k_mod[group][key].set_rhs(val*k_mod[group][key].value())

    s_base = myokit.Simulation(k_mod)

    t_max = 100000

    times = np.arange(0, t_max, .5)

    res_base = s_base.run(t_max, log_times=times)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    return (single_t, single_v)


def get_single_ap(t, v):
    t = np.array(t)

    interval = t[1] - t[0]

    min_v, max_v = np.min(v), np.max(v)

    if (max_v - min_v) < 10:
        return None, None

    dvdt_peaks = find_peaks(np.diff(v)/np.diff(t), distance=20/interval, width=1/interval, prominence=.3)[0]

    if dvdt_peaks.size  < 4:
        return None, None

    start_idx = int(dvdt_peaks[-3] - 200 / interval)
    end_idx = int(dvdt_peaks[-3] + 800 / interval)

    t = t - t[start_idx]

    return t[start_idx:end_idx], v[start_idx:end_idx] 


def get_mod_response(conductances, vc_proto, f_name):

    mod = myokit.load_model(f_name)

    for cond, g in conductances.items():
        group, param = cond.split('.')
        val = mod[group][param].value()
        mod[group][param].set_rhs(val*g)

    p = mod.get('engine.pace')
    p.set_binding(None)

    v = mod.get('membrane.V')
    v.demote()
    v.set_rhs(0)
    v.set_binding('pace')

    t_max = vc_proto.characteristic_time()

    sim = myokit.Simulation(mod, vc_proto)

    times = np.arange(0, t_max, 0.1)

    dat = sim.run(t_max, log_times=times)

    return times, dat


def test_plot_bg_sodium():
    fig, axs = plt.subplots(2, 1,sharex=True, figsize=(12, 8))

    inds = pickle.load(open('./data/ga_results/inds.pkl', 'rb'))
    pop = inds[-1]

    pop.sort(key=lambda x: x.fitness.values[0])
    best_ind = pop[0]

    labs = ['original', 'best ind']

    for i, param_updates in enumerate([{}, best_ind[0]]):
        mfile = './mmt/kernik_2019_mc.mmt'
        k_mod, p, x = myokit.load(mfile)

        for param, val in param_updates.items():
            group, key = param.split('.')
            k_mod[group][key].set_rhs(val*k_mod[group][key].value())

        s_base = myokit.Simulation(k_mod)

        t_max = 500000

        times = np.arange(0, t_max, .5)

        res_base = s_base.run(t_max, log_times=times)

        t = res_base.time()
        v = res_base['membrane.V']

        axs[0].plot(t, v)
        axs[1].plot(t, res_base['nai.Nai'], label=labs[i])

    
    mfile = './mmt/kernik_leak.mmt'
    k_mod, p, x = myokit.load(mfile)

    k_mod['membrane']['gLeak'].set_rhs(.2)

    s_base = myokit.Simulation(k_mod)

    t_max = 500000

    times = np.arange(0, t_max, .5)

    res_base = s_base.run(t_max, log_times=times)

    t = res_base.time()
    v = res_base['membrane.V']

    axs[0].plot(t, v)
    axs[1].plot(t, res_base['nai.Nai'], label='Baseline+Leak')

    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('Nai (mM)')
    axs[1].set_xlabel('Time (ms)')

    axs[1].legend()
    
    plt.show()


def plot_concentrations():
    scales = [#{'membrane.gLeak': .2,
              #'ibna.g_b_Na': 1,
              #'ibca.g_b_Ca': 1
              #},
              #{'ibna.g_b_Na': 7.081649113309206,
              # 'ibca.g_b_Ca': 0.9703176460570058}, 
              #{'ibna.g_b_Na': 1,
              # 'ibca.g_b_Ca': 1},
              {},
              {'membrane.gLeak': .2,
              'ibna.g_b_Na': 1,
              'ibca.g_b_Ca': 1
              },
              {'ibna.g_b_Na': .01,
               'ibca.g_b_Ca': .01,
               'membrane.gLeak': .2}, 
               ]

    names = ['Baseline+Leak', 'Best Fit', 'Baseline']
    names = ['Baseline+Leak', 'Baseline']
    names = ['Baseline', 'Kernik+leak', 'Kernik+leak (gbNa=.01, gbCa=.01)']

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(12, 8))

    for i, mod in enumerate(['mmt/kernik_2019_mc.mmt',
                             'mmt/kernik_leak_fixed.mmt',
                             'mmt/kernik_leak_fixed.mmt',
                             #'mmt/kernik_2019_mc_fixed.mmt',
                             #'mmt/kernik_2019_mc_fixed.mmt',
                             #'mmt/paci-2013-ventricular.mmt'
                             ]):
        mod = myokit.load_model(mod)

        for name, scale in scales[i].items():
            group, param = name.split('.')
            val = mod[group][param].value()
            mod[group][param].set_rhs(val*scale)

        sim = myokit.Simulation(mod)

        t_max = 100000
        times = np.arange(0, t_max, .5)

        res_base = sim.run(t_max, log_times=times)

        axs[0].plot(times, res_base['membrane.V'], label=names[i])
        axs[1].plot(times, res_base['nai.Nai'])
        #axs[1].plot(times, res_base['sodium.Nai'])
        #axs[2].plot(times, res_base['cai.Cai'])
        #axs[2].plot(times, res_base['calcium.Cai'])
        axs[3].plot(times, res_base['ki.Ki'])
        #axs[4].plot(times, res_base['calcium.CaSR'])
        #axs[4].plot(times, res_base['casr.Ca_SR'])
        print(i)
        
        print(np.min(res_base['nai.Nai'][-10000]))
        print(np.min(res_base['ki.Ki'][-10000]))



    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[-1].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('Nai')
    axs[2].set_ylabel('Cai')
    axs[3].set_ylabel('Ki')

    axs[0].legend()
    plt.show()


def plot_paci_kernik_baseline():
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 8))

    mod = myokit.load_model('./mmt/kernik_2019_mc.mmt')

    sim = myokit.Simulation(mod)
    t_max = 100000
    times = np.arange(0, t_max, .5)
    res_base = sim.run(t_max, log_times=times)
    axs[0].plot(times, res_base['membrane.V'], label='Kernik')
    axs[1].plot(times, res_base['nai.Nai'])
    axs[2].plot(times, res_base['cai.Cai'])
    axs[3].plot(times, res_base['ki.Ki'], label='Kernik')


    mod = myokit.load_model('./mmt/paci-2013-ventricular.mmt')
    sim = myokit.Simulation(mod)
    t_max = 100000
    times = np.arange(0, t_max, .5)
    res_base = sim.run(t_max, log_times=times)
    axs[0].plot(times, res_base['membrane.V'], label='Paci')
    axs[1].plot(times, res_base['sodium.Nai'])
    axs[2].plot(times, res_base['calcium.Cai'])
    #axs[3].plot(times, res_base['potassium.Ki'])
    

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[-1].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('Nai')
    axs[2].set_ylabel('Cai')
    axs[3].set_ylabel('Ki')

    axs[0].legend()
    plt.show()


def plot_no_bg_currs():
    scales = [{},
              {'membrane.gLeak': .2,
              'ibna.g_b_Na': 1,
              'ibca.g_b_Ca': 1
              },
              {'ibna.g_b_Na': .01,
               'ibca.g_b_Ca': .01,
               'membrane.gLeak': .2}, 
               ]

    names = ['Baseline', 'Kernik+leak', 'Kernik+leak (gbNa=.01, gbCa=.01)']

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(5, 8))

    for i, mod in enumerate(['mmt/kernik_2019_mc.mmt',
                             'mmt/kernik_leak_fixed.mmt',
                             'mmt/kernik_leak_fixed.mmt',
                             ]):
        mod = myokit.load_model(mod)

        for name, scale in scales[i].items():
            group, param = name.split('.')
            val = mod[group][param].value()
            mod[group][param].set_rhs(val*scale)

        sim = myokit.Simulation(mod)

        t_max = 100000
        times = np.arange(0, t_max, .5)

        res_base = sim.run(t_max, log_times=times)

        t, v = get_single_ap(times, res_base['membrane.V'])

        ax.plot(t, v, label=names[i])


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')

    ax.legend()
    plt.show()


def main():
    #plot_concentrations()
    #plot_paci_kernik_baseline()
    plot_no_bg_currs()


if __name__ == '__main__':
    main()
