import myokit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from scipy.signal import find_peaks

from methods import post, plots
import methods.data
import methods.plots
import methods.post

from utility_classes import get_single_ap


def plot_figure():
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(.07, .07, .95, .92)

    grid = fig.add_gridspec(1, 2, hspace=.15, wspace=0.2)

    plot_paci_btb(fig, grid[1])    
    plot_kernik_btb(fig, grid[0])    

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f7-btb-variability.pdf', transparent=True)

    plt.show()


def plot_kernik_btb(fig, grid_box):
    subgrid = grid_box.subgridspec(4, 1, wspace=.2, hspace=.5) 
    fs = 14

    ax = fig.add_subplot(subgrid[0:3])
    ax.set_title('Kernik', fontsize=16)

    axs = [ax]

    mfile = './mmt/kernik_leak_btb.py'
    mod, p, x = myokit.load(mfile)

    leak = 2900 
    amplitude = 150 
    period = 2

    mod['membrane']['gLeak'].set_rhs(f'1000/({amplitude}*sin(engine.time/1000/{period})+{leak})')

    s_base = myokit.Simulation(mod)
    s_base.pre(35000)

    t_max = 15000

    times = np.arange(0, t_max, 1)

    res_base = s_base.run(t_max, log_times = times)


    t = res_base.time()
    v = np.array(res_base['membrane.V'])

    t_aps, v_aps = get_all_ap(t, v)

    for i, t in enumerate(t_aps):
        ax.plot(t, v_aps[i], 'k', label='Baseline', alpha=.1)

    leaks = (amplitude*np.sin(times/1000/period)+leak)

    ax.set_ylabel('Voltage (mV)', fontsize=fs)
    ax.set_xlabel('Time (ms)', fontsize=fs)

    ax = fig.add_subplot(subgrid[3])

    axs.append(ax)
    f = '220628_010_ch2'

    nav_meta = pd.read_csv(f'data/nav-meta/{f}/NaIV_35C_0CP_meta.csv')

    times = np.linspace(0, 30, 17)

    ax.plot(times, nav_meta['rseal_Mohm'], 'k', label=f, marker='o')

    x = np.arange(0,30000,0.1)   # start,stop,step
    y = (amplitude*np.sin(x/period/1000)+leak)

    ax.plot(x/1000, y, 'grey', alpha=.5)
    
    #ax.set_ylim(0, 6000)
    ax.set_xlabel('Time (s)', fontsize=fs)
    ax.set_ylabel(r'$R_{leak}$ ($M\Omega$)', fontsize=14)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def plot_paci_btb(fig, grid_box):
    subgrid = grid_box.subgridspec(4, 1, wspace=.2, hspace=.5) 
    fs = 14

    ax = fig.add_subplot(subgrid[0:3])
    ax.set_title('Paci', fontsize=16)
    axs = [ax]

    mfile = './mmt/paci-2013-ventricular-btb.py'
    mod, p, x = myokit.load(mfile)

    leak = 775
    amplitude = 25 
    period = 2

    mod['membrane']['gLeak'].set_rhs(f'1000/({amplitude}*sin(engine.time/1000/{period})+{leak})')

    s_base = myokit.Simulation(mod)
    s_base.pre(35000)

    t_max = 30000

    times = np.arange(0, t_max, 1)

    res_base = s_base.run(t_max, log_times = times)

    t = res_base.time()
    v = np.array(res_base['membrane.V'])

    t_aps, v_aps = get_all_ap(t, v)

    for i, t in enumerate(t_aps):
        ax.plot(t, v_aps[i], 'k', label='Baseline', alpha=.3)


    ax = fig.add_subplot(subgrid[3])

    ax.set_xlabel('Time (ms)', fontsize=fs)

    axs.append(ax)

    f = '220629_006_ch4'

    nav_meta = pd.read_csv(f'data/nav-meta/{f}/NaIV_35C_40CP_meta.csv')

    times = np.linspace(0, 30, 17)

    ax.plot(times, nav_meta['rseal_Mohm'], 'k', label=f, marker='o')

    x = np.arange(0,30000,0.1)   # start,stop,step
    y = (amplitude*np.sin(x/period/1000)+leak)

    ax.plot(x/1000, y, 'grey', alpha=.5)
    
    #ax.set_ylim(0, 6000)
    ax.set_xlabel('Time (s)', fontsize=fs)
    ax.set_ylabel(r'$R_{leak}$ ($M\Omega$)', fontsize=fs)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def get_all_ap(t, v):
    pks = find_peaks(np.diff(v), distance=50, height=1)[0]

    time_array = []
    voltage_array = []

    pks = np.array(pks)-150

    for i, pk in enumerate(pks[1:-2]):
        print(pk)
        curr_times = np.array(t[pk:(pks[i+3])])
        curr_voltages = v[pk:(pks[i+3])]
        curr_times = curr_times - curr_times[0]
        time_array.append(curr_times)
        voltage_array.append(curr_voltages)

    return time_array, voltage_array


plot_figure()
