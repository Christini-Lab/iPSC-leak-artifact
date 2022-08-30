import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from scipy.signal import find_peaks
import matplotlib

from seaborn import histplot, regplot, pointplot, swarmplot
import numpy as np
import myokit

from utility_classes import VCSegment, VCProtocol


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10 
plt.rcParams['axes.labelsize'] = 10 
plt.rc('legend', fontsize = 8)


def plot_figure_80vs0():
    fig = plt.figure(figsize=(6.5, 2.75))
    fig.subplots_adjust(.12, .15, .99, .99)

    grid = fig.add_gridspec(1, 2, hspace=.2, wspace=0.25)

    #panel 1
    plot_gleak_effect_proto(fig, grid[0])


    #panel 2
    plot_rm_vs_rpred(fig, grid[1])

    plt.savefig('./figure-pdfs/f-80vs0.pdf')
    plt.show()


def plot_figure_rm_change():
    fig = plt.figure(figsize=(6.5, 2.75))
    fig.subplots_adjust(.07, .15, .99, .99)

    grid = fig.add_gridspec(1, 2, hspace=.2, wspace=0.3)

    #panel 1
    plot_rm_hist(fig, grid[0])

    #panel 2
    plot_rm_vs_t(fig, grid[1])


    plt.savefig('./figure-pdfs/f_rm_change.pdf')
    plt.show()


def plot_gleak_effect_proto(fig, grid_box):
    subgrid = grid_box.subgridspec(2, 1, wspace=.2, hspace=.1)

    ax_0 = fig.add_subplot(subgrid[0, 0]) 
    ax_80 = fig.add_subplot(subgrid[1, 0]) 

    protos = []
    for vhold in [0.1, -79.9]:
        proto = myokit.Protocol()
        proto.add_step(vhold, 100000)
        for i in range(0, 100):
            proto.add_step(vhold, 50)
            proto.add_step(vhold+5, 50)
        protos.append(proto)

    axs = [ax_0, ax_80]

    labels = [r'$g_f$=0.0435 nS/pF', '$g_f$=0.087 nS/pF']
    sts = ['-', '--']
    cols = ['k', 'grey']
    leak = 1

    for i, ax in enumerate(axs):
        proto = protos[i]

        for j, g_f in enumerate([1, 2]):
            t, dat = get_mod_response('./mmt/kernik_leak.mmt',
                                      {'membrane.gLeak': leak,
                                       'ifunny.g_f': g_f},
                                      vc_proto=proto)

            st = 1750
            en=750

            t = t - t[-st]

            ax.plot(t[-st:-en], dat['membrane.i_ion'][-st:-en],
                                        c=cols[j], linestyle=sts[j],
                                        label=labels[j])


    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    ax_0.set_xticklabels([])

    ax_0.set_ylabel(r'$I_{out} (0 mV)$ (A/F)')
    ax_80.set_ylabel(r'$I_{out} (-80 mV)$ (A/F)')

    ax_80.set_xlabel('Time (ms)')

    #ax_0.legend()


def plot_rm_vs_rpred(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    Cm=60

    ax = fig.add_subplot(subgrid[0]) 

    r_leaks = np.linspace(.25, 1.5, 5)
    g_leaks = [1/r for r in r_leaks]
    gf_labels = ['0.0435 nS/pF', '0.087 nS/pF']

    cols =  ['k', 'grey']

    delta_v = 5

    for it, base_v in enumerate([-80, 0]):
        base_v = base_v + .1
        proto = myokit.Protocol()
        proto.add_step(base_v, 100000)

        for i in range(0, 100):
            proto.add_step(base_v, 50)
            proto.add_step(base_v+5, 50)

        for gf in [1, 2]:
            rm_pred = []

            if gf == 1:
                col = 'k'
                st = '-'
            else:
                col = 'grey'
                st = '--'
            
            if it == 0:
                marker = 'o'
            else:
                marker = '^'
            
            for leak in g_leaks:
                t, dat = get_mod_response('./mmt/kernik_leak.mmt',
                                          {'membrane.gLeak': leak,
                                           'ifunny.g_f': gf,
                                           'ik1.g_K1': .1},
                                           vc_proto=proto)

                i_ion = dat['membrane.i_ion']
                rm = delta_v / (i_ion[-250] - i_ion[-750]) / Cm
                rm_pred.append(rm)

                print(leak)

            ax.plot(r_leaks, rm_pred,
                    c=col,
                    linestyle=st,
                    marker=marker,
                    label=f'{int(base_v-.1)}mV, {gf_labels[gf-1]}')
            #ax.scatter(r_leaks, rm_pred,
            #        c='r', marker=marker)
    
    ax.plot(r_leaks, r_leaks, 'r', linestyle='dotted', alpha=.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(r'$R_{seal} (G\Omega)$')
    ax.set_ylabel(r'$R_m (G\Omega)$')

    ax.legend()


def plot_rm_change_time(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    ax = fig.add_subplot(subgrid[0]) 

    all_cells = listdir('./data/cells')

    rm_vals = []
    pct_change = []

    for i, cell in enumerate(all_cells):
        if 'DS_Store' in cell:
            continue
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')
        rm_spont = cell_params['Rm'].values[0]
        rm_vc = cell_params['Rm'].values[1]

        if np.mod(i, 2) == 0:
            continue

        if ((rm_spont > 2500) or (rm_vc > 2500)):
            continue

        st = cell_params['param_time'].values[0].minute
        end = cell_params['param_time'].values[1].minute
        minute_diff = end - st

        if minute_diff == 0:
            continue

        if minute_diff < 0:
            minute_diff = 60 - st + end

        ax.plot([0, minute_diff], [rm_spont, rm_vc], color='k', marker='o', alpha=.4)

        rm_vals.append([rm_spont, rm_vc])

        pct_change.append((rm_vc - rm_spont)/rm_spont)

    #pct_change = [np.abs(p) for p in pct_change if np.abs(p) < .4]

    ax.set_xlabel('Time (min)')
    ax.set_ylabel(r'$R_m (M \Omega)$')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_rm_change_hist(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    ax = fig.add_subplot(subgrid[0]) 

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


def plot_rm_hist(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    ax = fig.add_subplot(subgrid[0]) 

    all_cells = listdir('./data/cells')

    all_rm = []
    all_rm_no_cutoff = []

    for cell in all_cells:
        if 'DS_Store' in cell:
            continue
        cell_params = pd.read_excel(f'./data/cells/{cell}/cell-params.xlsx')
        rm_spont = cell_params['Rm'].values[0]

        all_rm_no_cutoff.append(rm_spont)

        if (rm_spont > 2700):
            continue

        all_rm.append(rm_spont)

    print(len(all_rm))
    histplot(all_rm, ax=ax, bins=10, color='k')
    ax.axvline(np.average(all_rm), c='grey', alpha=.9, label='Average')
    ax.axvline(np.median(all_rm), c='grey', linestyle='--',
            alpha=.9, label='Median')

    ax.set_xlabel(r'$R_m (M \Omega)$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()

    print(f'Average: {np.average(all_rm_no_cutoff)}')
    print(f'Std: {np.std(all_rm_no_cutoff)}')
    print(f'Min: {np.min(all_rm_no_cutoff)}')
    print(f'Max: {np.max(all_rm_no_cutoff)}')


def plot_rm_vs_t(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    ax = fig.add_subplot(subgrid[0]) 

    all_cells = listdir('./data/cells')

    delta_rm = []
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

        rm_change = (rm_vc - rm_spont) / rm_spont

        if np.abs(rm_change) > .8:
            continue
        if rm_spont > 2700:
            continue

        delta_rm.append(rm_change)
        delta_t.append(minute_diff)

    print(len(delta_rm))

    ax.scatter(delta_t, 100*np.array(delta_rm), color='k', marker='o')

    ax.set_xlabel(r'$\Delta Time$ (min)')
    ax.set_ylabel(r'$R_m$ Change (%)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    print(f'Average time change: {np.average(np.abs(delta_rm))}')
    print(f'Median time change: {np.median(np.abs(delta_rm))}')
    print(f'Std time change: {np.std(np.abs(delta_rm))}')


#Utility function
def get_mod_response(f_name='./mmt/kernik_2019_mc.mmt',
                     conductances={},
                     vc_proto=None, tols=1E-8):
    mod = myokit.load_model(f_name)

    for cond, g in conductances.items():
        group, param = cond.split('.')
        val = mod[group][param].value()
        mod[group][param].set_rhs(val*g)

    p = mod.get('engine.pace')
    p.set_binding(None)

    if vc_proto is None:
        vc_proto = return_vc_proto()

    if 'myokit' not in vc_proto.__module__:
        proto = myokit.Protocol()

        proto.add_step(-80, 10000)

        piecewise, segment_dict, t_max = vc_proto.get_myokit_protocol()

        v = mod.get('membrane.V')
        v.demote()
        v.set_rhs(0)

        new_seg_dict = {}
        for k, vol in segment_dict.items():
            new_seg_dict[k] = vol

        segment_dict = new_seg_dict

        mem = mod.get('membrane')
        vp = mem.add_variable('vp')
        vp.set_rhs(0)
        vp.set_binding('pace')

        for v_name, st in segment_dict.items():
            v_new = mem.add_variable(v_name)
            v_new.set_rhs(st)

        v.set_rhs(piecewise)

        sim = myokit.Simulation(mod, proto)

        times = np.arange(0, t_max, 0.1)

        dat = sim.run(t_max, log_times=times)

    else:
        p = mod.get('engine.pace')
        p.set_binding(None)

        v = mod.get('membrane.V')
        v.demote()
        v.set_rhs(0)
        v.set_binding('pace')

        t_max = vc_proto.characteristic_time()

        sim = myokit.Simulation(mod, vc_proto)

        times = np.arange(0, t_max, 0.1)

        sim.set_tolerance(1E-6, tols)

        dat = sim.run(t_max, log_times=times)

    return times, dat


def main():
    plot_figure_80vs0()
    #plot_figure_rm_change()
        


if __name__ == "__main__":
    main()
