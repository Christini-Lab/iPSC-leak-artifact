import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from scipy.signal import find_peaks
import matplotlib

from seaborn import histplot, regplot, pointplot, swarmplot
import numpy as np
import myokit

from utility_classes import VCSegment, VCProtocol



def plot_figure():
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(.07, .10, .95, .98)

    grid = fig.add_gridspec(2, 2, hspace=.2, wspace=0.2)

    #panel 1
    #plot_gleak_effect_proto(fig, grid[0, 0])


    #panel 2
    #plot_rm_vs_rpred(fig, grid[0, 1])


    #panel 3 
    plot_rm_change_time(fig, grid[1, 0])
    

    #panel 4
    plot_rm_change_hist(fig, grid[1, 1])



    #plt.savefig('./figure-pdfs/f4-leak_changes.pdf')
    plt.show()


def plot_gleak_effect_proto(fig, grid_box):
    subgrid = grid_box.subgridspec(3, 1, wspace=.2, hspace=.1)

    ax_v1 = fig.add_subplot(subgrid[0, 0]) 
    ax_c1 = fig.add_subplot(subgrid[1, 0]) 
    ax_lk1 = fig.add_subplot(subgrid[2, 0]) 

    axs = [ax_v1, ax_c1, ax_lk1]

    leak = 1 
    cols = ['k', 'grey']
    styles = ['-', '--']

    vhold = 0.1
    segments = [VCSegment(50000, vhold)]
    for i in range(0, 125):
        segments.append(VCSegment(100, vhold))
        segments.append(VCSegment(100, vhold+1))

    py_proto = VCProtocol(segments)

    for j, g_f in enumerate([1, 2]):
        t, dat = get_mod_response('./mmt/kernik_leak.mmt',
                                  {'membrane.gLeak': leak,
                                   'ifunny.g_f': g_f},
                                  vc_proto=py_proto)
        t = t - t[-2500]

        axs[0].plot(t[-2500:-500], dat['membrane.V'][-2500:-500],
                                                    c=cols[j], linestyle=styles[j], label=f'$G_f$={g_f}')

        axs[1].plot(t[-2500:-500], dat['membrane.i_ion'][-2500:-500],
                                                    c=cols[j], linestyle=styles[j], label=f'G_f={g_f}')
        ion_leak = np.array(dat['membrane.i_ion']) - np.array(dat['membrane.ILeak'])
        axs[2].plot(t[-2500:-500], ion_leak[-2500:-500], c=cols[j], linestyle=styles[j])

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    for ax in [ax_v1, ax_c1]:
        ax.set_xticklabels([])

    ax_v1.set_ylabel(r'$V_{cmd}$')
    ax_c1.set_ylabel(r'$I_{out}$ (A/F)')
    ax_lk1.set_ylabel(r'$I_{out}$ - $I_{leak}$ (A/F)')

    ax_lk1.set_xlabel('Time (ms)')

    ax_v1.legend()


def plot_rm_vs_rpred(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    Cm=60

    ax = fig.add_subplot(subgrid[0]) 

    r_leaks = np.linspace(.25, 1.25, 5)
    g_leaks = [1/r for r in r_leaks]

    cols =  ['k', 'grey']

    delta_v = 1

    for it, base_v in enumerate([-80, 0]):
        base_v = base_v + .1
        segments = [VCSegment(50000, base_v)]
        for i in range(0, 100):
            segments.append(VCSegment(100, base_v))
            segments.append(VCSegment(100, base_v+1))

        py_proto = VCProtocol(segments)

        rm_pred = []

        for leak in g_leaks:
            t, dat = get_mod_response('./mmt/kernik_leak.mmt',
                                      {'membrane.gLeak': leak,
                                       'ifunny.g_f': 2,
                                       'ik1.g_K1': .1},
                                       vc_proto=py_proto)

            i_ion = dat['membrane.i_ion']
            rm = delta_v / (i_ion[-500] - i_ion[-1500]) / Cm
            rm_pred.append(rm)

            print(leak)

        ax.plot(r_leaks, rm_pred, c=cols[it], marker='o', label=f'V_hold={base_v-.1}mV')
    
    ax.plot(r_leaks, r_leaks, 'r', linestyle='--', alpha=.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('R_leak')
    ax.set_ylabel('Rm')

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




#Utility function
def get_mod_response(f_name='./mmt/kernik_2019_mc.mmt',
                     conductances={},
                     vc_proto=None):
    mod = myokit.load_model(f_name)

    for cond, g in conductances.items():
        group, param = cond.split('.')
        val = mod[group][param].value()
        mod[group][param].set_rhs(val*g)

    p = mod.get('engine.pace')
    p.set_binding(None)

    if vc_proto is None:
        vc_proto = return_vc_proto()

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

    t = proto.characteristic_time()
    sim = myokit.Simulation(mod, proto)

    times = np.arange(0, t_max, 0.1)

    dat = sim.run(t_max, log_times=times)

    return times, dat



def main():
    plot_figure()


if __name__ == "__main__":
    main()
