import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import myokit

from utility_classes import VCSegment, VCProtocol


# Figure functions
def plot_figure():
    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(.07, .10, .95, .98)

    grid = fig.add_gridspec(2, 3, hspace=.2, wspace=0.2)


    #panel 1
    plot_whole_vc(fig, grid[0, 0])

    ##panel 2
    subgrid = grid[0, 1].subgridspec(2, 1, wspace=.9, hspace=.1)
    ax_exp = fig.add_subplot(subgrid[0]) 
    plot_if_vc(ax_exp)
    ax_mod = fig.add_subplot(subgrid[1]) 
    plot_kernik_vc(ax_mod)

    #panel 3
    plot_gleak_effect_proto(fig, grid[0, 2])

    #panel 4
    plot_rm_vs_rpred(fig, grid[1, 0])

    #panel 5
    plot_vhold_vs_rmpred(fig, grid[1, 1])

    #panel 6
    plot_exp_rm(fig, grid[1, 2])

    #panel 6
    #plot_leak_vs_rm_neg80(fig, grid[2, 1])

    plt.show()


def plot_whole_vc(fig, grid_box):
    subgrid = grid_box.subgridspec(2, 1, wspace=.9, hspace=.1)

    ax_v = fig.add_subplot(subgrid[0]) 
    ax_c = fig.add_subplot(subgrid[1]) 

    pre = pd.read_csv('./data/042721_4_quinine/pre-drug_vc_proto.csv')
    post = pd.read_csv('./data/042721_4_quinine/post-drug_vc_proto.csv')

    window = 2
    cols = ['k', 'r']

    for i, dat in enumerate([pre, post]):
        t = moving_average(dat['Time (s)'].values, window)
        c = moving_average(dat['Current (pA/pF)'].values, window)
        v = moving_average(dat['Voltage (V)'].values, window)

        ax_c.plot(t*1000, c, cols[i])

    ax_v.plot(t*1000, v*1000, 'k')

    for ax in [ax_v, ax_c]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_c.set_xlabel('Time (ms)')
    ax_c.set_ylabel(r'$I_m$ (A/F)')
    ax_v.set_ylabel(r'$V_{cmd}$')

    ax_v.set_xticklabels([])

    ax_c.set_ylim(-15, 10)

    ax_c.axvspan(4000, 6000, -15, 10, facecolor='grey', alpha=.2)
    ax_v.axvspan(4000, 6000, -15, 10, facecolor='grey', alpha=.2)
    

def plot_if_vc(ax):
    pre = pd.read_csv('./data/042721_4_quinine/pre-drug_vc_proto.csv')
    post = pd.read_csv('./data/042721_4_quinine/post-drug_vc_proto.csv')

    window = 2
    cols = ['k', 'r']
    labs = ['No Drug', 'Drug']

    for i, dat in enumerate([pre, post]):
        t = moving_average(dat['Time (s)'].values, window)
        start_idx = np.argmin(np.abs(t-4))
        end_idx = np.argmin(np.abs(t-6))
        c = moving_average(dat['Current (pA/pF)'].values, window)
        v = moving_average(dat['Voltage (V)'].values, window)
        t = t[start_idx:end_idx]
        c = c[start_idx:end_idx]
        v = v[start_idx:end_idx]

        ax.plot(t*1000, c, cols[i], label=labs[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylabel(r'Exp $I_m (A/F)$')

    ax.set_xticklabels([])

    ax.legend()


def plot_kernik_vc(ax):
    cols = ['k--', 'r--']
    names = ['No Drug', 'Drug']
    for i, cond in enumerate([1, .68]):
        t, dat = get_mod_response('./mmt/kernik_2019_mc.mmt', {'ifunny.g_f': cond}) 

        #remove first 900 ms so it aligns with exp data
        t = t[np.argmin(np.abs(t-900)):] - 900
        i_ion = dat['membrane.i_ion'][np.argmin(np.abs(t-900)):]

        start_idx = np.argmin(np.abs(t-4000))
        end_idx = np.argmin(np.abs(t-6000))

        t_zoom = t[start_idx:end_idx] 
        i_zoom = i_ion[start_idx:end_idx]

        ax.plot(t_zoom, i_zoom, cols[i], label=names[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylabel(r'KC $I_m (A/F)$')

    ax.legend()
    ax.set_xlabel('Time (ms)')


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

        ax.plot(r_leaks, rm_pred, c=cols[it], marker='o', label=f'V_hold={base_v}mV')
    
    ax.plot(r_leaks, r_leaks, 'r', linestyle='--', alpha=.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('R_leak')
    ax.set_ylabel('Rm')

    ax.legend()


def plot_gleak_effect_proto(fig, grid_box):
    subgrid = grid_box.subgridspec(3, 2, wspace=.2, hspace=.1)

    ax_v1 = fig.add_subplot(subgrid[0, 0]) 
    ax_c1 = fig.add_subplot(subgrid[1, 0]) 
    ax_lk1 = fig.add_subplot(subgrid[2, 0]) 
    ax_v1.set_title('-80 mV')

    ax_v2 = fig.add_subplot(subgrid[0, 1]) 
    ax_c2 = fig.add_subplot(subgrid[1, 1]) 
    ax_lk2 = fig.add_subplot(subgrid[2, 1]) 
    ax_v2.set_title('0 mV')

    axs = [[ax_v1, ax_c1, ax_lk1], [ax_v2, ax_c2, ax_lk2]]

    leak = 1 
    cols = ['k', 'grey']
    styles = ['-', '--']

    for it, vhold in enumerate([-80, .1]): 
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
            t = t - t[-1500]
            #if j == 0:
            axs[it][0].plot(t[-1500:-500], dat['membrane.V'][-1500:-500],
                                                        c=cols[j], linestyle=styles[j], label=f'G_f={g_f}')

            axs[it][1].plot(t[-1500:-500], dat['membrane.i_ion'][-1500:-500],
                                                        c=cols[j], linestyle=styles[j], label=f'G_f={g_f}')
            #else:
            #    axs[it][1].plot(t[-1500:-500], dat['membrane.i_ion'][-1500:-500], c=cols[j], linestyle=styles[j])

            ion_leak = np.array(dat['membrane.i_ion']) - np.array(dat['membrane.ILeak'])
            axs[it][2].plot(t[-1500:-500], ion_leak[-1500:-500], c=cols[j], linestyle=styles[j])

    for ax_list in axs:
        for ax in ax_list:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    for ax in [ax_v1, ax_c1, ax_v2, ax_c2]:
        ax.set_xticklabels([])

    ax_v2.legend()
    
    ax_v1.set_ylabel(r'$V_{cmd}$')
    ax_c1.set_ylabel(r'$I_{out}$ (A/F)')
    ax_lk1.set_ylabel(r'$I_{out}$ - $I_{leak}$ (A/F)')

    ax_lk1.set_xlabel('Time (ms)')
    ax_lk2.set_xlabel('Time (ms)')


def plot_leak_vs_rm_neg80(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    Cm = 60

    ax_ion = fig.add_subplot(subgrid[0]) 
    ax = fig.add_subplot(subgrid[0]) 

    delta_v = 1
    v_base = -80

    segments = [VCSegment(10000, v_base)]
    for i in range(0, 10):
        segments.append(VCSegment(100, v_base))
        segments.append(VCSegment(100, v_base + delta_v))

    py_proto = VCProtocol(segments)

    num_leaks = 8 
    num_funny = 4


    cmap = cm.get_cmap('viridis')
    diff_cols = int(250 / num_funny)

    for j, g_funny in enumerate(np.linspace(.2, 3, num_funny)):
        rm_actual = []
        rm_pred = []
        for val in np.linspace(.2, 5, num_leaks): 
            rm_actual.append(1 / val)

            t, dat = get_mod_response('./mmt/kernik_leak.mmt',
                                      {'membrane.gLeak': val,
                                       'ik1.g_K1': .1,
                                       'ifunny.g_f': g_funny},
                                      vc_proto=py_proto)
            i_ion = dat['membrane.i_ion']
            rm = delta_v / (i_ion[120750] - i_ion[119750])/Cm
            #rleak = delta_v / (i_leak[120750] - i_leak[119750])/cm

            #ax_ion.plot(t, i_ion, label=f'Gf={g_funny}, Rm={1/val}')
            #ax_leak.plot(t, dat['membrane.ILeak'], label=f'gf={g_funny}, rm={1/val}')
            #ax_leak.plot(t, dat['ifunny.i_f'])


            rm_pred.append(rm)
            print(f'Gf={g_funny} AND Rm = {rm}')

        ax.plot(rm_actual, rm_pred, c=cmap(diff_cols*j), label=g_funny, marker='o')

        #ax.plot(rm_actual, rm_pred, c=cmap(diff_cols*j), marker='o', label=f'$G_f$=g_funny')

    ax.set_xlabel(r'$R_{leak}$ (Gohm)')
    ax.set_ylabel(r'$R_{m}$ (Gohm)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()


def plot_vhold_vs_rmpred(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    Cm = 60

    ax = fig.add_subplot(subgrid[0]) 

    delta_v = 1
    num_voltages = 13#27
    voltages = np.linspace(-90, 30, num_voltages) + .1

    cmap = cm.get_cmap('viridis')
    diff_cols = int(250 / num_voltages)

    g_leak = 1

    cols = ['k', 'grey', 'lightgrey']
    g_k1 = .1
    for it, g_f in enumerate([1, 2]):
        rm_pred = []
        for v_base in voltages:
            segments = [VCSegment(50000, v_base)]

            for i in range(0, 125):
                segments.append(VCSegment(100, v_base))
                segments.append(VCSegment(100, v_base + delta_v))

            py_proto = VCProtocol(segments)

            t, dat = get_mod_response('./mmt/kernik_leak.mmt',
                                      {'membrane.gLeak': g_leak,
                                       'ifunny.g_f': g_f,
                                       'ik1.g_K1': g_k1},
                                      vc_proto=py_proto)

            i_ion = dat['membrane.i_ion']
            rm = delta_v / (i_ion[-500] - i_ion[-1500]) / Cm
            rm_pred.append(rm)

            print(f'At {v_base}, Rmpred is {rm}')


        ax.plot(voltages, rm_pred, cols[it], marker='o', label=f'G_f={g_f}')
    #ax.plot(t, i_ion, label=f'G_f={cond}')
    #ax.plot(t, dat['ifunny.i_f'], label=f'G_f={cond}')

    ax.axhline(y=1, color='grey', linestyle='--')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$V_{hold}$')
    ax.set_ylabel(r'Rm')
    ax.set_ylim(-1, 2)
    ax.legend()


def plot_exp_rm(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)

    ax = fig.add_subplot(subgrid[0]) 

    rm_vals = pd.read_csv('./data/rm-data.csv')

    trials = [1, 2, 3]
    for index, row in rm_vals.iterrows():
        vals = np.array(row.values[1:4]) / 1000
        if any(v > 1.6 for v in vals):
            continue
        ax.plot(trials, vals, 'grey', alpha=.2)
        ax.scatter(trials, vals, c='k')

    ax.axhline(y=1, color='red', linestyle='--', alpha=.2)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([1, 2, 3])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Exp Step')
    ax.set_ylabel('Exp Rm Pred at 0 mV (Gohm)')
    ax.set_ylim(0, 1.6)
        


#def plot_vhold_effect_rm(fig, grid_box):


# Helper functions
def return_vc_proto(scale=1):
    segments = [
            VCSegment(500, -80),
            VCSegment(757, 6),
            VCSegment(7, -41),
            VCSegment(101, 8.5),
            VCSegment(500, -80),
            VCSegment(106, -81),
            VCSegment(103, -2, -34),
            VCSegment(500, -80),
            VCSegment(183, -87),
            VCSegment(102, -52, 14),
            VCSegment(500, -80),
            VCSegment(272, 54, -107),
            VCSegment(103, 60),
            VCSegment(500, -80),
            VCSegment(52, -76, -80),
            VCSegment(103, -120),
            VCSegment(500, -80),
            VCSegment(940, -120),
            VCSegment(94, -77),
            VCSegment(8.1, -118),
            VCSegment(500, -80),
            VCSegment(729, 55),
            VCSegment(1000, 48),
            VCSegment(895, 59, 28)
            ]

    new_segments = []
    for seg in segments:
        if seg.end_voltage is None:
            new_segments.append(VCSegment(seg.duration*scale, seg.start_voltage*scale))
        else:
            new_segments.append(VCSegment(seg.duration*scale,
                                          seg.start_voltage*scale,
                                          seg.end_voltage*scale))
    
    return VCProtocol(new_segments)


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


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs] 

    return np.array(new_vals)


def main():
    plot_figure()


if __name__ == '__main__':
    main()

