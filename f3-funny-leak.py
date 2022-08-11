import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import myokit
import matplotlib 

from utility_classes import VCSegment, VCProtocol


def plot_figure():
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(.07, .10, .95, .98)

    grid = fig.add_gridspec(2, 2, hspace=.2, wspace=0.2)

    #panel 1
    plot_whole_vc(fig, grid[0, 0])

    ##panel 2
    plot_exp_kernik_vc(fig, grid[0, 1])

    #panel 3
    plot_gleak_effect_proto(fig, grid[1, 0])

    #panel 4
    plot_vhold_vs_rmpred(fig, grid[1, 1])

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f3-funny-leak.pdf', transparent=True)

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


def plot_exp_kernik_vc(fig, grid_box):
    subgrid = grid_box.subgridspec(2, 1, wspace=.9, hspace=.1)
    ax_exp = fig.add_subplot(subgrid[0]) 
    plot_if_vc(ax_exp)
    ax_mod = fig.add_subplot(subgrid[1]) 
    plot_kernik_vc(ax_mod)


def plot_gleak_effect_proto(fig, grid_box):
    subgrid = grid_box.subgridspec(3, 1, wspace=.2, hspace=.1)

    ax_v1 = fig.add_subplot(subgrid[0, 0]) 
    ax_c1 = fig.add_subplot(subgrid[1, 0]) 
    ax_lk1 = fig.add_subplot(subgrid[2, 0]) 

    axs = [ax_v1, ax_c1, ax_lk1]

    leak = 1 
    cols = ['k', 'grey']
    styles = ['-', '--']

    vhold = -80
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


        ax.plot(voltages, rm_pred, cols[it], marker='o', label=f'$G_f$={g_f}')
    #ax.plot(t, i_ion, label=f'G_f={cond}')
    #ax.plot(t, dat['ifunny.i_f'], label=f'G_f={cond}')

    ax.axhline(y=1, color='grey', linestyle='--')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$V_{hold}$')
    ax.set_ylabel(r'Rm')
    ax.set_ylim(-1, 2)
    ax.legend()






#Helpers
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


plot_figure()
