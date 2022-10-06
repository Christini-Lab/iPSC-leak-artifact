import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import myokit
import matplotlib 
from multiprocessing import Pool

from utility_classes import VCSegment, VCProtocol


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10 
plt.rcParams['axes.labelsize'] = 10 
plt.rc('legend', fontsize = 8)


def plot_figure():
    fig = plt.figure(figsize=(6.5, 5))
    fig.subplots_adjust(.12, .1, .95, .95)

    #grid = fig.add_gridspec(2, 2, hspace=.2, wspace=0.2)
    grid = fig.add_gridspec(1, 1, hspace=.2, wspace=0.2)

    #panel 1
    #plot_if_vc(fig, grid[0, 0])

    ##panel 2
    #plot_kernik_vc(fig, grid[0, 1])

    #panel 3
    #plot_gleak_effect_proto(fig, grid[1, 0])

    #panel 4
    plot_vhold_vs_rmpred()

    matplotlib.rcParams['pdf.fonttype'] = 42

    plt.show()


def plot_vhold_vs_rmpred():
    Cm = 60

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax_i = axs[0] 
    ax_delta_i = axs[1] 
    ax_rm = axs[2] 

    delta_v = 5
    num_voltages = 25
    voltages = np.linspace(-90, 30, num_voltages) + .1

    cmap = cm.get_cmap('viridis')
    diff_cols = int(250 / num_voltages)

    cols = ['k', 'grey', 'lightgrey']

    #for it, mod in enumerate(['./mmt/kernik_2019_mc_fixed.mmt',
    #                                './mmt/kernik_leak_fixed.mmt']):
    for it, mod in enumerate(['./mmt/paci-2013-ventricular-fixed.mmt',
                                    './mmt/paci-2013-ventricular-leak-fixed.mmt']):
        new_dat = [[v, mod] for v in voltages]
        p = Pool()

        i_vals = p.map(get_i_ion, new_dat)
        i_vals = np.array(i_vals)

        if it == 0:
            ax_i.plot(voltages, i_vals[:, 0], cols[it], marker='o', label='Baseline')
            ax_delta_i.plot(voltages, i_vals[:, 1], cols[it], marker='o', label='Baseline')
            ax_rm.plot(voltages, i_vals[:, 2], cols[it], marker='o', label='Baseline')
        else:
            ax_i.plot(voltages, i_vals[:, 0], cols[it], marker='o', linestyle='--', label='Baseline+1Gohm')
            ax_delta_i.plot(voltages, i_vals[:, 1], cols[it], marker='o', linestyle='--', label='Baseline+1Gohm')
            ax_rm.plot(voltages, i_vals[:, 2], cols[it], marker='o', linestyle='--', label='Baseline+1Gohm')

    ax_i.axhline(y=0, color='r', linestyle='dotted', alpha=.3)
    ax_delta_i.axhline(y=0, color='r', linestyle='dotted', alpha=.3)
    ax_rm.axhline(y=1, color='blue', linestyle='dotted', alpha=.3)

    for ax in [ax_i, ax_delta_i, ax_rm]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    ax_rm.set_xlabel(r'$V_{hold}$ (mV)')
    ax_i.set_ylabel(r'$I_{out}$ (pA/pF)')
    ax_delta_i.set_ylabel(r'$I_{ion}(V+5)-I_{ion}(V)$ (pA/pF)')
    ax_rm.set_ylabel(r'$R_{in}$ (Gohms)')
    ax_i.legend()


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
            VCSegment(188, -119.5),
            VCSegment(752, -120),
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


def get_mod_response(f_name='./mmt/kernik_2019_mc_fixed.mmt',
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


def get_i_ion(inputs):
    print(inputs)
    v_base= inputs[0] 
    delta_v = 5
    mod = inputs[1]
    mk_proto = myokit.Protocol()
    mk_proto.add_step(v_base, 100000)
    g_leak = 1
    g_k1 = 1
    if 'paci' in mod:
        Cm = 98.7
    else:
        Cm = 60

    for i in range(0, 100):
        mk_proto.add_step(v_base, 50)
        mk_proto.add_step(v_base+delta_v, 50)

    if 'leak' in mod:
        t, dat = get_mod_response(mod,
                                  {'membrane.gLeak': g_leak},
                                  vc_proto=mk_proto)
    else:
        t, dat = get_mod_response(mod,
                                  {},
                                  vc_proto=mk_proto)

    i_ion = dat['membrane.i_ion']
    delta_i = i_ion[-250] - i_ion[-750]
    rm = delta_v / (delta_i) / Cm
    #rm_pred.append(rm)
    return i_ion[-750], delta_i, rm


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs]

    return np.array(new_vals)


plot_figure()
