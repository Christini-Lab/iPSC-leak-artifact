import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import myokit
import matplotlib 

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

    grid = fig.add_gridspec(2, 2, hspace=.2, wspace=0.2)

    #panel 1
    plot_if_vc(fig, grid[0, 0])

    ##panel 2
    plot_kernik_vc(fig, grid[0, 1])

    #panel 3
    plot_gleak_effect_proto(fig, grid[1, 0])

    #panel 4
    plot_vhold_vs_rmpred(fig, grid[1, 1])

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-funny-leak.pdf', transparent=True)

    plt.show()


def plot_if_vc(fig, grid_box):
    subgrid = grid_box.subgridspec(2, 1, wspace=.9, hspace=.1)

    ax_v = fig.add_subplot(subgrid[0])
    ax_v.set_title('A', y=.94, x=-.2)
    ax_c = fig.add_subplot(subgrid[1])

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

        ax_c.plot(t*1000, c, cols[i], label=labs[i])

    ax_v.plot(t*1000, v*1000, c='k')

    for ax in [ax_c, ax_v]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    ax_v.set_ylabel(r'$V_{cmd}$ (mV)')
    ax_c.set_ylabel(r'Exp $I_m (A/F)$')
    ax_c.set_xlabel('Time (ms)')

    ax_v.set_xticklabels([])
    ax_c.set_ylim(-12, 0)
    ax_c.legend(loc=3)


def plot_kernik_vc(fig, grid_box):
    subgrid = grid_box.subgridspec(2, 1, wspace=.9, hspace=.1)

    ax_v = fig.add_subplot(subgrid[0])
    ax_v.set_title('B', y=.94, x=-.2)
    ax_c = fig.add_subplot(subgrid[1])

    cols = ['k', 'r']
    names = ['No Drug', 'Drug']
    for i, cond in enumerate([1, .68]):
        t, dat = get_mod_response('./mmt/kernik_2019_mc_fixed.mmt', {'ifunny.g_f': cond})

        t = t[np.argmin(np.abs(t-900)):] - 900
        i_ion = dat['membrane.i_ion'][np.argmin(np.abs(t-900)):]
        v = dat['membrane.V'][np.argmin(np.abs(t-900)):]

        start_idx = np.argmin(np.abs(t-4000))
        end_idx = np.argmin(np.abs(t-6000))

        t_zoom = t[start_idx:end_idx]
        i_zoom = i_ion[start_idx:end_idx]
        v_zoom = v[start_idx:end_idx]

        ax_c.plot(t_zoom, i_zoom, cols[i], label=names[i])

    ax_v.plot(t_zoom, v_zoom, 'k')

    for ax in [ax_c, ax_v]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    ax_v.set_ylabel(r'$V_{cmd}$ (mV)')
    ax_c.set_ylabel(r'KC $I_{ion} (A/F)$')
    ax_c.set_xlabel('Time (ms)')

    ax_v.set_xticklabels([])
    ax_c.set_ylim(-12, 0)


def plot_gleak_effect_proto(fig, grid_box):
    subgrid = grid_box.subgridspec(3, 1, wspace=.2, hspace=.05)

    #ax_v1 = fig.add_subplot(subgrid[0, 0]) 
    ax_out = fig.add_subplot(subgrid[0, 0]) 
    ax_out.set_title('C', y=.94, x=-.2)
    ax_ion = fig.add_subplot(subgrid[1, 0]) 
    ax_lk = fig.add_subplot(subgrid[2, 0]) 

    #axs = [ax_v1, ax_out, ax_ion, ax_lk]
    axs = [ax_out, ax_ion, ax_lk]

    leak = 1 
    cols = ['k', 'grey']
    styles = ['-', '--']

    v_base = -80
    v_step = 5

    mk_proto = myokit.Protocol()
    mk_proto.add_step(v_base, 100000)

    for i in range(0, 100):
        mk_proto.add_step(v_base, 50)
        mk_proto.add_step(v_base+5, 50)

    gf_vals = {1: '0.0435 nS/pF',
               2: '0.087 nS/pF'}

    i_ion_vals = []
    i_out_vals = []

    for j, g_f in enumerate([1, 2]):
        t, dat = get_mod_response('./mmt/kernik_leak_fixed.mmt',
                                  {'membrane.gLeak': leak,
                                   'ifunny.g_f': g_f},
                                  vc_proto=mk_proto)
        st = 1750 
        en = 750 
        t = t - t[-st]

        axs[0].plot(t[-st:-en], dat['membrane.i_ion'][-st:-en],
                                                    c=cols[j], linestyle=styles[j])
        i_ion = np.array(dat['membrane.i_ion']) - np.array(dat['membrane.ILeak'])
        axs[1].plot(t[-st:-en], i_ion[-st:-en], c=cols[j], linestyle=styles[j])
        axs[2].plot(t[-st:-en], dat['membrane.ILeak'][-st:-en], c=cols[j], linestyle=styles[j], label=f'$g_f$={gf_vals[g_f]}')

        i_out_vals.append(dat['membrane.i_ion'][int(-((st+en)/2))])
        i_ion_vals.append(i_ion[int(-((st+en)/2))])

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    for ax in [ax_out, ax_ion]:
        ax.set_xticklabels([])

    ax_out.set_ylabel(r'$I_{out}$ (A/F)')
    ax_ion.set_ylabel(r'$I_{ion}$ (A/F)')
    ax_lk.set_ylabel(r'$I_{leak}$ (A/F)')

    ax_lk.set_xlabel('Time (ms)')

    line_x = 50 
    ax_out.annotate(s='', xy=(line_x,i_out_vals[0]), xytext=(line_x,i_out_vals[1]), arrowprops=dict(arrowstyle="<->, head_length=.25", color='r', lw=.6))
    #ax_out.hlines(y=i_out_vals[0], xmin=50, xmax=line_x, color='r', linestyle='dotted')
    #ax_out.hlines(y=i_out_vals[1], xmin=50, xmax=line_x, color='r', linestyle='dotted')

    #ax_ion.annotate(s='', xy=(line_x,i_ion_vals[0]), xytext=(line_x,i_ion_vals[1]), arrowprops=dict(arrowstyle="<->, head_length=.2", color='r', lw=1))
    #ax_ion.hlines(y=i_ion_vals[0], xmin=50, xmax=line_x, color='r', linestyle='dotted')
    #ax_ion.hlines(y=i_ion_vals[1], xmin=50, xmax=line_x, color='r', linestyle='dotted')

    fig.align_ylabels([ax_out, ax_ion, ax_lk])
    ax_lk.legend(loc=7, framealpha=1)


def plot_vhold_vs_rmpred(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    Cm = 60

    ax = fig.add_subplot(subgrid[0]) 
    ax.set_title('D', y=.94, x=-.2)

    delta_v = 5
    num_voltages = 13#27
    voltages = np.linspace(-90, 30, num_voltages) + .1

    cmap = cm.get_cmap('viridis')
    diff_cols = int(250 / num_voltages)

    g_leak = 1

    cols = ['k', 'grey', 'lightgrey']
    g_k1 = .1
    #g_k1 = 1

    gf_vals = {1: '0.0435 nS/pF',
               2: '0.087 nS/pF'}

    for it, g_f in enumerate([1, 2]):
        rm_pred = []
        for v_base in voltages:
            mk_proto = myokit.Protocol()
            mk_proto.add_step(v_base, 100000)
            for i in range(0, 100):
                mk_proto.add_step(v_base, 50)
                mk_proto.add_step(v_base+5, 50)

            t, dat = get_mod_response('./mmt/kernik_leak_fixed.mmt',
                                      {'membrane.gLeak': g_leak,
                                       'ifunny.g_f': g_f,
                                       'ik1.g_K1': g_k1},
                                      vc_proto=mk_proto)

            i_ion = dat['membrane.i_ion']
            rm = delta_v / (i_ion[-250] - i_ion[-750]) / Cm
            rm_pred.append(rm)

            print(f'At {v_base}, Rmpred is {rm}')



        if it == 0:
            #ax.plot(voltages, rm_pred, cols[it], marker='o', label=f'$g_f$={gf_vals[g_f]}')
            ax.plot(voltages, rm_pred, cols[it], marker='o')
        else:
            #ax.plot(voltages, rm_pred, cols[it], marker='o', label=f'$g_f$={gf_vals[g_f]}', linestyle='--')
            ax.plot(voltages, rm_pred, cols[it], marker='o', linestyle='--')

    #ax.plot(t, i_ion, label=f'G_f={cond}')
    #ax.plot(t, dat['ifunny.i_f'], label=f'G_f={cond}')

    ax.axhline(y=1, color='r', linestyle='dotted', alpha=.3)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$V_{hold}$ (mV)')
    ax.set_ylabel(r'$R_{in}$ ($G\Omega$)')
    ax.set_ylim(-1, 2)
    #ax.legend()



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


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs]

    return np.array(new_vals)


plot_figure()
