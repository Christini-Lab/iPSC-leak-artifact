import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import myokit
import matplotlib 
from multiprocessing import Pool

from utility import VCSegment, VCProtocol, moving_average, get_mod_response


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

    grid = fig.add_gridspec(2, 2, hspace=.2, wspace=0.3)

    #panel 1
    plot_if_vc(fig, grid[0, 0])

    ##panel 2
    plot_kernik_vc(fig, grid[0, 1])

    #panel 3
    plot_gleak_effect_proto(fig, grid[1, 0])

    #panel 4
    plot_vhold_vs_rmpred(fig, grid[1, 1])

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f5.pdf', transparent=True)

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
    ax_c.set_ylabel(r'Exp $I_{out} (A/F)$')
    ax_c.set_xlabel('Time (ms)')

    ax_v.set_xticklabels([])
    ax_c.set_ylim(-12, 0)
    ax_c.legend(loc=3, framealpha=1)

    fig.align_ylabels([ax_v, ax_c])


def plot_kernik_vc(fig, grid_box):
    subgrid = grid_box.subgridspec(2, 1, wspace=.9, hspace=.1)

    ax_v = fig.add_subplot(subgrid[0])
    ax_v.set_title('B', y=.94, x=-.2)
    ax_c = fig.add_subplot(subgrid[1])

    cols = ['k', 'r']
    names = ['No Drug', 'Drug']
    for i, cond in enumerate([1, .68]):
        t, dat = get_mod_response('./mmt/kernik_leak_fixed.mmt', {'ifunny.g_f': cond})

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

    fig.align_ylabels([ax_v, ax_c])


def plot_gleak_effect_proto(fig, grid_box):
    subgrid = grid_box.subgridspec(3, 1, wspace=.2, hspace=.05)

    ax_out = fig.add_subplot(subgrid[0, 0]) 
    ax_out.set_title('C', y=.94, x=-.2)
    ax_ion = fig.add_subplot(subgrid[1, 0]) 
    ax_lk = fig.add_subplot(subgrid[2, 0]) 

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

    fig.align_ylabels([ax_out, ax_ion, ax_lk])
    ax_lk.legend(loc=7, framealpha=1)

    fig.align_ylabels([ax_out, ax_ion, ax_lk])


def plot_vhold_vs_rmpred(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    Cm = 60

    ax = fig.add_subplot(subgrid[0]) 
    ax.set_title('D', y=.94, x=-.2)

    delta_v = 5
    num_voltages = 25
    voltages = np.linspace(-90, 30, num_voltages) + .1

    cmap = cm.get_cmap('viridis')
    diff_cols = int(250 / num_voltages)

    g_leak = 1

    cols = ['k', 'grey', 'lightgrey']
    g_k1 = .1

    gf_vals = {1: '0.0435 nS/pF',
               2: '0.087 nS/pF'}


    for it, g_f in enumerate([1, 2]):
        new_dat = [[v, g_f] for v in voltages]

        p = Pool()
        # Change p.map to map if you get an error 
        rm_pred = p.map(get_rm, new_dat)

        if it == 0:
            ax.plot(voltages, rm_pred, cols[it], marker='o')
        else:
            ax.plot(voltages, rm_pred, cols[it], marker='o', linestyle='--')

    ax.axhline(y=1, color='r', linestyle='dotted', alpha=.3)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$V_{hold}$ (mV)')
    ax.set_ylabel(r'$R_{in}$ ($G\Omega$)')
    ax.set_ylim(-1, 2)


def get_rm(inputs):
    print(inputs)
    v_base, g_f = inputs[0], inputs[1]
    delta_v = 5
    Cm = 60
    mk_proto = myokit.Protocol()
    mk_proto.add_step(v_base, 100000)
    g_k1 = .1
    g_leak = 1
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
    return rm


def main():
    plot_figure()


if __name__ == '__main__':
    main()
