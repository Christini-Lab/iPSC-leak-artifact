import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import myokit

from utility import VCSegment, VCProtocol, get_mod_response


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10 
plt.rcParams['axes.labelsize'] = 10 
plt.rc('legend', fontsize = 8)


def plot_figure_80vs0():
    fig = plt.figure(figsize=(6.5, 2.75))
    fig.subplots_adjust(.12, .15, .95, .95)

    grid = fig.add_gridspec(1, 2, hspace=.2, wspace=0.25)

    #panel 1
    plot_gleak_effect_proto(fig, grid[0])


    #panel 2
    plot_rm_vs_rpred(fig, grid[1])

    plt.savefig('./figure-pdfs/f6.pdf')
    plt.show()


def plot_gleak_effect_proto(fig, grid_box):
    subgrid = grid_box.subgridspec(2, 1, wspace=.2, hspace=.1)

    ax_0 = fig.add_subplot(subgrid[0, 0]) 
    ax_0.set_title('A', y=.94, x=-.2)

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

    labels = [r'$g_f$=0 nS/pF', r'$g_f$=0.0435 nS/pF', '$g_f$=0.087 nS/pF']
    sts = ['-', 'dotted', '--']
    cols = ['k', 'grey', 'skyblue']
    leak = 1

    for i, ax in enumerate(axs):
        proto = protos[i]

        for j, g_f in enumerate([0.1, 1, 2]):
            t, dat = get_mod_response('./mmt/kernik_leak_fixed.mmt',
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


def plot_rm_vs_rpred(fig, grid_box):
    subgrid = grid_box.subgridspec(1, 1, wspace=.9, hspace=.1)
    Cm=50

    ax = fig.add_subplot(subgrid[0]) 
    ax.set_title('B', y=.94, x=-.2)

    r_leaks = np.linspace(.25, 1.5, 5)
    g_leaks = [1/r for r in r_leaks]
    gf_labels = ['0 nS/pF', '.0435 nS/pF', '.087 nS/pF']

    cols =  ['k', 'grey']

    delta_v = 5

    for it, base_v in enumerate([-80, 0]):
        base_v = base_v + .1
        proto = myokit.Protocol()
        proto.add_step(base_v, 100000)

        for i in range(0, 100):
            proto.add_step(base_v, 50)
            proto.add_step(base_v+5, 50)

        for iteration, gf in enumerate([0.1, 1, 2]):
            rm_pred = []

            if gf == .1:
                col = 'k'
                st = '-'
            elif gf == 1:
                col = 'grey'
                st = 'dotted'
            else:
                col = 'skyblue'
                st = 'dashed'
            
            if it == 0:
                marker = 'o'
            else:
                marker = '^'
            
            for leak in g_leaks:
                t, dat = get_mod_response('./mmt/kernik_leak_fixed.mmt',
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
                    label=f'{int(base_v-.1)}mV, $g_f$={gf_labels[iteration]}')
    
    ax.plot(r_leaks, r_leaks, 'r', linestyle='dotted', alpha=.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(r'$R_{seal} (G\Omega)$')
    ax.set_ylabel(r'$R_{in} (G\Omega)$')

    ax.set_ylim(0, 7)

    ax.legend(loc=2, framealpha=1)


def main():
    plot_figure_80vs0()
        

if __name__ == "__main__":
    main()
