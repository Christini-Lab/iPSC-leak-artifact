import myokit
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib

from utility import get_single_ap


def figure_leak_paced():
    fig = plt.figure(figsize=(6.5, 5))
    fig.subplots_adjust(.1, .1, .93, .93)
    grid = fig.add_gridspec(2, 2, hspace=.3, wspace=.25)
    axs = []

    #Panel A
    subgrid = grid[0, 0].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_mod_paced(ax, 'Kernik', stim=.03, period=600, leak=.1, title='Kernik Spont')
    axs.append(ax)
    ax.set_ylabel('Voltage (mV)')

    #Panel B
    subgrid = grid[0, 1].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_mod_paced(ax, 'Kernik', stim=3, period=600, leak=.1, title='Kernik Paced')
    axs.append(ax)

    #Panel C
    subgrid = grid[1, 0].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_mod_paced(ax, 'Paci', stim=.03, period=900, leak=.8, title='Paci Spont')
    axs.append(ax)
    ax.set_xlabel('Time (ms)')

    #Panel D
    subgrid = grid[1, 1].subgridspec(1, 1)
    ax = fig.add_subplot(subgrid[0])
    plot_mod_paced(ax, 'Paci', stim=3, period=900, leak=.8, title='Paci Paced')
    axs.append(ax)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')

    plt.legend()

    #alphas = ['A', 'B', 'C', 'D']
    for i, ax in enumerate(axs):
        #ax.set_title(alphas[i], y=.94, x=-.2)
        #ax.set_xlim(-5, 120)
        ax.set_ylim(-80, 40)

    plt.savefig('./figure-pdfs/f-fixed.pdf')
    plt.show()


def plot_mod_paced(ax, mod_name, stim, period, leak, title):
    if mod_name == 'Kernik':
        mfile = './mmt/kernik_2019_mc_fixed.mmt'
        mfile_leak = './mmt/kernik_leak_fixed.mmt'
    else:
        mfile = './mmt/paci-2013-ventricular-fixed.mmt'
        mfile_leak = './mmt/paci-2013-ventricular-leak-fixed.mmt'

    base, p, x = myokit.load(mfile)

    proto = myokit.Protocol()
    proto.add(myokit.ProtocolEvent(stim, 10, 2, period))

    s_base = myokit.Simulation(base, proto)
    s_base.pre(35000)

    res_base = s_base.run(10000)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    ax.plot(single_t, single_v, 'k', label='Baseline')

    ### NEXT #####
    proto = myokit.Protocol()
    proto.add(myokit.ProtocolEvent(stim, 10, 1, period))

    #leak = .2
    mod, p, x = myokit.load(mfile_leak)
    mod['membrane']['gLeak'].set_rhs(leak)
    s_base = myokit.Simulation(mod, proto)

    s_base.pre(35000)

    res_base = s_base.run(10000)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    ax.plot(single_t, single_v, 'k--', label='Leak')
    ax.set_xlim(-100, 1000)
    ax.set_title(title)
    #ax.legend()




def leak_effects2():
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    all_biomarkers = {0: [], 1:[]}

    mfile = './mmt/kernik_2019_mc_fixed.mmt'
    mfile = './mmt/paci-2013-ventricular-fixed.mmt'
    kernik_base, p, x = myokit.load(mfile)

    proto = myokit.Protocol()
    proto.add(myokit.ProtocolEvent(.03, 10, 2, 900))

    s_base = myokit.Simulation(kernik_base, proto)
    s_base.pre(35000)

    res_base = s_base.run(10000)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    ax.plot(single_t, single_v, 'k', label='Baseline')

    ### NEXT #####
    proto = myokit.Protocol()
    proto.add(myokit.ProtocolEvent(.03, 10, 1, 900))

    leak = .2
    mfile = './mmt/kernik_leak_fixed.mmt'
    mfile = './mmt/paci-2013-ventricular-leak-fixed.mmt'
    mod, p, x = myokit.load(mfile)
    mod['membrane']['gLeak'].set_rhs(leak)
    s_base = myokit.Simulation(mod, proto)

    s_base.pre(35000)

    res_base = s_base.run(10000)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)


    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    ax.plot(single_t, single_v, 'k--', label='Leak')
    ax.set_xlim(-100, 1000)
    #ax.tick_params(
    #    axis='x',          # changes apply to the x-axis
    #    which='both',      # both major and minor ticks are affected
    #    bottom=False,      # ticks along the bottom edge are off
    #    top=False,         # ticks along the top edge are off
    #    labelbottom=False)

    #ax.tick_params(
    #    axis='y',          # changes apply to the x-axis
    #    which='both',      # both major and minor ticks are affected
    #    left=False,      # ticks along the bottom edge are off
    #    right=False,         # ticks along the top edge are off
    #    labelleft=False)

    ax.axhline(0, c='grey', linestyle='dotted', alpha=.5)

    #plt.legend()
    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f1-ap.pdf', transparent=True)

    plt.show()


def main():
    #leak_effects()
    #leak_effects2()
    figure_leak_paced()


if __name__ == '__main__':
    main()
