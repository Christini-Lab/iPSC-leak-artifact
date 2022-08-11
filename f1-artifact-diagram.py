import myokit
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib

from methods import post, plots
import methods.data
import methods.plots
import methods.post

from utility_classes import get_single_ap


def leak_effects():
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    all_biomarkers = {0: [], 1:[]}

    mfile = './mmt/kernik_2019_mc.mmt'
    kernik_base, p, x = myokit.load(mfile)
    s_base = myokit.Simulation(kernik_base)
    s_base.pre(35000)

    res_base = s_base.run(10000)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    ax.plot(single_t, single_v, 'k', label='Baseline')



    ### NEXT #####
    leak = .2
    mfile = './mmt/kernik_leak.mmt'
    mod, p, x = myokit.load(mfile)
    mod['membrane']['gLeak'].set_rhs(leak)
    s_base = myokit.Simulation(mod)

    s_base.pre(35000)

    res_base = s_base.run(10000)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    ax.plot(single_t, single_v, 'k--', label='Leak')
    ax.set_xlim(-100, 480)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        labelleft=False)


    plt.legend()
    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f1-ap.pdf', transparent=True)

    plt.show()


def plot_vc_dat():
    proto = myokit.Protocol()
    proto.add_step(-90, 1000)
    proto.add_step(-40, 30)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i, m_name in enumerate(['kernik_2019_mc.mmt', 'kernik_artifact.mmt']):
        mod = myokit.load_model(f'./mmt/{m_name}')

        if i == 0:
            p = mod.get('engine.pace')
            p.set_binding(None)

            v = mod.get('membrane.V')
            v.demote()
            v.set_rhs(0)
            v.set_binding('pace') # Bind to the pacing mechanism

        sim = myokit.Simulation(mod, proto)
        t = proto.characteristic_time()

        times = np.arange(0, t, .05)

        dat = sim.run(t, log_times=times)

        if i == 0:
            axs[0].plot(times, dat['membrane.V'], 'k')
            axs[1].plot(times, dat['membrane.i_ion'], 'k', label='Baseline')
        else:
            cm = mod['geom']['Cm'].value()
            i_out = [i_out/cm for i_out in dat['voltageclamp.Iout']]
            axs[1].plot(times, i_out, 'k--', label='W/ Access resistance')

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])

    axs[0].set_xlim(998, 1010)
    axs[1].set_ylim(-210, 18)
    axs[1].legend()

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f1-vc.pdf', transparent=True)
    plt.show()


def get_exp_sodium_proto(scale):
    proto = myokit.Protocol()
    for i, val in enumerate(range(-7, 3, 1)):
        val = val / 100
        proto.add_step(-.08*scale, .4*scale)
        if val == 0:
            val += .001

        proto.add_step(val*scale, .05*scale)

    for i, val in enumerate(range(4, 7, 2)):
        val = val / 100
        proto.add_step(-.08*scale, .4*scale)
        proto.add_step(val*scale, .05*scale)

    return proto




#leak_effects()
plot_vc_dat()
    

