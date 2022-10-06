import myokit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from methods import post, plots
import methods.data
import methods.plots
import methods.post

from utility_classes import get_single_ap


def plot_leak():
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fs = 14

    mfile = './mmt/kernik_2019_mc.mmt'
    kernik_base, p, x = myokit.load(mfile)
    s_base = myokit.Simulation(kernik_base)
    s_base.pre(35000)

    res_base = s_base.run(10000)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Time (ms)', fontsize=fs)

    ax.set_ylabel('Voltage (mV)', fontsize=fs)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    ax.plot(single_t, single_v, c='grey', linestyle='--', alpha=.8, label='Baseline')

    modname='./mmt/kernik_leak.mmt'

    mod, p, x = myokit.load(modname)
    leak = .05 

    labs = ['G_bNa=0, Rleak=20Gohms', 'G_bNa=1, Rleak=20Gohms']
    cols = ['k', 'r--']

    for i, bg in enumerate([.01, 1]):
        mod['membrane']['gLeak'].set_rhs(leak)
        mod['ibna']['g_scale'].set_rhs(bg)

        s_base = myokit.Simulation(mod)
        s_base.pre(15000)

        res_base = s_base.run(10000)

        t = res_base.time()
        v = res_base['membrane.V']

        t = np.asarray(t) - 6000
        arg_t = np.argmin(np.abs(t))

        single_t, single_v = get_single_ap(np.array(res_base.time()),
                                           np.array(res_base['membrane.V']))
        ax.plot(single_t, single_v, cols[i], label=labs[i])


    plt.legend()
    plt.show()


plot_leak()
