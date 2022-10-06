import myokit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from methods import post, plots
import methods.data
import methods.plots
import methods.post

from utility_classes import get_single_ap


def plot_tord_leak():
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    st = ['k', 'k--']
    labs = ['Baseline', '1 Gohm Leak']

    t_max = 10000

    times = np.arange(0, t_max, .1)

    mfile = './mmt/tor_ord_endo.mmt'
    mod, proto, x = myokit.load(mfile)
    sim = myokit.Simulation(mod, proto)

    dat_normal = sim.run(t_max, log_times=times)

    v = dat_normal['membrane.v']

    ax.plot(times[-40000:], v[-40000:], 'r--', label='Baseline')

    cmap = matplotlib.cm.get_cmap('viridis')

    for i, val in enumerate(np.arange(.2, 1.5, .1)):
        mfile = './mmt/tor_ord_endo_leak.mmt'
        mod, proto, x = myokit.load(mfile)
        mod['membrane']['gLeak'].set_rhs(val)
        sim = myokit.Simulation(mod, proto)

        dat_normal = sim.run(t_max, log_times=times)

        v = dat_normal['membrane.v']

        if ((val>.96) and (val< 1.02)):
            ax.plot(times[-40000:], v[-40000:], 'k--')
        else:
            ax.plot(times[-40000:], v[-40000:], c=cmap(15*i))


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Voltage (mV)', fontsize=14)
    plt.show()


plot_tord_leak()
