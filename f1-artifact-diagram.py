import myokit
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib

from utility import get_single_ap


def leak_effects():
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    all_biomarkers = {0: [], 1:[]}

    mfile = './mmt/kernik_2019_mc_fixed.mmt'
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
    mfile = './mmt/kernik_leak_fixed.mmt'
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

    ax.axhline(0, c='grey', linestyle='dotted', alpha=.5)

    plt.legend()
    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f1-ap.pdf', transparent=True)

    plt.show()


def main():
    leak_effects()


if __name__ == '__main__':
    main()
