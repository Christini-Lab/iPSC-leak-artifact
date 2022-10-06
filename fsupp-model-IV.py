import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import myokit
import matplotlib 
from multiprocessing import Pool

from utility import VCSegment, VCProtocol, get_mod_response, moving_average, return_vc_proto


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10 
plt.rcParams['axes.labelsize'] = 10 
plt.rc('legend', fontsize = 8)


def plot_vhold_vs_rmpred(models):
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

    for it, mod in enumerate(models):
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
    return i_ion[-750], delta_i, rm


plot_vhold_vs_rmpred(['./mmt/kernik_2019_mc_fixed.mmt',
                                    './mmt/kernik_leak_fixed.mmt'])
plot_vhold_vs_rmpred(['./mmt/paci-2013-ventricular-fixed.mmt',
                                    './mmt/paci-2013-ventricular-leak-fixed.mmt'])
