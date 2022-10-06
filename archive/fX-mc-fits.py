import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import myokit

random.seed(1)

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10 
plt.rcParams['axes.labelsize'] = 10 
plt.rc('legend', fontsize = 8)


def get_kernik_ap(param_updates={}):
    mfile = './mmt/kernik_2019_mc.mmt'
    k_mod, p, x = myokit.load(mfile)

    for param, val in param_updates.items():
        group, key = param.split('.')
        k_mod[group][key].set_rhs(val*k_mod[group][key].value())

    s_base = myokit.Simulation(k_mod)

    t_max = 100000

    times = np.arange(0, t_max, .5)

    res_base = s_base.run(t_max, log_times=times)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    return (single_t, single_v)


def get_target_ap(param_updates={'membrane.gLeak': .2}):
    mfile = './mmt/kernik_leak.mmt'
    k_mod, p, x = myokit.load(mfile)
    
    for param, val in param_updates.items():
        group, key = param.split('.')
        k_mod[group][key].set_rhs(val*k_mod[group][key].value())

    s_base = myokit.Simulation(k_mod)

    t_max = 100000

    times = np.arange(0, t_max, .5)

    res_base = s_base.run(t_max, log_times=times)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    return (single_t, single_v)


def get_single_ap(t, v):
    t = np.array(t)

    interval = t[1] - t[0]

    min_v, max_v = np.min(v), np.max(v)

    if (max_v - min_v) < 10:
        return None, None

    dvdt_peaks = find_peaks(np.diff(v)/np.diff(t), distance=20/interval, width=1/interval, prominence=.3)[0]

    if dvdt_peaks.size  < 4:
        return None, None

    start_idx = int(dvdt_peaks[-3] - 200 / interval)
    end_idx = int(dvdt_peaks[-3] + 800 / interval)

    t = t - t[start_idx]

    return t[start_idx:end_idx], v[start_idx:end_idx]


def main():
    t_base, v_base = get_kernik_ap()
    t_leak, v_leak = get_target_ap()
    t_fit, v_fit = get_kernik_ap({'ibna.g_b_Na': 7.081649113309206,
                                  'ibca.g_b_Ca': 0.3127089921965326})

    plt.plot(t_base, v_base, 'grey', alpha=.5, label='Baseline')
    plt.plot(t_leak, v_leak, 'k', label='Kernik+Leak')
    plt.plot(t_fit, v_fit, 'tomato', label='Kernik Fit')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
