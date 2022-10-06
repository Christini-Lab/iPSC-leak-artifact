import myokit
import pints
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class Model(pints.ForwardModel):
    def __init__(self):
        self.model = myokit.load_model('./mmt/kernik_2019_mc_fixed.mmt')
        self.sim = myokit.Simulation(self.model)


    def n_parameters(self):
        return 2


    def simulate(self, parameters, times):
        print('HELLO')

        self.sim.reset()

        param_names = ['ibna.g_b_Na', 'ibca.g_b_Ca']
        for param, val in dict(zip(param_names, parameters)).items():
            group, key = param.split('.')
            p_val = val * self.model[group][key].value()
            self.sim.set_constant(param, p_val)

        tmax = times[-1] + .1

        log = self.sim.run(tmax, log_times=times, log=['membrane.V'])

        t, v = get_single_ap(times, log['membrane.V'])

        return (t, v)


def get_target_ap(times, param_updates={'membrane.gLeak': .2}):
    mfile = './mmt/kernik_leak_fixed.mmt'
    k_mod, p, x = myokit.load(mfile)
    
    for param, val in param_updates.items():
        group, key = param.split('.')
        k_mod[group][key].set_rhs(val*k_mod[group][key].value())

    s_base = myokit.Simulation(k_mod)

    t_max = 0.1 + times[-1]

    res_base = s_base.run(t_max, log_times=times, log=['membrane.V'])

    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(times, v)

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


def run_fit():
    #Get values
    t_max = 100000
    times = np.arange(0, t_max, .1)

    target_t, target_v = get_target_ap(times=times)

    print(len(target_v))

    model = Model()
    problem = pints.SingleOutputProblem(model, times, target_v)
    error = pints.MeanSquaredError(problem)
    parameters = [1, 1]
    err = error(parameters)


#parameters = [3, 1]
#t_max = 10000
#times = np.arange(0, t_max, .1)
#
#vals = model.simulate(parameters, times)
#
#plt.figure(figsize=(16, 5))
#plt.xlabel('Time (ms)')
#plt.ylabel('Voltage (mV)')
#plt.plot(times, vals)
#plt.show()
run_fit()
