import myokit
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
#from vc_opt_ga import simulate_model


class VCProtocol():
    def __init__(self, segments):
        self.segments = segments #list of VCSegments

    def get_protocol_length(self):
        proto_length = 0
        for s in self.segments:
            proto_length += s.duration
        
        return proto_length

    def get_myokit_protocol(self, scale=1):
        segment_dict = {'v0': f'{-82*scale}'}
        piecewise_txt = f'piecewise((engine.time >= 0 and engine.time < {500*scale}), v0, '
        current_time = 500*scale

        #piecewise_txt = 'piecewise( '
        #current_time = 0
        #segment_dict = {}

        for i, segment in enumerate(self.segments):
            start = current_time
            end = current_time + segment.duration
            curr_step = f'v{i+1}'
            time_window = f'(engine.time >= {start} and engine.time < {end})'
            piecewise_txt += f'{time_window}, {curr_step}, '

            if segment.end_voltage is None:
                segment_dict[curr_step] = f'{segment.start_voltage}'
            else:
                slope = ((segment.end_voltage - segment.start_voltage) /
                                                                segment.duration)
                intercept = segment.start_voltage - slope * start

                segment_dict[curr_step] = f'{slope} * engine.time + {intercept}'
            
            current_time = end
        
        piecewise_txt += 'vp)'

        return piecewise_txt, segment_dict, current_time
        
    def plot_protocol(self, is_shown=False):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        pts_v = []
        pts_t = []
        current_t = 0
        for seg in self.segments:
            pts_v.append(seg.start_voltage)
            if seg.end_voltage is None:
                pts_v.append(seg.start_voltage)
            else:
                pts_v.append(seg.end_voltage)
            pts_t.append(current_t)
            pts_t.append(current_t + seg.duration)

            current_t += seg.duration

        plt.plot(pts_t, pts_v)

        if is_shown:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Time (ms)', fontsize=16)
            ax.set_xlabel('Voltage (mV)', fontsize=16)

            plt.show()

    def plot_with_curr(self, curr, cm=60):
        mod = myokit.load_model('mmt-files/kernik_2019_NaL_art.mmt')

        p = mod.get('engine.pace')
        p.set_binding(None)

        c_m = mod.get('artifact.c_m')
        c_m.set_rhs(cm)

        v_cmd = mod.get('artifact.v_cmd')
        v_cmd.set_rhs(0)
        v_cmd.set_binding('pace') # Bind to the pacing mechanism

        # Run for 20 s before running the VC protocol
        holding_proto = myokit.Protocol()
        holding_proto.add_step(-81, 30000)
        t = holding_proto.characteristic_time()
        sim = myokit.Simulation(mod, holding_proto)
        dat = sim.run(t)
        mod.set_state(sim.state())

        # Get protocol to run
        piecewise_function, segment_dict, t_max = self.get_myokit_protocol()
        mem = mod.get('artifact')

        for v_name, st in segment_dict.items():
            v_new = mem.add_variable(v_name)
            v_new.set_rhs(st)

        vp = mem.add_variable('vp')
        vp.set_rhs(0)

        v_cmd = mod.get('artifact.v_cmd')
        v_cmd.set_binding(None)
        vp.set_binding('pace')

        v_cmd.set_rhs(piecewise_function)
        times = np.arange(0, t_max, 0.1)
        ## CHANGE THIS FROM holding_proto TO SOMETHING ELSE
        sim = myokit.Simulation(mod, holding_proto)
        dat = sim.run(t_max, log_times=times)

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
        axs[0].plot(times, dat['membrane.V'])
        axs[0].plot(times, dat['artifact.v_cmd'], 'k--')
        axs[1].plot(times, np.array(dat['artifact.i_out']) / cm)
        axs[2].plot(times, dat[curr])

        axs[0].set_ylabel('Voltage (mV)')
        axs[1].set_ylabel('I_out (A/F)')
        axs[2].set_ylabel(curr)
        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.show()


class VCSegment():
    def __init__(self, duration, start_voltage, end_voltage=None):
        self.duration = duration
        self.start_voltage = start_voltage
        self.end_voltage = end_voltage


def get_single_ap(t, v):
    t = np.array(t)
    max_t = t[-1]

    min_v, max_v = np.min(v), np.max(v)

    if (max_v - min_v) < 10:
        sample_end = np.argmin(np.abs(t - 1100))
        new_t = t[0:sample_end] -100
        new_t = [-100, 1000]
        new_v = [v[0], v[0]]
        return new_t[0:2000], new_v[0:2000]

    dvdt_peaks = find_peaks(np.diff(v), distance=400, height=.2)[0]
    if dvdt_peaks.size == 0:
        sample_end = np.argmin(np.abs(t - 1100))
        new_t = t[0:sample_end] -100
        new_t = [-100, 1000]
        new_v = [v[0], v[0]]
        return new_t, new_v

    start_idx = dvdt_peaks[int(len(dvdt_peaks) / 2)]
        
    ap_start = np.argmin(np.abs(t - t[start_idx]+100))
    ap_end = np.argmin(np.abs(t - t[start_idx]-2000))

    new_t = t[ap_start:ap_end] - t[start_idx]
    new_v = v[ap_start:ap_end]

    return new_t, new_v 
