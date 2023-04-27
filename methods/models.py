#
# PINTS ForwardModel implementation.
#
import os
import warnings

import myokit
import numpy as np
import pints

from . import DIR_METHOD

#_time_units = 'ms'
#_voltage_units = 'mV'
#_current_units = 'A/F'
#_concentration_units = 'mM'
#_capacitance_units = 'pF'

# Model files
_model_dir = os.path.join(DIR_METHOD, '..', 'models')
_model_files = {
    'paci': 'paci-2013-ventricular.mmt',
    'kernik': 'kernik-2019.mmt',
}

# Voltage clamp models
VC_NONE = 0
VC_IDEAL = 1
VC_SIMPLE = 10
VC_FULL = 100
_vc_levels = (VC_NONE, VC_IDEAL, VC_SIMPLE, VC_FULL)

# Simple VC: Artefact variable names
_artefact_variable_names_simple = [
]

# Full VC: Artefact variable names (all in component `voltage_clamp`)
# Parameters are Cm + this + optionally E_leak
_artefact_variable_names_full = [
    'R_series',
    'C_prs',
    'V_offset_eff',
    'Cm_est',
    'R_series_est',
    'C_prs_est',
    'alpha',
    'g_leak',
    'g_leak_est',
]

# Cached artefact models
_artefact_model_simple = None
_artefact_model_full = None


def mmt(model):
    """
    Takes a short name (e.g. "paci") and returns the path to its .mmt file.
    """
    return os.path.join(_model_dir, _model_files[model])


def load_artefact_model(full=False):
    """ Loads and returns an artefact model (or retrieves it from cache). """
    if full:
        global _artefact_model_full
        if _artefact_model_full is None:
            _artefact_model_full = myokit.load_model(os.path.join(
                _model_dir, 'voltage-clamp.mmt'))
        return _artefact_model_full
    else:
        global _artefact_model_simple
        if _artefact_model_simple is None:
            _artefact_model_simple = myokit.load_model(os.path.join(
                _model_dir, 'simplified-voltage-clamp.mmt'))
        return _artefact_model_simple


class Timeout(myokit.ProgressReporter):
    """
    A :class:`myokit.ProgressReporter` that halts the simulation after
    ``max_time`` seconds.
    """
    def __init__(self, max_time):
        self.max_time = float(max_time)

    def enter(self, msg=None):
        self.b = myokit.tools.Benchmarker()

    def exit(self):
        pass

    def update(self, progress):
        return self.b.time() < self.max_time


def prepare(model, vc_level=VC_IDEAL, e_leak=False, c_clamp=True):
    """
    Prepares a :class:`myokit.Model` for use with the ForwardModel classes in
    this module.
    """
    from . import required_currents, optional_currents, concentrations

    if vc_level not in _vc_levels:
        raise ValueError(f'Unknown value {vc_level} for vc_level.')
    if e_leak and vc_level == VC_NONE:
        raise ValueError('e_leak can only be selected with VC')

    # Check model
    model.validate()

    # Check model has name and display_name properties set
    name = model.meta.get('name', None)
    if not name:
        warnings.warn(
            'Property name not set for model. Using "unnamed model" instead.')
        name = model.meta['name'] = 'unnamed model'
    if 'display_name' not in model.meta:
        warnings.warn(
            f'Property display_name not set for model. Using {name} instead.')
        model.meta['display_name'] = name

    # Check major variables exist and have the right units
    model.timex()
    #model.convert_units(time, _time_units)
    vm = model.labelx('membrane_potential')
    #model.convert_units(vm, _voltage_units)

    # Check scaling variables: don't care about the units!
    def check_scale_var(suffix):
        # Check that variable exists
        var = model.label('g' + suffix)
        if var is None:
            var = model.label('P' + suffix)
        if var is None:
            raise myokit.IncompatibleModelError(
                name, f'No variable labelled as g{suffix} or P{suffix} found.')

        # Ensure that variable is a literal constant
        var.clamp()

        return var

    # Total (native) transmembrane current
    i_ion = model.labelx('cellular_current')
    #model.convert_units(i_ion, _current_units)

    # Check currents and conductance/permeabilities
    # Note: This works by looking for labels g_X or P_X, that go with the
    # current I_X. A better way to find the conductances/permeabilities might
    # be to have some structure between the terms, so that we could ask myokit
    # for a variable that scales I_X, and it would then know from the ontology
    # that this meant either a scaling var, or failing that a P or g.
    currents = []
    conductances = []
    for suffix in required_currents:
        var = model.labelx('I' + suffix)
        currents.append(var)
        #model.convert_units(var, _current_units)

        var = check_scale_var(suffix)
        # No conversion! Just make sure it exists
        conductances.append(var)

    # Check optional currents
    for suffix in optional_currents:
        var = model.label('I' + suffix)
        if var is not None:
            currents.append(var)
            #model.convert_units(var, _current_units)
            var = check_scale_var(suffix)
            conductances.append(var)

    # Clamp concentrations
    if c_clamp:
        all_c_clamped = True
        for var, value in concentrations.items():
            var = model.labelx(var)
            #model.convert_units(var, _concentration_units)
            if value is None:
                all_c_clamped = False
            else:
                var.clamp(value)
    else:
        all_c_clamped = False

    # Prepare artefact parameter variables
    artefact_variables = []

    # Apply voltage clamp
    if vc_level == VC_IDEAL:
        # Unbind anything bound to 'pace'
        pace = model.binding('pace')
        if pace is not None:
            pace.set_binding(None)

        # Clamp Vm
        vm.clamp(0)
        vm.set_binding('pace')

    elif vc_level in (VC_SIMPLE, VC_FULL):
        # Make sure something is bound to 'pace'
        pace = model.binding('pace')
        if pace is None:
            pace = vm.component().add_variable_allow_renaming('pace')
            pace.set_rhs(0)
            pace.set_binding('pace')

        # Find membrane capacitance
        # TODO: Handle case where model doesn't specify Cm ?
        cm = model.labelx('membrane_capacitance')
        artefact_variables.append(cm)
        #model.convert_units(cm, _capacitance_units)
        cm.clamp()

        # Add in artefact model
        # TODO: Allow renaming if "voltage_clamp" is already used?
        # Map from voltage clamp model variable names to vars in `model`:
        var_map = {
            'engine.pace': pace,
            'membrane.I_ion': i_ion,
            'membrane.V': vm,
            'cell.Cm': cm,
        }
        am = load_artefact_model(vc_level == VC_FULL)
        model.import_component(am.get('voltage_clamp'), var_map=var_map)
        vc = model.get('voltage_clamp')

        # Add in remaining artefact parameters
        if vc_level == VC_FULL:
            required = _artefact_variable_names_full
        else:
            required = _artefact_variable_names_simple
        if e_leak:
            required.append('E_leak')
        for var in required:
            var = vc.get(var)
            artefact_variables.append(var)

            # Replace any symbolically set params with literal expressions
            var.clamp()

        # Voltage-clamp
        assert(vm.is_state())
        rhs = myokit.Minus(
            myokit.Plus(
                myokit.Name(vc.get('V_p')),
                myokit.Name(vc.get('V_offset_eff'))
            ),
            myokit.Name(vm)
        )
        rhs = myokit.Divide(
            rhs, myokit.Multiply(
                myokit.Name(cm),
                myokit.Name(vc.get('R_series'))
            ))
        rhs = myokit.Minus(
            rhs,
            myokit.Plus(
                myokit.Name(i_ion),
                myokit.Name(vc.get('I_leak'))
            ))
        vm.set_rhs(rhs)

        # Set initial V to -80
        vm.set_state_value(-80)

    # Validate final model
    # TODO: At this point we can use model.validate(True) to remove any unused
    # variables (which will not remove bound variables, but will remove
    # labelled variables). But it won't remove unused states, which might arise
    # as some subnetwork that does not affect the rest of the model. Would need
    # a bit of graph theory here?
    # https://github.com/MichaelClerx/myokit/issues/800
    model.validate()  # remove_unused_variables=True)
    # print(model.code())

    return i_ion, currents, conductances, artefact_variables, all_c_clamped


class VCModel(pints.ForwardModel):
    """
    A :class`pints.ForwardModel` representing a cell under voltage-clamp
    conditions.

    Parameters
    ----------
    model_file
        An ``mmt`` model file for an annotated cell model.
    vc_level
        Set to ``model.VC_IDEAL``, ``model.VC_SIMPLE``, or ``model.VC_FULL`` to
        use ideal, simplified, or full artefact model.
    E_leak
        If using the full voltage-clamp model, this can be used to set an
        estimated leak reversal potential (if not set, the voltage-clamp model
        default will be used).
    max_evaluation_time
        The maximum time (in seconds, as a float) allowed for one call to
        ``simulate()``. Simulations that take longer will be terminated.
    modifier
        A function to modify the Myokit model before the simulation is created.
        This is only applied to the "real" simulation, not in any pre pacing at
        the holding potential. The lambda gets a single model as input, which
        it can modify in place (no return value needed).

    """
    def __init__(self, model_file, vc_level=VC_IDEAL, E_leak=None,
                 max_evaluation_time=60, c_clamp=True, modifier=None):
        if vc_level == VC_NONE:
            raise ValueError('Unclamped vc-level selected.')

        # Check and store arguments
        self._model_file = model_file
        self._model_file_name = os.path.basename(model_file)
        self._vc_level = int(vc_level)
        self._E_leak = None if vc_level == VC_IDEAL else E_leak
        self._max_evaluation_time = max_evaluation_time

        # Load model and apply voltage clamp
        self._model1 = myokit.load_model(model_file)
        prep = prepare(
            self._model1, vc_level, self._E_leak is not None, c_clamp)
        self._i_total = prep[0]         # Total current variable
        self._currents = prep[1]        # Fitted current variables
        self._conductances = prep[2]    # Conductance/permeability variables
        self._artefact_vars = prep[3]   # Artefact variables
        self._linear_combo = prep[4] and (vc_level == VC_IDEAL)
        del(prep)

        # Parameter names
        self._parameter_names = [
            str(v) for v in self._conductances + self._artefact_vars]

        # Protocol variables
        self._dt = None
        self._v_hold = None
        self._t_hold = None
        self._protocol = None
        self._times = None

        # Modify model 2, if required
        if modifier is None:
            self._model2 = self._model1
        else:
            self._model2 = self._model1.clone()
            modifier(self._model2)

        # Create simulations
        self._simulation1 = self._simulation2 = None

        # Cached voltage and currents (ideal VC mode only)
        self._cached_voltage = None
        self._cached_currents = None
        self._cached_remainder = None

        # Get original parameter values
        self._original = [var.eval() for var in self._conductances]

        # Generate artefact parameters
        self._artefact_values = None
        if self._vc_level == VC_FULL:
            self.generate_artefact_parameters()

    def _create_simulations(self):
        """Creates the simulation engines for this model."""
        if self._simulation2 is not None:
            return

        self._simulation1 = myokit.Simulation(self._model1)
        self._simulation2 = myokit.Simulation(self._model2)
        if self._vc_level == VC_FULL:
            self._simulation2.set_tolerance(1e-8, 1e-10)
            self._simulation2.set_max_step_size(1e-2)  # ms
        elif self._vc_level == VC_SIMPLE:
            self._simulation2.set_tolerance(1e-8, 1e-8)
        else:
            self._simulation2.set_tolerance(1e-8, 1e-8)

    def code(self):
        """ Returns code representing the internal model, for debugging. """
        return self._model2.code()

    def current_names(self):
        """
        Returns the names of the (known) current variables in this model, and
        the name of the total current variable.
        """
        return list(self._currents), self._i_total

    def n_parameters(self):
        return len(self._conductances)  # + len(self._artefact_vars)

    def parameter_names(self):
        return list(self._parameter_names)

    def set_protocol(self, p, mask=None, dt=0.1, v_hold=-80, t_hold=0):
        """
        Sets this model's voltage protocol.

        Parameters
        ----------
        p
            A :class:`myokit.Protocol` or a tuple ``(times, values)`` to set a
            fixed-form protocol (where times are in ms and voltages in mV).
        mask
            An optional numpy array used to filter out capacitance artefacts or
            make the fitting problem more difficult.
        dt
            Sampling interval duration (in ms).
        v_hold
            The holding potential to use before applying the protocol (in mV).
        t_hold
            The time to simulate at the holding potential, before applying the
            protocol (in ms).

        """
        self._create_simulations()

        # Set pre-pacing protocol
        self._v_hold = float(v_hold)
        self._t_hold = float(t_hold)
        self._dt = float(dt)
        self._simulation1.set_protocol(myokit.pacing.constant(self._v_hold))

        # Set main protocol
        if isinstance(p, myokit.Protocol):
            self._simulation2.set_protocol(p)
            duration = p.characteristic_time()
        else:
            duration = p[0][-1]
            self._simulation2.set_fixed_form_protocol(*p)

        # Generate logging times
        self._times = np.arange(0, duration, self._dt)
        if mask is not None:
            self._times = self._times[mask]

    def simulate_full(self, parameters=None):
        """ Runs a simulation (no caching) and returns a myokit DataLog. """
        self._create_simulations()

        # Default parameters
        if parameters is None:
            parameters = np.ones(len(self._conductances))

        # Set model parameters (as original_value * scaling)
        for v, x, y in zip(self._conductances, self._original, parameters):
            self._simulation1.set_constant(v, x * y)
            self._simulation2.set_constant(v, x * y)

        # Pre-pace
        self._simulation2.reset()
        if self._t_hold > 0:
            self._simulation1.reset()
            self._simulation1.run(self._t_hold, log=myokit.LOG_NONE)
            self._simulation2.set_state(self._simulation1.state())

        return self._simulation2.run(
            self._times[-1] + 0.02, log_times=self._times)

    def _simulate(self, parameters, extra=None):
        """Internal method to run a (pre-simulation and) simulation."""
        if self._times is None:
            raise RuntimeError('No protocol set; unable to simulate.')
        self._create_simulations()

        # Set model parameters (as original_value * scaling)
        for v, x, y in zip(self._conductances, self._original, parameters):
            self._simulation1.set_constant(v, x * y)
            self._simulation2.set_constant(v, x * y)

        # Pre-pace
        self._simulation2.reset()
        if self._t_hold > 0:
            self._simulation1.reset()
            self._simulation1.run(self._t_hold, log=myokit.LOG_NONE)
            self._simulation2.set_state(self._simulation1.state())

        # Variables to log
        log = [self._i_total]
        if extra is not None:
            log += extra

        try:
            d = self._simulation2.run(
                self._times[-1] + 0.02,
                log_times=self._times,
                log=log,
                progress=Timeout(self._max_evaluation_time),
            ).npview()

            # Convert dict to 2d array
            return np.asarray([d[key] for key in log])
        except (myokit.SimulationError, myokit.SimulationCancelledError):
            # Return array of NaNs
            return np.ones((len(log), len(self._times))) * float('nan')

    def _simulate_and_cache(self):
        """
        Runs a simulation and caches the obtained voltage, plus all fitted
        currents and the total current.
        """
        extra = self._currents + [self._model2.labelx('membrane_potential')]

        x = self._simulate(np.ones(len(self._conductances)), extra)

        self._cached_voltage = x[-1]
        self._cached_currents = x[1:-1]  # All individual I's

        # Calculate total - sum(fitted currents)
        self._cached_remainder = x[0] - np.sum(self._cached_currents, axis=0)
        #print(np.max(np.abs(self._cached_remainder)))

    def simulate(self, parameters=None, times=None):
        """
        Simulate a voltage-clamp experiment with the scalings given in
        ``parameters``.

        Parameters
        ----------
        parameters
            A sequence of scaling factors for the conductances/permeabilities.
        times
            Unused: included only to match PINTS' ForwardModel interface.
        """
        # Default parameters
        if parameters is None:
            parameters = np.ones(len(self._conductances))

        if self._linear_combo:
            # Used cached currents
            if self._cached_currents is None:
                self._simulate_and_cache()
            p = np.asarray(parameters).reshape(-1, 1)
            i = 0 * self._cached_remainder + np.sum(
                p * self._cached_currents, axis=0)
        else:
            # Full simulation
            i = self._simulate(parameters)[0]

        return i

    def times(self):
        """
        Returns a copy of the simulation logging times generated from the
        voltage protocol.
        """
        return np.array(self._times) if self._times is not None else None

    def value(self, label):
        """Returns the initial value of a labelled variable."""
        return self._model2.label(label).eval()

    def variable(self, label):
        """Converts a label to a variable name."""
        return self._model2.label(label).qname()

    def voltage(self, parameters=None):
        """
        Returns the membrane voltage, simulated with the given parameters.

        In ideal VC mode, this is always the same trace.
        """
        if self._linear_combo:
            # Use cached trace
            if self._cached_voltage is None:
                self._simulate_and_cache()
            return np.copy(self._cached_voltage)
        else:
            # Simulate with modified model
            if parameters is None:
                parameters = np.ones(len(self._conductances))
            vm = self._model2.labelx('membrane_potential')
            x = self._simulate(parameters, extra=[vm])
            return x[1]

    def generate_artefact_parameters(
            self, seed=None, smallrs=False, smallcm=False, use_v2=False):
        """
        Generates and applies a new set of parameters for the full voltage
        clamp model.
        """
        if self._vc_level != VC_FULL:
            raise RuntimeError(
                'Artefact parameters can only be set when using the full'
                ' voltage-clamp model.')

        if use_v2:
            self._generate_artefact_parameters_2(seed)
        else:
            self._generate_artefact_parameters(seed, smallrs, smallcm)

        self._create_simulations()
        for var, value in zip(self._artefact_vars, self._artefact_values):
            #print(f'Setting {var} to {value}')
            self._simulation1.set_constant(var, value)
            self._simulation2.set_constant(var, value)

    def _generate_artefact_parameters(
            self, seed=None, smallrs=False, smallcm=False):
        # Taken from Lei et al. 2020's synthetics data study
        # https://github.com/CardiacModelling/VoltageClampModel/
        #   blob/master/herg-syn-study/syn.py

        # Set mean parameters
        p_voffset_mean = 0  # mV
        if smallrs:
            print('Using small Rs')
            p_rseries_mean = 4e-3  # GOhm
        else:
            p_rseries_mean = 12.5e-3  # GOhm
        p_cprs_mean = 4.  # pF
        if smallcm:
            print('Using small Cm')
            # from Ma et al. 2011; doi:10.1152/ajpheart.00694.2011 for INa
            p_cm_mean = 15.8  # pF;
        else:
            p_cm_mean = 98.7109  # pF; from Paci model

        alpha_mean = 0.7  # 70% compensation

        # Set std of the parameters
        std_voffset = 1.5  # mV, see paper 1
        if smallrs:
            std_rseries = 1e-3  # GOhm; LogNormal
        else:
            std_rseries = 2e-3  # GOhm; LogNormal
        std_cprs = 1.0  # pF; LogNormal
        if smallcm:
            std_cm = 2.0  # pF; LogNormal; for 15.8, 23.3
        else:
            std_cm = 5.0  # pF; LogNormal; for 98.7109, 88.7
        std_est_error = 0.05  # 5% error for each estimation?

        # Fix seed
        if seed is not None:
            np.random.seed(seed)
            fit_seed = np.random.randint(0, 2**30)
            print('Generating VC artefact with seed: ', fit_seed)
            np.random.seed(fit_seed)

        # Generate parameter sample
        voffset = np.random.normal(p_voffset_mean, std_voffset)
        rseries_logmean = np.log(p_rseries_mean) - 0.5 * np.log(
            (std_rseries / p_rseries_mean) ** 2 + 1)
        rseries_scale = np.sqrt(np.log(
            (std_rseries / p_rseries_mean) ** 2 + 1))
        rseries = np.random.lognormal(rseries_logmean, rseries_scale)
        cprs_logmean = np.log(p_cprs_mean) - 0.5 * np.log(
            (std_cprs / p_cprs_mean) ** 2 + 1)
        cprs_scale = np.sqrt(np.log((std_cprs / p_cprs_mean) ** 2 + 1))
        cprs = np.random.lognormal(cprs_logmean, cprs_scale)
        cm_logmean = np.log(p_cm_mean) - 0.5 * np.log(
            (std_cm / p_cm_mean) ** 2 + 1)
        cm_scale = np.sqrt(np.log((std_cm / p_cm_mean) ** 2 + 1))
        cm = np.random.lognormal(cm_logmean, cm_scale)
        est_rseries = rseries * (1.0 + np.random.normal(0, std_est_error))
        est_cm = cm * (1.0 + np.random.normal(0, std_est_error))
        est_cprs = cprs * (1.0 + np.random.normal(0, std_est_error))
        alpha = min(1, max(0, np.random.normal(alpha_mean, 0.01)))

        # Lump parameters together
        p = np.array([
            cm,  # pF
            rseries,  # GOhm
            cprs,  # pF
            voffset,  # mV
            est_cm,  # pF
            est_rseries,  # GOhm
            est_cprs,  # pF
            alpha,
        ])

        # Leak
        i_s_mean = 1.0  # -> 1.0 GOhm seal resistance
        i_s_std = 0.1
        i_s_logmean = np.log(i_s_mean) - 0.5 * np.log(
            (i_s_std / i_s_mean) ** 2 + 1)
        i_s_scale = np.sqrt(np.log((i_s_std / i_s_mean) ** 2 + 1))
        i_s = np.random.lognormal(i_s_logmean, i_s_scale)  # scaled in model
        est_i_s = i_s * (1.0 + np.random.normal(0, 0.15))  # gleak* ~ +/- 0.15
        p = np.append(p, [i_s, est_i_s])

        if self._E_leak is not None:
            p.append(float(self._E_leak))

        self._artefact_values = p

    def _generate_artefact_parameters_2(self, seed=None):
        import scipy.stats as stats

        if self._vc_level != VC_FULL:
            raise RuntimeError(
                'Artefact parameters can only be set when using the full'
                ' voltage-clamp model.')

        # Taken from Alex Clark's hiPSC-CM experiments
        p_voffset_mean = 0  # mV
        std_voffset = 1.5  # mV, see paper 1

        lower_rseries = 9e-3
        upper_rseries = 40e-3
        p_rseries_mean = 21e-3  # GOhm
        std_rseries = 9e-3  # GOhm; LogNormal

        p_cprs_mean = 4.  # pF
        std_cprs = 1.0  # pF; LogNormal

        lower_cm = 26.
        upper_cm = 80.
        p_cm_mean = 51.  # pF
        std_cm = 15.

        alpha_mean = 0.7  # 70% compensation

        std_est_error = 0.05  # 5% error for each estimation?

        # Fix seed
        if seed is not None:
            np.random.seed(seed)
            fit_seed = np.random.randint(0, 2**30)
            print('Generating VC artefact with seed: ', fit_seed)
            np.random.seed(fit_seed)

        # Generate parameter sample
        voffset = np.random.normal(p_voffset_mean, std_voffset)
        upper2_rseries = (upper_rseries - p_rseries_mean) / std_rseries
        lower2_rseries = (lower_rseries - p_rseries_mean) / std_rseries
        rseries = stats.truncnorm(
            lower2_rseries,
            upper2_rseries,
            loc=p_rseries_mean,
            scale=std_rseries
        ).rvs(size=1)[0]
        cprs_logmean = np.log(p_cprs_mean) - 0.5 * np.log(
            (std_cprs / p_cprs_mean) ** 2 + 1)
        cprs_scale = np.sqrt(np.log((std_cprs / p_cprs_mean) ** 2 + 1))
        cprs = np.random.lognormal(cprs_logmean, cprs_scale)
        upper2_cm = (upper_cm - p_cm_mean) / std_cm
        lower2_cm = (lower_cm - p_cm_mean) / std_cm
        cm = stats.truncnorm(
            lower2_cm,
            upper2_cm,
            loc=p_cm_mean,
            scale=std_cm
        ).rvs(size=1)[0]
        est_rseries = rseries * (1.0 + np.random.normal(0, std_est_error))
        est_cm = cm * (1.0 + np.random.normal(0, std_est_error))
        est_cprs = cprs * (1.0 + np.random.normal(0, std_est_error))
        alpha = min(1, max(0, np.random.normal(alpha_mean, 0.01)))

        # Lump parameters together
        p = np.array([
            cm,  # pF
            rseries,  # GOhm
            cprs,  # pF
            voffset,  # mV
            est_cm,  # pF
            alpha * est_rseries,  # GOhm
            est_cprs,  # pF
            alpha,
        ])

        # Leak
        i_s_mean = 0.745  # GOhm
        i_s_std = 0.54
        i_s_upper = 2.0
        i_s_lower = .4
        i_s_upper2 = (i_s_upper - i_s_mean) / i_s_std
        i_s_lower2 = (i_s_lower - i_s_mean) / i_s_std
        i_s_inv = stats.truncnorm(
            i_s_lower2,
            i_s_upper2,
            loc=i_s_mean,
            scale=i_s_std
        ).rvs(size=1)[0]  # R (GOhm) -> g (nS, with mV -> pA)
        i_s = 1. / i_s_inv  # scaled in model
        est_i_s = i_s * (1.0 + np.random.normal(0, 0.01))  # from gleak*~+/-5%
        p = np.append(p, [i_s, est_i_s])

        if self._E_leak is not None:
            p.append(float(self._E_leak))

        self._artefact_values = p


class ICModel(pints.ForwardModel):
    """
    A model representing a cell under current-clamp conditions.

    Parameters
    ----------
    model_file
        An ``mmt`` model file for an annotated cell model.
    pre
        The time to pre-pace before the simulation (in ms).
    c_clamp
        True if the concentrations should be clamped (default = True).

    """
    def __init__(self, model_file, pre=0, c_clamp=True):

        # Check and store arguments
        self._model_file = model_file
        self._model_file_name = os.path.basename(model_file)
        self._pre = float(pre)

        # Load model without applying voltage clamp
        self._model = myokit.load_model(model_file)
        prep = prepare(self._model, VC_NONE, False, c_clamp)
        self._i_total = prep[0]         # Total current variable
        self._currents = prep[1]        # Fitted current variables
        self._conductances = prep[2]    # Conductance/permeability variables
        del(prep)

        # Get original parameter values
        self._original = [var.eval() for var in self._conductances]

        # Create simulation
        self._simulation = myokit.Simulation(self._model)
        self._simulation.set_tolerance(1e-8, 1e-8)

    def current_names(self):
        """
        Returns the names of the (known) current variables in this model, and
        the name of the total current variable.
        """
        return list(self._currents), self._i_total

    def n_parameters(self):
        """ Returns the number of parameters in this model. """
        return len(self._conductances)

    def simulate(self, parameters=None):
        """
        Simulate an unpaced current-clamp experiment with the scalings given in
        ``parameters``.

        Parameters
        ----------
        parameters
            A sequence of scaling factors for the conductances/permeabilities
            (or ``None`` to use the default values).

        Returns a tuple ``(times, voltages)``.
        """
        if parameters is None:
            parameters = np.ones(len(self._conductances))

        # Set model parameters (as original_value * scaling)
        for v, x, y in zip(self._conductances, self._original, parameters):
            self._simulation.set_constant(v, x * y)

        # Run simulation
        t, v = self._model.time(), self._model.label('membrane_potential')
        self._simulation.reset()
        if self._pre:
            self._simulation.run(self._pre, log=myokit.LOG_NONE)
            self._simulation.set_time(0)
        d = self._simulation.run(10000, log=[t, v])
        return d[t], d[v]

    def simulate_full(self, parameters=None):
        """ Runs a simulation and logs all variables. """
        if parameters is None:
            parameters = np.ones(len(self._conductances))

        # Set model parameters (as original_value * scaling)
        for v, x, y in zip(self._conductances, self._original, parameters):
            self._simulation.set_constant(v, x * y)

        # Run simulation
        self._simulation.reset()
        return self._simulation.run(10000)

