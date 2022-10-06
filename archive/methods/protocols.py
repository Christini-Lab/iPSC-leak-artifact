#
# Pre-defined voltage-clamp protocols
#
import myokit
import numpy as np
import os

_proto_dir = 'protocols'


def load(filename):
    """
    Creates a :class:`myokit.Protocol` from a step file using
    :meth:`load_steps(filename)` and :meth:`from_steps()`.
    """
    return from_steps(load_steps(filename))


def load_named(*names):
    """
    Loads one or more named OED protocols and returns a tuple
    ``(protocol, dt, modifier)``.

    A 0.1s step at the holding potential (-80mV) is inserted before the first
    protocol. If multiple names are specified, a 5.1s holding potential step is
    inserted between the protocols.
    """
    step_files = {
        'B': 'B-alex.txt',
        'O': 'O-alex.txt',
        'vcp_opt': 'vcp_opt.txt',
    }
    dt = 0.1

    steps = np.array([])
    for i, name in enumerate(names):
        name = os.path.join(_proto_dir, step_files[name])
        steps = np.append(steps, load_steps(name))

    # Model modifier
    modifier = None
    if 'vcp_opt' in names:
        if len(names) > 1:
            raise ValueError(
                'vcp_opt cannot be combined with other protocols.')

        # Method to modify (possibly fancily clamped) model
        def modifier(m):
            # Ramps: t0, dt, dv
            ramps = (
                (1971, 103.7, -32.110413609918844),
                (2757.7, 101.9, 65.36745364287121),
                (3359.6, 272.1, -161.43947608426976),
                (4234.5, 52.2, -4.17731209749318),
                (6429.2, 729.4, 0.6554508694203413),
                (8155.2, 894.9, -30.687646597436324),
            )

            # Unbind pacing variable
            p = m.binding('pace')
            if p is None:
                return
            p.set_binding(None)

            # Introduce new paced varaible
            q = p.parent().add_variable_allow_renaming('pace_new')
            q.set_rhs(0)
            q.set_binding('pace')

            # Replace original pacing variable rhs with ref to new variable
            # and/or ramps.
            tn = m.time().qname()
            # Create list of conditions and values for piecewise
            args = []
            for t0, dt, dv in ramps:
                args.append(f'{tn} >= {t0} and {tn} < {t0 + dt}')
                args.append(f'({tn} - {t0}) * {dv / dt}')
            args.append('0')
            p.set_rhs(q.name() + ' + piecewise(' + ', '.join(args) + ')')

    return from_steps(steps), dt, modifier


def load_steps(filename):
    """ Loads a list of voltage protocol steps from ``filename``. """
    return np.loadtxt(filename).flatten()


def from_steps(p):
    """
    Takes a list ``[step_1_voltage, step_1_duration, step_2_voltage, ...]`` and
    returns a :clas:`myokit.Protocol`.
    """
    protocol = myokit.Protocol()
    for i in range(0, len(p), 2):
        protocol.add_step(p[i], p[i + 1])
    return protocol


def generate_filter(p, dt=1, duration=5):
    """
    Generates a capacitance filter based on a :class:`myokit.Protocol`: each
    ``duration`` time units after every step will be filtered out.
    """
    times = np.arange(0, duration, dt)
    mask = np.ones(times.shape, dtype=bool)
    for step in p:
        mask *= times < p.start
        mask *= times >= p.start + p.duration
    return mask

