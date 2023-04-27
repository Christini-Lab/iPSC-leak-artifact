#!/usr/bin/env python3
#
# Fitting I/O methods
# Adapted from: https://github.com/CardiacModelling/fitting-notebooks
#
import glob
import os

import numpy as np

import pints

from . import data


class reserve_base_name(object):
    """
    Context manager that reserves a location for storing results, but deletes
    any partial results if an error occurs.

    A template path is specified by the user, for example
    ``output/result.txt``. Upon entering, this is converted to a numbered path,
    for example ``output/result-i.txt``, such that ``i`` equals one plus the
    highest indice already found in the same directory. To "reserve" the path,
    a file is placed at ``output/result-i.txt``, which can be overwritten by
    the user. Finally, the path to the numbered file is returned.

    If an exception occurs within the manager's context, the numbered file is
    deleted, **along with any files starting with the same basename as the
    numbered path**. For example, if the numbered path is ``result-001.txt``,
    files such as ``result-001-log.dat`` will also be deleted.

    Example::

        with reserve_base_name('output/result.txt') as basename:
            # Write to output/result-001.txt
            with open(basename + '.txt', 'w') as f:
                f.write('Writing stuff')

        with reserve_base_name('output/result.txt') as basename:
            # Write to output/result-002-log.txt
            with open(basename + '-log.txt', 'w') as f:
                f.write('Writing stuff')

    Parameters
    ----------
    template_path
        A template for the path to store results at: to store results such as
        ``results-1.txt``, ``results-2.txt`` etc., pass in the template
        ``name='results.txt'``.

    """
    def __init__(self, template_path):

        # Split path into directory, basename, and extension
        dirname, basename = os.path.split(template_path)
        self._dirname = dirname
        self._basename, self._extension = os.path.splitext(basename)

        # Indice, as integer
        self._indice = None

        # Indice formatting (must be fixed width and start with hyphen)
        self._format = '-{:03d}'
        self._nformat = 4

    def __enter__(self):

        # Find potential indice
        fs = glob.glob(os.path.join(self._dirname, self._basename + '*'))
        if fs:
            i1 = len(self._basename) + 1
            i2 = i1 + self._nformat - 1
            fs = [int(os.path.basename(f)[i1:i2]) for f in fs]
            indice = max(fs)
        else:
            indice = 0

        # Make reservation
        running = True
        while running:
            indice += 1
            path = self._basename + self._format.format(indice)
            path = os.path.join(self._dirname, path + self._extension)
            f = None
            try:
                f = open(path, 'x')     # Note: Python 3.3+ only
                f.write('Reserved\n')
                running = False
            except FileExistsError:
                # File already exists, try next indice
                pass
            finally:
                if f is not None:
                    f.close()

        # Store indice
        self._indice = indice

        # Update stored basename
        self._basename += self._format.format(indice)

        # Return numbered path
        return os.path.join(self._dirname, self._basename + self._extension)

    def __exit__(self, exc_type, exc_val, exc_tb):

        # No exception? Then exit without deleting
        if exc_type is None:
            return

        # Delete files matching pattern
        pattern = os.path.join(self._dirname, self._basename + '*')
        for path in glob.glob(pattern):
            print('Removing unfinished result file: ' + path)
            os.remove(path)

        # Don't suppress the exception
        return False


def save(path, parameters, error, time, iterations, evaluations):
    """
    Stores a result at the given ``path``.

    Parameters
    ----------
    path
        The path to store the result at, e.g. ``output/result-123.txt``.
    parameters
        A list of paramater values.
    error
        The corresponding error (or likelihood or score).
    time
        The time taken to reach the result.
    iterations
        The number of iterations performed.
    evaluations
        The number of function evaluations performed.

    """
    error = float(error)
    time = float(time)
    iterations = int(iterations)
    evaluations = int(evaluations)

    print('Writing results to ' + str(path))
    with open(path, 'w') as f:
        f.write('error: ' + pints.strfloat(error).strip() + '\n')
        f.write('time: ' + pints.strfloat(time).strip() + '\n')
        f.write('iterations: ' + str(iterations) + '\n')
        f.write('evaluations: ' + str(evaluations) + '\n')
        f.write('parameters:\n')
        for p in parameters:
            f.write('    ' + pints.strfloat(p) + '\n')
    print('Done')


def load(template_path, n_parameters):
    """
    Loads and returns all results stored at a given ``template_path``.

    Parameters
    ----------
    template_path
        A template path, e.g. ``output/results.txt``, such that results can be
        found at ``output/results-001.txt``, ``output/results-002.txt``, etc.

    Returns
    -------
    A tuple ``(parameters, info)``, where ``parameters`` is a numpy array
    (with shape ``(n_entries, n_parameters)``) containing all obtained
    parameter sets, and where ``info`` is a numpy array containing one row per
    entry, and each row is structured as ``(run, error, time, iterations,
    evaluations)``. Both arrays are ordered by error (lowest error first).
    """
    # Split path into directory, base ('results'), and extension ('.txt')
    dirname, filename = os.path.split(template_path)
    basename, ext = os.path.splitext(filename)

    # Create pattern to find result files
    pattern = os.path.join(dirname, basename + '-*.txt')

    # Create empty lists
    parameters = []
    info = []

    # Find and process matching files
    for path in glob.glob(pattern):

        # Get run index from filename
        filename = os.path.split(path)[1]
        run = os.path.splitext(filename)[0]
        try:
            run = int(run.rsplit('-', 1)[1])
        except ValueError:
            print('Unable to parse filename, skipping ' + filename)
            continue

        # Naively parse file, warn and skip unparseable files
        error = time = iters = evals = params = None
        try:
            todo = 5
            with open(path, 'r') as f:
                for i in range(100):    # Give up after 100 lines
                    line = f.readline().strip()
                    if line.startswith('error:'):
                        error = float(line[6:])
                        todo -= 1
                    elif line.startswith('time:'):
                        time = float(line[5:])
                        todo -= 1
                    elif line.startswith('iterations:'):
                        iters = int(line[11:])
                        todo -= 1
                    elif line.startswith('evaluations:'):
                        evals = int(line[12:])
                        todo -= 1
                    elif line == 'parameters:':
                        params = [
                            float(f.readline()) for j in range(n_parameters)]
                        todo -= 1
                    if todo == 0:
                        break
                if todo:
                    print('Unable to find all information, skipping '
                          + filename)
                    continue

        except Exception as e:
            print('Error when parsing file, skipping ' + filename)
            print(e)
            continue

        # Store
        parameters.append(params)
        info.append(np.array([run, error, time, iters, evals]))

    # Convert to arrays
    parameters = np.array(parameters)
    info = np.array(info)

    # Sort by error
    if len(parameters) > 0:
        order = np.argsort(info[:, 1])
        parameters = parameters[order]
        info = info[order]

    return parameters, info


def count(template_path, n_parameters, parse=True):
    """
    Counts the number of results matching the given ``template_path``.

    Parameters
    ----------
    template_path
        A template path, e.g. using ``result.txt`` will count the number of
        files named ``result-x.txt`` where ``x`` can be parsed to an integer.
    n_parameters
        The expected number of parameters in each result file. This will be
        ignored if ``parse`` is ``False``.
    parse
        If set to ``True``, this method will read all files matching the
        template, and so count the number of valid, parseable files. If set to
        false any files matching the template will be counted, regardless of
        their content.
    """
    # Load and count all files
    if parse:
        parameters, info = load(template_path, n_parameters)
        return len(parameters)

    # Scan for files matching the template
    n = 0
    base, ext = os.path.splitext(template_path)
    pattern = base + '-*' + ext
    for path in glob.glob(pattern):
        # Chop off extension, and start of path
        path = os.path.splitext(path)[0]
        path = path[len(base) + 1:]

        # Attempt to parse as number
        try:
            int(path)
        except ValueError:
            continue
        n += 1

    return n


def fit(name, error, boundaries, transformation=None, repeats=1, cap=None):
    """
    Minimises the given ``error``, and stores the results in the directory
    ``name``.

    All files are called ``results-i.txt``, with ``i`` automatically increased
    until an available filename is found. Optimisations are run until either
    (1) the requested number of ``repeats`` is reached, or (2) until the
    specified directory contains ``cap`` results.

    Parameters
    ----------
    name
        The directory to store results in (a string). Will be created if it
        does not exist.
    error
        A ``pints.ErrorMeasure`` to minimise (or a ``pints.LogLikelihood`` to
        maximise).
    boundaries
        A boundaries object, used to constrain the search and to sample initial
        starting points.
    transformation
        An optional :class:`pints.Transformation` to pass to the optimiser.
    repeats
        The maximum number of optimisations to run (default is 1).
    cap
        The maximum number of results to obtain in the given directory (default
        is ``None``, for unlimited).

    """
    debug = False

    # Ensure directory exists
    if not os.path.isdir(name):
        os.makedirs(name)

    # Create a template path
    template_path = os.path.join(name, 'result.txt')

    # Get the number of parameters
    n_parameters = error.n_parameters()

    # Check the number of repeats
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError('Number of repeats must be at least 1.')

    # Check the cap on total number of runs
    if cap is not None:
        cap = int(cap)
        if cap < 1:
            raise ValueError(
                'Cap on total number of runs must be at least 1 (or None).')

    # Run
    for i in range(repeats):

        # Cap the maximum number of runs
        cap_info = ''
        if cap:
            n = count(template_path, n_parameters=n_parameters, parse=False)
            if n >= cap:
                print()
                print('Maximum number of runs reached: terminating.')
                print()
                return
            cap_info = ' (run ' + str(n + 1) + ', capped at ' + str(cap) + ')'

        # Show configuration
        print()
        print('Repeat ' + str(1 + i) + ' of ' + str(repeats) + cap_info)
        print()

        # Get base filename to store results in
        with reserve_base_name(template_path) as path:
            print('Storing results in ' + path)

            # Choose starting point
            # Allow resampling, in case error calculation fails
            print('Choosing starting point')
            p0 = s0 = float('inf')
            while not np.isfinite(s0):
                p0 = boundaries.sample(1)[0]
                s0 = error(p0)

            # Create a file path to store the optimisation log in
            log_path = os.path.splitext(path)
            log_path = log_path[0] + '-log.csv'

            # Create optimiser
            opt = pints.OptimisationController(
                error,
                p0,
                boundaries=boundaries,
                transformation=transformation,
                method=pints.CMAES,
            )
            opt.set_log_to_file(log_path, csv=True)
            opt.set_max_iterations(3 if debug else None)
            opt.set_parallel(True)

            # Run optimisation
            print('Running')
            with np.errstate(all='ignore'):  # Ignore numpy warnings
                p, s = opt.run()

            # Store results for this run
            time = opt.time()
            iters = opt.iterations()
            evals = opt.evaluations()
            save(path, p, s, time, iters, evals)


def show_summary(parameters, info):
    """ Shows a summary of the results obtained with :meth:`load`. """

    print('Total results found: ' + str(len(parameters)))
    if len(parameters) > 0:
        print('Best score : ' + str(info[0, 1]))
        print('Worst score: ' + str(info[-1, 1]))
        print('Mean: ' + str(np.mean(info[:, 1])))
        print('Std : ' + str(np.std(info[:, 1])))


def cmd(title):
    """
    Parses command line args, returns a tuple ``()``
    """
    # Get model to fit
    import argparse
    parser = argparse.ArgumentParser(title)
    parser.add_argument(
        'model',
        nargs='?',
        choices=['paci', 'kernik'],
        default='paci',
        help='The model to use')
    parser.add_argument(
        'protocol',
        nargs='?',
        choices=['B', 'O', 'vcp_opt'],
        default='B',
        help='The protocol to use')
    parser.add_argument(
        'data',
        nargs='?',
        choices=data.data_sets(),
        default='synth1',
        help='The data to use')
    args = parser.parse_args()

    # Experiment name
    name = f'results-test-{args.model}-{args.protocol}-{args.data}'

    return args.model, args.protocol, args.data, name
