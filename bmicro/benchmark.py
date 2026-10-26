"""
Benchmark for the Evaluate action.

Run it against a local measurement file and its saved session::

    python -m bmicro.benchmark MEASUREMENT.h5 MEASUREMENT.session.h5 \\
        --repetition 1 --repeats 3

For every run the command measures three steps separately:

session load
    ``Session.set_file``: opens the measurement and loads the saved session
    (extraction, calibration, peak selection, and evaluation settings).
calculation
    ``EvaluationController.evaluate()`` on a freshly loaded session.
full Evaluate
    ``EvaluationView.evaluate(blocking=True)``, the real Evaluate button
    path, on a freshly loaded session (offscreen Qt). This includes the
    worker thread, the progress timer, and the plot refreshes.

By default bmlab chooses serial or parallel execution and the number of
worker processes itself. ``--workers N`` forces parallel execution with
exactly N worker processes, ``--workers serial`` forces serial execution.
This makes it possible to compare worker counts on one computer.

For a parallel run the report also shows how long the main process waited
for results and how busy the workers were (measured by bmlab). A run is
marked "parent-bound" if the main process waits less than 10 % of the
wall time: then the main process, not the number of workers, limits the
speed.

The saved repetition and settings are used as they are. The command never
saves the session and never writes to either file. It compares size and
modification time of both files before and after the run and fails if they
changed.

The benchmark is not part of the automated test run.
"""
import argparse
import json
import logging
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

STEPS = ('session_load', 'calculation', 'full_evaluate')
STEP_LABELS = {
    'session_load': 'Session load',
    'calculation': 'Calculation',
    'full_evaluate': 'Full Evaluate',
}

# A run is parent-bound if the main process waits for results less than
# this share of the wall time.
PARENT_BOUND_WAIT_SHARE = 0.10

# Shown when the worker processes of the evaluation were not observed.
WORKER_STARTUP = 'n/a (worker processes were not observed)'


class WorkerObserver:
    """
    Records the pools of worker processes the evaluation starts.

    bmlab does not report its workers, so the observer wraps the one place
    where bmlab creates a pool (``bmlab.parallel_fit.create_spawn_pool``)
    for the time of the benchmark and records the number of workers and
    the time the creation took. A pool that fails to start is only counted
    in ``failed_starts``. With a bmlab that has no worker processes
    nothing is recorded.
    """

    def __init__(self):
        self.pools = []
        self.failed_starts = 0
        self.supported = False
        self._module = None
        self._original = None

    def __enter__(self):
        try:
            from bmlab import parallel_fit
        except ImportError:
            return self
        self._module = parallel_fit
        self._original = parallel_fit.create_spawn_pool
        self.supported = True

        def create_spawn_pool(workers, *args, **kwargs):
            start = time.perf_counter()
            try:
                pool = self._original(workers, *args, **kwargs)
            except Exception:
                self.failed_starts += 1
                raise
            self.pools.append((workers, time.perf_counter() - start))
            return pool

        parallel_fit.create_spawn_pool = create_spawn_pool
        return self

    def __exit__(self, *exc):
        if self._module is not None:
            self._module.create_spawn_pool = self._original


def measure_worker_startup(workers, repeats=3):
    """
    Measure the start of a pool of ``workers`` processes, outside of an
    evaluation: the time until the pool exists, and until the workers
    have imported their libraries and answered the first jobs.
    Returns a list of (created, answered) in seconds.
    """
    from bmlab import parallel_fit
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        pool = parallel_fit.create_spawn_pool(workers)
        created = time.perf_counter() - start
        try:
            # Empty batches: the workers only have to start and answer.
            jobs = [pool.apply_async(parallel_fit.fit_batch, ([], None))
                    for _ in range(3 * workers)]
            for job in jobs:
                job.get(timeout=120)
            answered = time.perf_counter() - start
        finally:
            pool.terminate()
            pool.join()
        timings.append((created, answered))
    return timings


def describe_workers(observer, evaluations, startup=None,
                     serial_forced=False):
    """
    Describe the worker processes of the evaluations for the report.
    Returns the one-line text and a dictionary for the JSON report.

    observer: WorkerObserver that was active during the evaluations
    evaluations: number of evaluations that ran (calculation and full
        Evaluate, warm-up runs included)
    startup: result of ``measure_worker_startup`` or None
    serial_forced: True if ``--workers serial`` was given
    """
    if not observer.supported:
        return ('n/a (this bmlab has no worker processes, '
                'serial evaluation)', {'parallel': False})
    if not observer.pools and observer.failed_starts:
        return (f'none started: {observer.failed_starts} pool start(s) '
                'failed, bmlab fell back to serial evaluation',
                {'parallel': False, 'evaluations': evaluations,
                 'failed_starts': observer.failed_starts})
    if not observer.pools and serial_forced:
        return ('none started: serial evaluation (serial was forced '
                'with --workers serial)',
                {'parallel': False, 'evaluations': evaluations})
    if not observer.pools:
        return ('none started: serial evaluation (job below the '
                'threshold for parallel evaluation, or fewer than four '
                'CPUs)', {'parallel': False, 'evaluations': evaluations})
    workers = sorted({w for w, _ in observer.pools})
    created = [t for _, t in observer.pools]
    details = {
        'parallel': True,
        'evaluations': evaluations,
        'pools_started': len(observer.pools),
        'failed_starts': observer.failed_starts,
        'workers': workers,
        'pool_creation_seconds': created,
        'startup_probe_seconds': startup,
    }
    text = (f'parallel, {"/".join(map(str, workers))} worker processes '
            f'(spawn), a new pool for each of {len(observer.pools)} '
            f'of {evaluations} evaluations; pool creation median '
            f'{median(created):.3f} s')
    if observer.failed_starts:
        text += (f'; {observer.failed_starts} pool start(s) failed '
                 '(serial fallback)')
    if startup:
        answered = [a for _, a in startup]
        text += (f'; separate probe: until the workers answered a job, '
                 f'median {median(answered):.3f} s '
                 f'({len(startup)} starts)')
    return text, details


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m bmicro.benchmark',
        description='Time session loading, calculation, and full Evaluate '
                    'for one repetition of a saved session.')
    parser.add_argument('measurement', type=pathlib.Path,
                        help='path of the measurement file (.h5)')
    parser.add_argument('session', type=pathlib.Path,
                        help='path of the saved session file (.session.h5)')
    parser.add_argument('--repetition', default=None,
                        help='repetition key to evaluate '
                             '(default: first repetition in the file)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of timed runs (default: 3)')
    parser.add_argument('--warmup', type=int, default=0,
                        help='number of untimed warm-up runs before the '
                             'timed runs (default: 0, all caches cold at '
                             'the first run)')
    parser.add_argument('--workers', type=_workers_arg, default='auto',
                        help="'auto' (default: bmlab chooses), 'serial', "
                             'or the number of worker processes (2 or '
                             'more) to force parallel execution')
    parser.add_argument('--no-gui', action='store_true',
                        help='skip the full Evaluate timing through '
                             'EvaluationView')
    parser.add_argument('--output', type=pathlib.Path, default=None,
                        help='also write the text report to this file')
    parser.add_argument('--json', type=pathlib.Path, default=None,
                        dest='json_path',
                        help='also write the report as JSON to this file')
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error('--repeats must be at least 1')
    if args.warmup < 0:
        parser.error('--warmup must not be negative')
    return args


def _workers_arg(value):
    if value in ('auto', 'serial'):
        return value
    try:
        workers = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected 'auto', 'serial' or a number, got {value!r}")
    if workers < 2:
        raise argparse.ArgumentTypeError(
            'the number of workers must be at least 2')
    return workers


def fit_execution(workers):
    """
    The bmlab execution for the ``--workers`` option, or None for 'auto'.
    A forced worker count uses the tuned batch size and job limit of bmlab
    for that count.
    """
    if workers == 'auto':
        return None
    from bmlab import parallel_fit
    if workers == 'serial':
        return parallel_fit.serial_execution()
    # bmlab may choose the batch size by the worker count
    batch_size_for = getattr(parallel_fit, 'batch_size_for', None)
    batch_size = batch_size_for(workers) if batch_size_for \
        else parallel_fit.BATCH_SIZE
    return parallel_fit.FitExecution(
        parallel=True, workers=workers,
        batch_size=batch_size,
        max_outstanding_jobs=parallel_fit.OUTSTANDING_JOBS_PER_WORKER
        * workers)


def median(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.median(values)


def summarize(runs):
    """
    Build per-step statistics from a list of per-run timing dictionaries.
    """
    summary = {}
    for step in STEPS:
        timings = [run.get(step) for run in runs]
        timings = [t for t in timings if t is not None]
        summary[step] = {
            'runs': timings,
            'median': median(timings),
            # Median without the first run. Only meaningful if the
            # first run is the only cold one.
            'median_after_first': median(timings[1:])
            if len(timings) >= 3 else None,
        }
    return summary


def summarize_parallel(calculations):
    """
    Describe the main-process waiting time and the worker utilisation of
    the calculation runs.

    calculations: list of (wall_seconds, stats) per run. ``stats`` is the
        ``stats`` dictionary of the bmlab run (``wait_seconds``,
        ``busy_seconds``, ``utilisation``, ...), or None if the run was
        serial or bmlab does not report it.

    Returns None if no run has statistics, otherwise a dictionary with
    the values per run, their medians and the verdict ``parent_bound``.
    """
    runs = []
    for wall, stats in calculations:
        if not stats or not wall:
            continue
        runs.append({
            'wall_seconds': wall,
            'wait_seconds': stats['wait_seconds'],
            'wait_share': stats['wait_seconds'] / wall,
            'utilisation': stats['utilisation'],
            'workers': stats['workers'],
            'batch_size': stats['batch_size'],
            'first_result_seconds': stats.get('first_result_seconds'),
            'parent_bound': stats['wait_seconds'] / wall
            < PARENT_BOUND_WAIT_SHARE,
        })
    if not runs:
        return None
    share = median([run['wait_share'] for run in runs])
    return {
        'runs': runs,
        'median_wait_seconds': median([r['wait_seconds'] for r in runs]),
        'median_wait_share': share,
        'median_utilisation': median([r['utilisation'] for r in runs]),
        'parent_bound': share < PARENT_BOUND_WAIT_SHARE,
    }


def _fmt(seconds):
    return 'n/a' if seconds is None else f'{seconds:.3f} s'


def format_report(report):
    """
    Format the report dictionary as plain text.
    """
    env = report['environment']
    cfg = report['configuration']
    lines = []
    add = lines.append
    add('BMicro Evaluate benchmark')
    add('=' * 25)
    add('')
    add('Input')
    add(f'  Measurement file : {cfg["measurement"]}')
    add(f'  Session file     : {cfg["session"]}')
    add(f'  Repetition       : {cfg["repetition"]}')
    add(f'  Point count      : {cfg["point_count"]}')
    add(f'  Peak mode        : {cfg["nr_brillouin_peaks"]}-peak')
    add(f'  Brillouin regions: {cfg["brillouin_regions"]}')
    add(f'  Rayleigh regions : {cfg["rayleigh_regions"]}')
    add(f'  Source files unchanged: {cfg["inputs_unchanged"]}')
    add('')
    add('Environment')
    add(f'  Date             : {env["date"]}')
    add(f'  Machine          : {env["machine"]}')
    add(f'  CPU              : {env["cpu"]} ({env["cpu_count"]} logical)')
    add(f'  Memory           : {env["memory"]}')
    add(f'  OS               : {env["os"]}')
    add(f'  Python           : {env["python"]}')
    add(f'  BMicro           : {env["bmicro"]}')
    add(f'  bmlab            : {env["bmlab"]}')
    add(f'  numpy / scipy    : {env["numpy"]} / {env["scipy"]}')
    add(f'  h5py / Qt (PyQt6): {env["h5py"]} / {env["qt"]}')
    add('')
    add('Conditions')
    add(f'  Workers setting  : {cfg["workers"]}')
    add(f'  Worker startup   : {report["worker_startup"]}')
    add(f'  Warm-up runs     : {cfg["warmup"]} (untimed)')
    add(f'  Timed runs       : {cfg["repeats"]}')
    add(f'  Cache conditions : {report["cache_conditions"]}')
    add('')
    add('Timings (seconds)')
    for step in STEPS:
        stats = report['summary'][step]
        add(f'  {STEP_LABELS[step]}')
        if not stats['runs']:
            add('    not measured')
            continue
        for i, value in enumerate(stats['runs'], start=1):
            add(f'    run {i}: {_fmt(value)}')
        add(f'    median: {_fmt(stats["median"])}')
        if stats['median_after_first'] is not None:
            add('    median without run 1: '
                f'{_fmt(stats["median_after_first"])}')
    add('')
    cpu_times = report.get('calculation_cpu')
    if cpu_times:
        add('Main-process CPU time of the calculation (all threads, '
            'seconds)')
        for i, value in enumerate(cpu_times, start=1):
            add(f'    run {i}: {_fmt(value)}')
        add(f'    median: {_fmt(median(cpu_times))}')
        add('')
    add('Main process and workers (calculation, parallel runs)')
    parallel = report.get('parallel_stats')
    if parallel is None:
        add('  not available (serial run, or bmlab does not report it)')
    else:
        for i, run in enumerate(parallel['runs'], start=1):
            add(f'    run {i}: main process waited '
                f'{_fmt(run["wait_seconds"])} of '
                f'{_fmt(run["wall_seconds"])} '
                f'({run["wait_share"]:.1%}), worker utilisation '
                f'{run["utilisation"]:.1%} ({run["workers"]} workers, '
                f'batch size {run["batch_size"]}), first result '
                f'{_fmt(run["first_result_seconds"])} after pool start'
                + (' [parent-bound]' if run['parent_bound'] else ''))
        add(f'    median: waited {_fmt(parallel["median_wait_seconds"])} '
            f'({parallel["median_wait_share"]:.1%} of wall time), '
            f'worker utilisation {parallel["median_utilisation"]:.1%}')
        if parallel['parent_bound']:
            add('    Verdict: PARENT-BOUND. The main process waits less '
                f'than {PARENT_BOUND_WAIT_SHARE:.0%} of the wall time, '
                'so more workers will not make the run faster.')
        else:
            add('    Verdict: not parent-bound. The main process waits '
                f'{PARENT_BOUND_WAIT_SHARE:.0%} or more of the wall '
                'time, so workers are the limit.')
    add('')
    add('Result check (last run)')
    check = report['result_check']
    for key, value in check.items():
        add(f'  {key}: {value}')
    add('')
    add('Notes')
    for note in report['notes']:
        add(f'  - {note}')
    return '\n'.join(lines) + '\n'


def _cpu_name():
    if sys.platform == 'darwin':
        try:
            return subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or 'unknown'


def _memory():
    try:
        if sys.platform == 'darwin':
            out = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                 capture_output=True, text=True,
                                 check=True).stdout.strip()
            return f'{int(out) / 2**30:.0f} GiB'
        if hasattr(os, 'sysconf'):
            size = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            return f'{size / 2**30:.0f} GiB'
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 'unknown'


def _package_version(name, fallback):
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return fallback


def collect_environment():
    import h5py
    import numpy
    import scipy
    from PyQt6.QtCore import QT_VERSION_STR
    from bmicro import _version as bmicro_version

    return {
        'date': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'machine': f'{platform.node()} ({platform.machine()})',
        'cpu': _cpu_name(),
        'cpu_count': os.cpu_count(),
        'memory': _memory(),
        'os': platform.platform(),
        'python': platform.python_version(),
        'bmicro': _package_version('bmicro', bmicro_version.version),
        # The _version module of bmlab can report the version of the
        # repository in the working directory, so use package metadata.
        'bmlab': _package_version('bmlab', 'unknown'),
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'h5py': h5py.__version__,
        'qt': QT_VERSION_STR,
    }


def _fingerprint(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def validate_paths(measurement, session_path):
    """
    Check the files exist and belong together. Returns a list of error
    messages, empty if everything is fine.
    """
    errors = []
    for label, path in (('measurement', measurement),
                        ('session', session_path)):
        if not pathlib.Path(path).is_file():
            errors.append(f'The {label} file does not exist: {path}')
    if errors:
        return errors
    from bmlab.session import get_session_file_path
    expected = get_session_file_path(measurement).resolve()
    if expected != pathlib.Path(session_path).resolve():
        errors.append(
            'The session file does not belong to the measurement file. '
            f'bmlab would load "{expected}" for this measurement.')
    return errors


def _load(measurement, repetition):
    from bmlab.session import Session
    session = Session.get_instance()
    session.clear()
    start = time.perf_counter()
    session.set_file(measurement)
    elapsed = time.perf_counter() - start
    keys = session.file.repetition_keys()
    if repetition is None:
        repetition = keys[0]
    if repetition not in keys:
        raise ValueError(
            f'Repetition "{repetition}" not found. Available: {keys}')
    session.set_current_repetition(repetition)
    return session, repetition, elapsed


def _time_calculation():
    """
    Returns the time of the calculation, the statistics of the bmlab
    run (None for serial runs and for a bmlab that reports none), and the
    CPU time of this process (all its threads) during the calculation.
    """
    from bmlab.controllers import EvaluationController
    controller = EvaluationController()
    start = time.perf_counter()
    cpu_start = time.process_time()
    controller.evaluate()
    elapsed = time.perf_counter() - start
    cpu = time.process_time() - cpu_start
    stats = getattr(controller.last_run, 'stats', None)
    return elapsed, stats, cpu


def _select_repetition_in_gui(window, repetition):
    """
    Select the repetition the way a user does, in the data tab.

    Updating the GUI resets the current repetition to the first one in
    the file whenever the repetition list is repopulated, so the
    repetition must be selected after the GUI update.
    """
    from bmlab.session import Session
    window.update_ui()
    combo = window.widget_data_view.comboBox_repetition
    combo.setCurrentText(repetition)
    window.update_ui()
    session = Session.get_instance()
    if session._current_repetition_key != repetition:
        raise RuntimeError(
            f'Could not select repetition "{repetition}" in the GUI.')


def _time_full_evaluate(window, repetition, point_count):
    view = window.widget_evaluation_view
    _select_repetition_in_gui(window, repetition)
    start = time.perf_counter()
    view.evaluate(blocking=True)
    elapsed = time.perf_counter() - start
    # Guard against an Evaluate that returned without processing the
    # selected repetition.
    if not (view.count.value == view.max_count.value == point_count):
        raise RuntimeError(
            'Full Evaluate did not process all points: '
            f'{view.count.value} of {view.max_count.value} done, '
            f'{point_count} expected.')
    return elapsed


def _describe(session):
    pm = session.peak_selection_model()
    evm = session.evaluation_model()
    return {
        'point_count': len(session.get_image_keys()),
        'nr_brillouin_peaks': evm.nr_brillouin_peaks,
        'brillouin_regions': _regions_ghz(pm.get_brillouin_regions()),
        'rayleigh_regions': _regions_ghz(pm.get_rayleigh_regions()),
    }


def _regions_ghz(regions):
    out = []
    for region in regions:
        out.append(f'{region[0] / 1e9:g} to {region[1] / 1e9:g} GHz')
    return ', '.join(out) if out else 'none'


def _result_check(session):
    import numpy as np
    evm = session.evaluation_model()
    shift = np.asarray(evm.results['brillouin_shift_f'])
    finite = np.isfinite(shift)
    check = {
        'brillouin_shift_f shape': tuple(shift.shape),
        'finite values': f'{int(finite.sum())} of {shift.size}',
    }
    if finite.any():
        mean = np.mean(shift[finite]) / 1e9
        check['mean brillouin_shift_f'] = f'{mean:.6f} GHz'
    return check


def run_benchmark(args):
    """
    Run the benchmark and return the report dictionary.
    """
    from bmlab.controllers import EvaluationController
    from bmlab.session import Session
    from bmicro.gui.main import BMicro

    workers_setting = args.workers
    execution = fit_execution(workers_setting)
    previous_execution = EvaluationController._fit_execution
    if execution is not None:
        # Internal switch of bmlab, also used by its tests
        EvaluationController._fit_execution = execution

    measurement = args.measurement
    inputs = (measurement, args.session)
    before = [_fingerprint(p) for p in inputs]

    window = None
    if not args.no_gui:
        window = BMicro()

    runs = []
    calculations = []
    cpu_times = []
    description = None
    check = {}
    repetition = args.repetition
    observer = WorkerObserver()
    evaluations = 0
    with observer:
        try:
            for run_index in range(-args.warmup, args.repeats):
                timed = run_index >= 0
                session, repetition, t_load = _load(measurement, repetition)
                if description is None:
                    description = _describe(session)
                t_calc, stats, cpu = _time_calculation()
                evaluations += 1
                check = _result_check(session)
                run = {'session_load': t_load, 'calculation': t_calc}
                if window is not None:
                    # Fresh session, so Evaluate starts from the same state
                    # as the calculation. This load is not timed.
                    _load(measurement, repetition)
                    run['full_evaluate'] = _time_full_evaluate(
                        window, repetition, description['point_count'])
                    evaluations += 1
                    check = _result_check(Session.get_instance())
                if timed:
                    runs.append(run)
                    calculations.append((t_calc, stats))
                    cpu_times.append(cpu)
                label = 'warm-up' if not timed else f'run {run_index + 1}'
                print(f'[{label}] ' + ', '.join(
                    f'{k}={v:.2f}s' for k, v in run.items()), file=sys.stderr,
                    flush=True)
        finally:
            EvaluationController._fit_execution = previous_execution
            if window is not None:
                window.close()
            Session.get_instance().clear()

    startup = None
    if observer.pools:
        # Outside of the timed evaluations
        startup = measure_worker_startup(observer.pools[0][0])
    worker_text, worker_details = describe_workers(
        observer, evaluations, startup,
        serial_forced=workers_setting == 'serial')

    after = [_fingerprint(p) for p in inputs]
    unchanged = before == after

    if args.warmup:
        cache = (f'{args.warmup} untimed warm-up run(s) before timing, '
                 'so all timed runs are warm.')
    else:
        cache = ('no warm-up. Run 1 is the first access in this process '
                 '(cold). Later runs are warm.')
    cache += (' The operating system file cache is not dropped or '
              'controlled, so run 1 may already be warm at OS level.')

    notes = [
        'Session load is timed once per run. Each measured step starts from '
        'a freshly loaded session.',
        'Calculation is EvaluationController.evaluate() without progress '
        'inputs.',
        'Full Evaluate is EvaluationView.evaluate(blocking=True) under '
        'offscreen Qt. It includes the Qt thread, progress timer, plot '
        'refreshes, and the 0.1 s polling interval of the blocking wait.',
    ]
    if window is None:
        notes.append('Full Evaluate was skipped (--no-gui).')

    description['measurement'] = str(measurement)
    description['session'] = str(args.session)
    description['repetition'] = repetition
    description['repeats'] = args.repeats
    description['warmup'] = args.warmup
    description['workers'] = (
        'auto (bmlab chooses)' if workers_setting == 'auto'
        else 'serial (forced)' if workers_setting == 'serial'
        else f'{workers_setting} (forced)')
    description['inputs_unchanged'] = 'yes' if unchanged else 'NO'

    report = {
        'environment': collect_environment(),
        'configuration': description,
        'worker_startup': worker_text,
        'workers': worker_details,
        'parallel_stats': summarize_parallel(calculations),
        'calculation_cpu': cpu_times,
        'cache_conditions': cache,
        'summary': summarize(runs),
        'result_check': check,
        'notes': notes,
    }
    if not unchanged:
        raise RuntimeError('A source file changed during the benchmark.')
    return report


def main(argv=None):
    args = parse_args(argv)
    errors = validate_paths(args.measurement, args.session)
    if errors:
        for error in errors:
            print(f'error: {error}', file=sys.stderr)
        return 2

    # Needs no display. Must be set before Qt is loaded.
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    logging.disable(logging.CRITICAL)

    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # The GUI module shows unhandled errors in a dialog box. Under
    # offscreen Qt nobody can see or close it, and the benchmark would
    # wait for ever. Errors go to the console instead.
    import bmicro.gui.main  # noqa: F401
    sys.excepthook = sys.__excepthook__

    report = run_benchmark(args)
    text = format_report(report)
    print(text)
    if args.output:
        args.output.write_text(text)
    if args.json_path:
        args.json_path.write_text(
            json.dumps(report, indent=2, default=str) + '\n')
    del app
    return 0


if __name__ == '__main__':
    # Necessary to make multiprocessing work with pyinstaller
    from multiprocessing import freeze_support, set_start_method
    freeze_support()
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    sys.exit(main())
