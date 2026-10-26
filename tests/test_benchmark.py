import pathlib
import shutil

import pytest

from bmicro import benchmark


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def make_report():
    runs = [
        {'session_load': 1.0, 'calculation': 10.0, 'full_evaluate': 12.0},
        {'session_load': 3.0, 'calculation': 30.0, 'full_evaluate': 14.0},
        {'session_load': 2.0, 'calculation': 20.0, 'full_evaluate': 13.0},
    ]
    return {
        'environment': {
            'date': 'today', 'machine': 'host (arm64)', 'cpu': 'cpu',
            'cpu_count': 8, 'memory': '16 GiB', 'os': 'os',
            'python': '3.12', 'bmicro': '1.0', 'bmlab': '2.0',
            'numpy': '1', 'scipy': '1', 'h5py': '3', 'qt': '6',
        },
        'configuration': {
            'measurement': 'm.h5', 'session': 'm.session.h5',
            'repetition': '1', 'point_count': 42,
            'nr_brillouin_peaks': 2, 'brillouin_regions': '4 to 6 GHz',
            'rayleigh_regions': '-2 to 2 GHz', 'inputs_unchanged': 'yes',
            'warmup': 0, 'repeats': 3,
            'workers': 'auto (bmlab chooses)',
        },
        'worker_startup': benchmark.WORKER_STARTUP,
        'cache_conditions': 'cold first run',
        'summary': benchmark.summarize(runs),
        'result_check': {'finite values': '1 of 2'},
        'notes': ['a note'],
    }


def test_parse_args_defaults():
    args = benchmark.parse_args(['m.h5', 'm.session.h5'])
    assert args.measurement == pathlib.Path('m.h5')
    assert args.session == pathlib.Path('m.session.h5')
    assert args.repetition is None
    assert args.repeats == 3
    assert args.warmup == 0
    assert not args.no_gui


def test_parse_args_options():
    args = benchmark.parse_args([
        'm.h5', 'm.session.h5', '--repetition', '1', '--repeats', '5',
        '--warmup', '1', '--no-gui', '--json', 'out.json'])
    assert args.repetition == '1'
    assert args.repeats == 5
    assert args.warmup == 1
    assert args.no_gui
    assert args.json_path == pathlib.Path('out.json')


def test_parse_args_workers():
    parse = benchmark.parse_args
    assert parse(['m.h5', 's.h5']).workers == 'auto'
    assert parse(['m.h5', 's.h5', '--workers', 'serial']).workers == 'serial'
    assert parse(['m.h5', 's.h5', '--workers', '16']).workers == 16


@pytest.mark.parametrize('value', ['1', '0', 'many'])
def test_parse_args_rejects_bad_workers(value):
    with pytest.raises(SystemExit):
        benchmark.parse_args(['m.h5', 's.h5', '--workers', value])


def test_fit_execution_for_the_workers_option():
    from bmlab import parallel_fit
    assert benchmark.fit_execution('auto') is None
    assert not benchmark.fit_execution('serial').parallel
    execution = benchmark.fit_execution(16)
    assert execution.parallel
    assert execution.workers == 16
    assert execution.batch_size == parallel_fit.batch_size_for(16)
    assert benchmark.fit_execution(4).batch_size == parallel_fit.BATCH_SIZE
    assert execution.max_outstanding_jobs == \
        parallel_fit.OUTSTANDING_JOBS_PER_WORKER * 16


def test_benchmark_forces_the_worker_count(tmp_path, water_session):
    """
    --workers forces parallel execution with that many workers, also for
    a job that bmlab would evaluate serially, and restores bmlab after.
    """
    import json
    from bmlab.controllers import EvaluationController
    measurement, session_path = water_session
    json_path = tmp_path / 'report.json'
    code = benchmark.main([
        str(measurement), str(session_path), '--repeats', '1',
        '--no-gui', '--workers', '3', '--json', str(json_path)])
    assert code == 0
    report = json.loads(json_path.read_text())
    assert report['workers']['workers'] == [3]
    # waiting time and worker utilisation of the run
    parallel = report['parallel_stats']
    assert len(parallel['runs']) == 1
    assert parallel['runs'][0]['workers'] == 3
    assert parallel['runs'][0]['wait_seconds'] >= 0
    assert 0 < parallel['median_utilisation']
    assert len(report['calculation_cpu']) == 1
    assert report['configuration']['workers'] == '3 (forced)'
    assert EvaluationController._fit_execution is None


def test_benchmark_error_ends_the_process_with_a_traceback(water_session):
    """
    An error must not open the error dialog of the GUI: under offscreen
    Qt nobody can close it, and the benchmark would wait for ever.
    """
    import subprocess
    import sys
    measurement, session_path = water_session
    process = subprocess.run(
        [sys.executable, '-m', 'bmicro.benchmark', str(measurement),
         str(session_path), '--repetition', 'missing', '--repeats', '1',
         '--no-gui'],
        capture_output=True, text=True, timeout=120,
        stdin=subprocess.DEVNULL)
    assert process.returncode == 1
    assert 'Traceback' in process.stderr
    assert 'Repetition "missing" not found' in process.stderr


def test_parse_args_rejects_zero_repeats():
    with pytest.raises(SystemExit):
        benchmark.parse_args(['m.h5', 'm.session.h5', '--repeats', '0'])


def test_summarize_reports_runs_and_median():
    summary = benchmark.summarize([
        {'session_load': 1.0, 'calculation': 10.0},
        {'session_load': 3.0, 'calculation': 30.0},
        {'session_load': 2.0, 'calculation': 20.0},
    ])
    assert summary['calculation']['runs'] == [10.0, 30.0, 20.0]
    assert summary['calculation']['median'] == 20.0
    assert summary['calculation']['median_after_first'] == 25.0
    # Not measured at all
    assert summary['full_evaluate']['runs'] == []
    assert summary['full_evaluate']['median'] is None


def test_format_report_lists_all_required_items():
    text = benchmark.format_report(make_report())
    for expected in ('Session load', 'Calculation', 'Full Evaluate',
                     'run 1: 10.000 s', 'run 3: 20.000 s',
                     'median: 20.000 s', 'host (arm64)',
                     'BMicro           : 1.0', 'bmlab            : 2.0',
                     '2-peak', 'Point count      : 42',
                     'Worker startup   : n/a', 'Cache conditions'):
        assert expected in text


def test_missing_files_fail_cleanly(tmp_path, capsys):
    code = benchmark.main([str(tmp_path / 'a.h5'),
                           str(tmp_path / 'a.session.h5')])
    assert code == 2
    assert 'does not exist' in capsys.readouterr().err


def test_mismatching_session_is_rejected(tmp_path):
    measurement = tmp_path / 'a.h5'
    other = tmp_path / 'b.session.h5'
    measurement.touch()
    other.touch()
    errors = benchmark.validate_paths(measurement, other)
    assert errors
    assert 'does not belong' in errors[0]


@pytest.fixture
def water_session(tmp_path, monkeypatch):
    """
    The public Water fixture in a temporary folder with a generated
    session. Nothing is written to tests/data.
    """
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from bmlab.controllers import Controller
    from bmlab.models import Orientation
    from bmlab.models.setup import AVAILABLE_SETUPS
    from bmlab.session import Session

    measurement = tmp_path / 'Water.h5'
    shutil.copy(data_file_path('Water.h5'), measurement)
    Controller().evaluate(
        measurement, AVAILABLE_SETUPS[0],
        Orientation(rotation=1, reflection={
            'vertically': False, 'horizontally': False}),
        [(4.0e9, 6.0e9), (9.0e9, 11.0e9)],
        [(-2.0e9, 2.0e9), (13.0e9, 17.0e9)])
    session = Session.get_instance()
    session.save()
    session.clear()
    session_path = tmp_path / 'Water.session.h5'
    assert session_path.exists()
    return measurement, session_path


def test_benchmark_runs_on_public_fixture(tmp_path, water_session):
    """
    End-to-end run on a generated session for the public Water fixture.
    The job is small, so bmlab evaluates it in this process.
    """
    measurement, session_path = water_session

    report_path = tmp_path / 'report.txt'
    code = benchmark.main([
        str(measurement), str(session_path), '--repeats', '2',
        '--output', str(report_path)])
    assert code == 0
    text = report_path.read_text()
    assert 'Full Evaluate' in text
    assert 'Source files unchanged: yes' in text
    assert 'median:' in text
    assert 'Worker startup   : none started: serial evaluation' in text


def test_benchmark_reports_parallel_workers(tmp_path, water_session, mocker):
    """
    With parallel execution forced (the internal switch of bmlab), the
    report names the workers that really ran and their start-up time.
    """
    from bmlab import parallel_fit
    from bmlab.controllers import EvaluationController
    measurement, session_path = water_session
    original_create_pool = parallel_fit.create_spawn_pool
    mocker.patch.object(
        EvaluationController, '_fit_execution',
        parallel_fit.FitExecution(
            parallel=True, workers=2, batch_size=1,
            max_outstanding_jobs=2))

    report_path = tmp_path / 'report.txt'
    json_path = tmp_path / 'report.json'
    code = benchmark.main([
        str(measurement), str(session_path), '--repeats', '1',
        '--no-gui', '--output', str(report_path),
        '--json', str(json_path)])

    assert code == 0
    text = report_path.read_text()
    assert 'Worker startup   : parallel, 2 worker processes (spawn)' in text
    assert 'separate probe' in text
    import json
    workers = json.loads(json_path.read_text())['workers']
    assert workers['parallel']
    assert workers['workers'] == [2]
    assert workers['pools_started'] == 1
    assert len(workers['startup_probe_seconds']) == 3
    # The observer left bmlab as it was
    assert parallel_fit.create_spawn_pool is original_create_pool


def test_describe_workers_without_pools():
    observer = benchmark.WorkerObserver()
    observer.supported = True

    text, details = benchmark.describe_workers(observer, 4)

    assert text.startswith('none started: serial evaluation')
    assert details == {'parallel': False, 'evaluations': 4}


def test_observer_does_not_record_a_pool_that_failed_to_start(mocker):
    from bmlab import parallel_fit
    mocker.patch.object(parallel_fit, 'create_spawn_pool',
                        side_effect=OSError('no processes'))
    observer = benchmark.WorkerObserver()

    with observer:
        with pytest.raises(OSError):
            parallel_fit.create_spawn_pool(2)

    assert observer.pools == []
    assert observer.failed_starts == 1
    text, details = benchmark.describe_workers(observer, 2)
    assert text.startswith('none started: 1 pool start(s) failed')
    assert details['parallel'] is False
    assert details['failed_starts'] == 1


def test_describe_workers_with_pools():
    observer = benchmark.WorkerObserver()
    observer.supported = True
    observer.pools = [(8, 0.5), (8, 0.7), (8, 0.6)]

    text, details = benchmark.describe_workers(
        observer, 3, [(0.5, 1.5), (0.6, 1.7), (0.7, 1.6)])

    assert 'parallel, 8 worker processes (spawn)' in text
    assert '3 of 3 evaluations' in text
    assert 'pool creation median 0.600 s' in text
    assert 'median 1.600 s' in text
    assert details['workers'] == [8]
    assert details['pools_started'] == 3


def test_describe_workers_without_parallel_bmlab():
    text, details = benchmark.describe_workers(
        benchmark.WorkerObserver(), 3)

    assert text.startswith('n/a (this bmlab has no worker processes')
    assert details == {'parallel': False}


def test_describe_workers_says_serial_was_forced():
    observer = benchmark.WorkerObserver()
    observer.supported = True

    text, details = benchmark.describe_workers(
        observer, 4, serial_forced=True)

    assert 'serial was forced' in text
    assert 'threshold' not in text
    assert details == {'parallel': False, 'evaluations': 4}


def test_benchmark_with_serial_workers_reports_forced_serial(
        tmp_path, water_session):
    measurement, session_path = water_session
    report_path = tmp_path / 'report.txt'
    code = benchmark.main([
        str(measurement), str(session_path), '--repeats', '1',
        '--no-gui', '--workers', 'serial', '--output', str(report_path)])
    assert code == 0
    text = report_path.read_text()
    assert 'serial was forced' in text
    assert 'below the threshold' not in text
    assert 'not available (serial run' in text


def stats(wait, workers=16, utilisation=0.9):
    return {'workers': workers, 'batch_size': 8, 'wait_seconds': wait,
            'utilisation': utilisation, 'first_result_seconds': 0.8}


def test_summarize_parallel_marks_a_parent_bound_run():
    summary = benchmark.summarize_parallel([
        (10.0, stats(0.5, utilisation=0.7)),
        (10.0, stats(0.9, utilisation=0.8)),
        (10.0, stats(1.1, utilisation=0.9))])
    assert [run['wait_share'] for run in summary['runs']] == \
        pytest.approx([0.05, 0.09, 0.11])
    assert [run['parent_bound'] for run in summary['runs']] == \
        [True, True, False]
    assert summary['median_wait_seconds'] == 0.9
    assert summary['median_utilisation'] == 0.8
    assert summary['parent_bound']


def test_summarize_parallel_threshold_is_ten_percent_of_wall_time():
    assert not benchmark.summarize_parallel(
        [(10.0, stats(1.0))])['parent_bound']
    assert benchmark.summarize_parallel(
        [(10.0, stats(0.99))])['parent_bound']
    assert not benchmark.summarize_parallel(
        [(10.0, stats(3.0))])['parent_bound']


def test_summarize_parallel_without_statistics():
    assert benchmark.summarize_parallel([]) is None
    assert benchmark.summarize_parallel([(10.0, None)]) is None


def test_format_report_shows_waiting_time_and_utilisation():
    report = make_report()
    report['calculation_cpu'] = [6.0, 7.0]
    report['parallel_stats'] = benchmark.summarize_parallel(
        [(10.0, stats(0.5)), (10.0, stats(2.0, utilisation=0.95))])
    text = benchmark.format_report(report)
    assert 'main process waited 0.500 s of 10.000 s (5.0%)' in text
    assert 'worker utilisation 90.0% (16 workers, batch size 8)' in text
    assert '[parent-bound]' in text
    assert 'Main-process CPU time of the calculation' in text
    assert 'run 2: 7.000 s' in text
    # median wait share 12.5 %: not parent-bound as a whole
    assert 'Verdict: not parent-bound' in text

    report['parallel_stats'] = benchmark.summarize_parallel(
        [(10.0, stats(0.5))])
    assert 'Verdict: PARENT-BOUND' in benchmark.format_report(report)


def test_format_report_without_parallel_statistics():
    text = benchmark.format_report(make_report())
    assert 'not available (serial run, or bmlab does not report it)' \
        in text
