"""
Lifecycle tests for the Evaluate action of the evaluation view.

These tests drive ``EvaluationView.evaluate`` the way a user does (the
Evaluate/Cancel button) and check observable behaviour only: results,
progress, button state, plot data, saved files, threads and child
processes. They do not depend on how the calculation is executed, so
every test runs twice: with serial evaluation and with parallel
evaluation in real spawned worker processes.

Execution mode
--------------
The public fixtures are far below the size at which bmlab starts workers
on its own. The ``execution`` fixture forces the mode through the internal
switch ``EvaluationController._fit_execution`` (not a user setting, see
bmlab ``FitExecution``). The reference results are always computed with
serial execution, so parallel runs are compared with serial results.

Test seam
---------
The public fixtures are tiny: a whole run takes a fraction of a second.
To act in the middle of a run, ``run_gate`` holds the extraction of one
spectrum until the test releases it. Extraction stays in the controlling
process (and in the worker thread of the GUI) in every execution mode, so
the gate works for serial and for parallel evaluation. Everything that is
held by the gate has a timeout, so a failing test cannot hang.

Test data
---------
The public fixtures in ``tests/data`` are copied to a temporary folder.
A session without results is generated once per test run with the bmlab
building blocks. Nothing is written to ``tests/data``.

Known behaviour that these tests work around
---------------------------------------------
``BMicro.update_ui`` resets the current repetition to the first one when
the repetition combo box is repopulated. Tests therefore
select the repetition after the GUI update, see ``select_repetition``.
"""
import dataclasses
import gc
import json
import logging
import multiprocessing
import multiprocessing.pool
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import weakref

import numpy as np
import psutil
import pytest
from PyQt6 import QtCore, QtWidgets, uic
from PyQt6.QtCore import QThread

from bmlab import parallel_fit
from bmlab.controllers import (
    CalibrationController, EvaluationController, ExtractionController,
    PeakSelectionController)
from bmlab.models import Orientation
from bmlab.models.setup import AVAILABLE_SETUPS
from bmlab.session import Session

from bmicro.gui import evaluation as evaluation_package
from bmicro.gui import main as main_module
from bmicro.gui.main import BMicro

evaluation_view_module = evaluation_package.evaluation_view

BRILLOUIN_REGIONS = [(4.0e9, 6.0e9), (9.0e9, 11.0e9)]
RAYLEIGH_REGIONS = [(-2.0e9, 2.0e9), (13.0e9, 17.0e9)]

# Public fixtures: file name and the repetition keys of each
WATER = 'Water.h5'
TWO_REPETITIONS = 'Water_two_repetitions.h5'

# Parallel evaluation of the tests: two workers, one point per job and two
# jobs in flight. The run then hands results back as soon as the workers
# return them, and the producer waits for a free slot while it extracts, so
# a run held in the extraction of a later point has stored earlier points.
PARALLEL_EXECUTION = parallel_fit.FitExecution(
    parallel=True, workers=2, batch_size=1, max_outstanding_jobs=2)

# Extraction calls to let through before the gate holds. The first call
# only determines the number of images per point, so the run is held while
# extracting the third measurement point. All fixtures have more than
# three points.
HOLD_AFTER = 3

# Hold later in the run, for tests that need stored points before the hold
# (the Water fixture has ten points). In parallel mode the producer only
# gets this far after earlier results were handed back and stored.
LATE_HOLD = 8

# Upper bound for every wait, in milliseconds. A healthy test is far
# faster; the bound only keeps a broken test from hanging.
TIMEOUT = 20000


def data_file_path(file_name):
    return pathlib.Path(__file__).parent.parent / 'data' / file_name


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def prepare_session(path):
    """
    Create the session file next to ``path`` for all repetitions: extraction
    points, calibration, and peak selection, but no evaluation results.
    """
    session = Session.get_instance()
    session.clear()
    session.set_file(path)
    for repetition in session.file.repetition_keys():
        session.set_current_repetition(repetition)
        session.set_setup(AVAILABLE_SETUPS[0])
        session.orientation = Orientation(
            rotation=1,
            reflection={'vertically': False, 'horizontally': False})
        ExtractionController().find_points_all()
        calibration = CalibrationController()
        for calib_key in session.get_calib_keys():
            calibration.find_peaks(calib_key)
            calibration.calibrate(calib_key)
        peak_selection = PeakSelectionController()
        for region in BRILLOUIN_REGIONS:
            peak_selection.add_brillouin_region_frequency(region)
        for region in RAYLEIGH_REGIONS:
            peak_selection.add_rayleigh_region_frequency(region)
    session.save()
    session.clear()


@pytest.fixture(scope='session')
def prepared_dir(tmp_path_factory):
    """Measurement files with generated sessions, never modified by tests."""
    folder = tmp_path_factory.mktemp('prepared')
    for name in (WATER, TWO_REPETITIONS):
        shutil.copy(data_file_path(name), folder / name)
        prepare_session(folder / name)
    return folder


@pytest.fixture
def files(prepared_dir, tmp_path):
    """A fresh writable copy of the prepared files for one test."""
    folder = tmp_path / 'work'
    shutil.copytree(prepared_dir, folder)
    return folder


def snapshot(session):
    """
    Copy everything the evaluation produced for the current repetition.
    """
    evm = session.evaluation_model()
    return {
        'results': {k: np.array(v, copy=True)
                    for k, v in evm.results.items()},
        'spectra': {k: [np.array(s, copy=True) for s in v]
                    for k, v in evm.spectra.items()},
        'frequencies': {k: [np.array(f, copy=True) for f in v]
                        for k, v in evm.frequencies.items()},
        'positions': {k: np.array(v, copy=True) for k, v in
                      session.get_payload_positions().items()},
    }


def assert_same_snapshot(actual, expected):
    """
    Exact comparison of every result array. NaN and infinity must sit at
    the same places; there is no tolerance.
    """
    assert actual['results'].keys() == expected['results'].keys()
    for key, array in expected['results'].items():
        assert actual['results'][key].shape == array.shape, key
        np.testing.assert_array_equal(actual['results'][key], array, key)
    for name in ('spectra', 'frequencies'):
        assert actual[name].keys() == expected[name].keys(), name
        for image_key, items in expected[name].items():
            assert len(actual[name][image_key]) == len(items)
            for a, e in zip(actual[name][image_key], items):
                np.testing.assert_array_equal(a, e)
    assert actual['positions'].keys() == expected['positions'].keys()
    for key, array in expected['positions'].items():
        np.testing.assert_array_equal(actual['positions'][key], array)


def processed_points(session):
    """Number of measurement points that have a fitted Brillouin peak."""
    results = session.evaluation_model().results
    data = results['brillouin_peak_position_f']
    if data.size == 0:
        return 0
    per_point = data.reshape(-1, *data.shape[3:])
    return int(np.sum(~np.all(np.isnan(per_point), axis=(1, 2, 3))))


def point_count(session):
    return len(session.get_image_keys())


@pytest.fixture(scope='session')
def reference(prepared_dir):
    """
    Results of a clean, uninterrupted run for every file and repetition,
    computed through the evaluation controller on its own.

    Serial execution is forced, so that parallel runs are compared with
    serial results and not with parallel results.
    """
    references = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(EvaluationController, '_fit_execution',
                      parallel_fit.serial_execution())
        for name in (WATER, TWO_REPETITIONS):
            session = Session.get_instance()
            session.clear()
            session.set_file(prepared_dir / name)
            for repetition in session.file.repetition_keys():
                session.set_current_repetition(repetition)
                EvaluationController().evaluate()
                references[(name, repetition)] = snapshot(session)
            session.clear()
    return references


# ---------------------------------------------------------------------------
# Test seams: run gate, thread and process observation
# ---------------------------------------------------------------------------

class Execution:
    """
    How the evaluation of a test is executed. In parallel mode it records
    the pool of every run.
    """

    def __init__(self, mode):
        self.mode = mode
        self.pools = []

    @property
    def parallel(self):
        return self.mode == 'parallel'

    def create_pool(self, workers):
        pool = parallel_fit.create_spawn_pool(workers)
        self.pools.append(pool)
        return pool

    def fit_execution(self):
        if not self.parallel:
            return parallel_fit.serial_execution()
        return dataclasses.replace(
            PARALLEL_EXECUTION, pool_factory=self.create_pool)


def parallel_only(test):
    """Mark a test that only makes sense with worker processes."""
    test.parallel_only = True
    return test


def pytest_generate_tests(metafunc):
    """Run every test that uses ``execution`` in both modes."""
    if 'execution' in metafunc.fixturenames:
        modes = ['serial', 'parallel']
        if getattr(metafunc.function, 'parallel_only', False):
            modes = ['parallel']
        metafunc.parametrize('execution', modes, indirect=True)


@pytest.fixture
def execution(request, mocker):
    """
    Force serial or parallel execution for the evaluation of one test.
    """
    execution = Execution(request.param)
    mocker.patch.object(EvaluationController, '_fit_execution',
                        execution.fit_execution())
    return execution


class RunGate:
    """
    Holds the extraction of one spectrum until ``release`` is called.
    """

    def __init__(self, hold_after):
        self.hold_after = hold_after
        self.calls = 0
        self.reached = threading.Event()
        self.released = threading.Event()
        self._lock = threading.Lock()

    def before_extraction(self):
        with self._lock:
            self.calls += 1
            hold = self.calls == self.hold_after
        if hold:
            self.reached.set()
            if not self.released.wait(timeout=TIMEOUT / 1000):
                raise RuntimeError('Run gate was never released.')

    def release(self):
        self.released.set()


@pytest.fixture
def run_gate(mocker):
    """
    Install the gate in front of spectrum extraction. Yields a function
    that returns the (single) gate; the gate holds on the HOLD_AFTER-th
    extraction call after it was installed.
    """
    gates = []
    original = EvaluationController.extract_spectra

    def install(hold_after=HOLD_AFTER):
        gate = RunGate(hold_after)

        def gated(self, key):
            gate.before_extraction()
            return original(self, key)

        mocker.patch.object(EvaluationController, 'extract_spectra', gated)
        gates.append(gate)
        return gate

    yield install
    # Never leave a worker blocked behind a failed test.
    for gate in gates:
        gate.release()


class ThreadRegistry:
    """Remembers every QThread the evaluation view starts."""

    def __init__(self):
        self.threads = []

    @staticmethod
    def alive(thread):
        try:
            return thread.isRunning()
        except RuntimeError:
            # The C++ object is gone, the thread has finished long ago.
            return False

    def running(self):
        return [t for t in self.threads if self.alive(t)]

    def wait_all(self, timeout=TIMEOUT):
        for thread in self.threads:
            try:
                thread.wait(timeout)
            except RuntimeError:
                pass


@pytest.fixture
def thread_registry(mocker):
    registry = ThreadRegistry()

    class SpyThread(QThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The registry also keeps the Python object alive. Dropping it
            # while the thread still runs would destroy a running QThread.
            registry.threads.append(self)

    mocker.patch.object(evaluation_view_module, 'QThread', SpyThread)
    return registry


def is_infrastructure(process):
    """
    Helper processes of the multiprocessing module that live as long as
    the parent process. They are not workers.
    """
    try:
        command = ' '.join(process.cmdline())
    except psutil.Error:
        return True
    return ('multiprocessing.resource_tracker' in command
            or 'multiprocessing.forkserver' in command)


class ChildProcesses:
    """
    Detects worker processes that were started after the observer was
    created and are still around.
    """

    def __init__(self):
        self.baseline = self._current()

    @staticmethod
    def _current():
        # Reaps finished multiprocessing children as a side effect.
        pids = {p.pid for p in multiprocessing.active_children()}
        pids |= {p.pid for p in psutil.Process().children(recursive=True)
                 if not is_infrastructure(p)}
        return pids

    def stray(self):
        return sorted(self._current() - self.baseline)


@pytest.fixture
def child_processes():
    return ChildProcesses()


def wait_until_idle(qtbot, view, registry):
    """
    Wait until the view reports that evaluation is over and no evaluation
    thread runs any more.
    """
    qtbot.waitUntil(
        lambda: not view.evaluation_running and not registry.running(),
        timeout=TIMEOUT)


def wait_for_no_children(qtbot, child_processes):
    qtbot.waitUntil(lambda: not child_processes.stray(), timeout=TIMEOUT)
    assert child_processes.stray() == []


# ---------------------------------------------------------------------------
# GUI fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def window(qtbot, tmp_path, execution, thread_registry, run_gate,
           child_processes):
    # Keep the application settings of the user untouched.
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope, str(tmp_path / 'settings'))
    window = BMicro()
    qtbot.addWidget(window)
    Session.get_instance().clear()
    yield window
    # Let a still running evaluation end before the window goes away.
    view = window.widget_evaluation_view
    if view.evaluation_running:
        view.evaluation_abort.value = True
    thread_registry.wait_all()
    window.close()
    Session.get_instance().clear()
    # Whatever a test did: no worker process may be left behind
    wait_for_no_children(qtbot, child_processes)


@pytest.fixture
def view(window):
    return window.widget_evaluation_view


def click_evaluate(qtbot, view):
    """Click the Evaluate/Cancel button like a user."""
    qtbot.mouseClick(view.button_evaluate,
                     QtCore.Qt.MouseButton.LeftButton)


def select_repetition(window, repetition):
    """
    Select a repetition like a user does in the data tab. The repetition
    has to be selected after the GUI update, because the update resets the
    current repetition when it repopulates the repetition list.
    """
    window.update_ui()
    combo = window.widget_data_view.comboBox_repetition
    combo.setCurrentText(repetition)
    window.update_ui()
    session = Session.get_instance()
    assert session._current_repetition_key == repetition
    assert combo.currentText() == repetition


def open_measurement(window, files, name, repetition='0'):
    window.open_file(str(files / name))
    select_repetition(window, repetition)
    return Session.get_instance()


def run_evaluation(qtbot, view, registry):
    """Press Evaluate and wait until the run is over."""
    click_evaluate(qtbot, view)
    wait_until_idle(qtbot, view, registry)


# ---------------------------------------------------------------------------
# Progress and completion
# ---------------------------------------------------------------------------

def test_evaluate_matches_clean_reference(
        qtbot, window, view, thread_registry, files, reference):
    session = open_measurement(window, files, WATER)
    assert view.button_evaluate.text() == 'Evaluate'

    click_evaluate(qtbot, view)
    assert view.evaluation_running
    assert view.button_evaluate.text() == 'Cancel'
    wait_until_idle(qtbot, view, thread_registry)

    assert_same_snapshot(snapshot(session), reference[(WATER, '0')])


def test_progress_reaches_completion(
        qtbot, window, view, thread_registry, files):
    session = open_measurement(window, files, WATER)
    points = point_count(session)

    run_evaluation(qtbot, view, thread_registry)

    assert view.count.value == points
    assert view.max_count.value == points
    assert view.evaluation_progress.maximum() == points
    assert view.evaluation_progress.value() == points
    assert view.button_evaluate.text() == 'Evaluate'
    assert not view.evaluation_running
    assert not view.evaluation_timer.isActive()
    assert view.nrBrillouinPeaksGroup.isEnabled()
    assert processed_points(session) == points


def test_progress_is_partial_while_run_is_held(
        qtbot, window, view, thread_registry, run_gate, files):
    session = open_measurement(window, files, WATER)
    gate = run_gate(LATE_HOLD)

    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    # Make the timer slot update the progress bar
    view.refresh_ui()

    assert 0 < view.count.value < point_count(session)
    assert view.evaluation_progress.value() == view.count.value
    assert view.evaluation_progress.maximum() == point_count(session)
    assert view.evaluation_running

    gate.release()
    wait_until_idle(qtbot, view, thread_registry)
    assert view.evaluation_progress.value() == point_count(session)


def test_evaluate_without_file_does_nothing(
        qtbot, window, view, thread_registry):
    click_evaluate(qtbot, view)

    assert not view.evaluation_running
    assert view.button_evaluate.text() == 'Evaluate'
    assert thread_registry.threads == []


# ---------------------------------------------------------------------------
# Cancel and restart
# ---------------------------------------------------------------------------

def test_cancel_during_run_stops_run(
        qtbot, window, view, thread_registry, run_gate, files,
        child_processes):
    session = open_measurement(window, files, WATER)
    points = point_count(session)
    gate = run_gate(LATE_HOLD)

    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    assert view.button_evaluate.text() == 'Cancel'

    # The same button cancels
    click_evaluate(qtbot, view)
    assert view.button_evaluate.text() == 'Evaluate'
    assert not view.evaluation_running
    assert not view.evaluation_timer.isActive()
    assert view.nrBrillouinPeaksGroup.isEnabled()

    gate.release()
    wait_until_idle(qtbot, view, thread_registry)

    # The run stopped before it finished
    assert view.evaluation_progress.value() < points
    assert 0 < processed_points(session) < points
    # The last point was never reached
    results = session.evaluation_model().results
    assert np.all(np.isnan(
        results['brillouin_peak_position_f'][points - 1, 0, 0]))
    assert view.button_evaluate.isEnabled()
    assert view.button_evaluate.text() == 'Evaluate'
    wait_for_no_children(qtbot, child_processes)


def test_restart_after_finished_cancel_equals_clean_run(
        qtbot, window, view, thread_registry, run_gate, files, reference):
    session = open_measurement(window, files, WATER)
    gate = run_gate()

    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    click_evaluate(qtbot, view)
    gate.release()
    wait_until_idle(qtbot, view, thread_registry)
    assert processed_points(session) < point_count(session)

    run_evaluation(qtbot, view, thread_registry)

    assert_same_snapshot(snapshot(session), reference[(WATER, '0')])


def test_immediate_restart_after_cancel_equals_clean_run(
        qtbot, window, view, thread_registry, run_gate, files, reference,
        child_processes):
    """
    The user presses Cancel and then Evaluate again at once. The button
    is ready for a new run immediately, while the cancelled run is still
    shutting down. Everything the first run does late must not change the
    results of the second run.
    """
    session = open_measurement(window, files, WATER)
    gate = run_gate()

    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    click_evaluate(qtbot, view)
    assert view.button_evaluate.text() == 'Evaluate'
    # Restart while the cancelled run has not even noticed the cancel
    click_evaluate(qtbot, view)
    assert view.evaluation_running
    assert view.button_evaluate.text() == 'Cancel'
    # Guard: this test is only meaningful while the first run is alive
    assert len(thread_registry.threads) == 2
    assert thread_registry.alive(thread_registry.threads[0])
    gate.release()
    wait_until_idle(qtbot, view, thread_registry)

    assert view.count.value == view.max_count.value == point_count(session)
    assert view.evaluation_progress.value() == point_count(session)
    assert_same_snapshot(snapshot(session), reference[(WATER, '0')])
    wait_for_no_children(qtbot, child_processes)


def test_repeated_cancel_and_restart_cycles(
        qtbot, window, view, thread_registry, run_gate, files, reference,
        child_processes):
    session = open_measurement(window, files, WATER)
    for _ in range(3):
        gate = run_gate()
        click_evaluate(qtbot, view)
        qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
        click_evaluate(qtbot, view)
        gate.release()
        wait_until_idle(qtbot, view, thread_registry)
        assert processed_points(session) < point_count(session)
        wait_for_no_children(qtbot, child_processes)

    run_evaluation(qtbot, view, thread_registry)

    assert_same_snapshot(snapshot(session), reference[(WATER, '0')])
    wait_for_no_children(qtbot, child_processes)


def test_late_results_of_cancelled_run_do_not_reach_new_run(
        qtbot, window, view, thread_registry, run_gate, files, reference,
        child_processes):
    """
    Cancel a run on repetition 0, switch to repetition 1 and evaluate it
    to the end, and only then let the cancelled run go on. Whatever the
    cancelled run still writes must not appear in the results of
    repetition 1.

    TODO(integration): the strongest form of this test needs the parallel
    code. With parallel evaluation, make the cancelled run deliver real
    late results from worker processes after the new run started (hold
    the result delivery, not only the extraction) and also run the new
    run on the same repetition. Check the result arrays of the new run
    against the clean reference exactly. The structure below stays the
    same.
    """
    session = open_measurement(window, files, TWO_REPETITIONS, '0')
    gate = run_gate()

    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    click_evaluate(qtbot, view)

    select_repetition(window, '1')
    # Second run is not held: the gate only holds one call
    run_evaluation_with_old_run_alive(qtbot, view, thread_registry)
    # Guard: the cancelled run is still held when the new run is done
    assert len(thread_registry.running()) == 1
    assert thread_registry.alive(thread_registry.threads[0])
    new_run = snapshot(session)

    gate.release()
    wait_until_idle(qtbot, view, thread_registry)

    assert_same_snapshot(snapshot(session), new_run)
    assert_same_snapshot(snapshot(session),
                         reference[(TWO_REPETITIONS, '1')])
    wait_for_no_children(qtbot, child_processes)


def run_evaluation_with_old_run_alive(qtbot, view, registry):
    click_evaluate(qtbot, view)
    qtbot.waitUntil(
        lambda: not view.evaluation_running
        and len(registry.running()) <= 1,
        timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# Run identity: every run has its own state
# ---------------------------------------------------------------------------

def test_execution_mode_is_what_the_test_asks_for(
        qtbot, window, view, thread_registry, files, reference, execution,
        child_processes):
    """
    Guard for the whole module: parallel runs use worker processes, serial
    runs none, and both give the results of the serial reference.
    """
    session = open_measurement(window, files, WATER)

    run_evaluation(qtbot, view, thread_registry)

    assert len(execution.pools) == (1 if execution.parallel else 0)
    for pool in execution.pools:
        # The pool of the run is over: no worker is left
        assert all(worker.exitcode is not None
                   for worker in pool._first_workers)
    assert_same_snapshot(snapshot(session), reference[(WATER, '0')])
    wait_for_no_children(qtbot, child_processes)


def test_every_run_has_its_own_abort_flag_and_progress(
        qtbot, window, view, thread_registry, run_gate, files):
    open_measurement(window, files, WATER)
    gate = run_gate()
    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    first_abort, first_count = view.evaluation_abort, view.count
    click_evaluate(qtbot, view)
    assert first_abort.value

    click_evaluate(qtbot, view)

    # The new run does not share anything with the cancelled run
    assert view.evaluation_abort is not first_abort
    assert view.count is not first_count
    assert not view.evaluation_abort.value
    # and starting it did not revive the cancelled run
    assert first_abort.value
    gate.release()
    wait_until_idle(qtbot, view, thread_registry)
    assert first_abort.value
    assert view.count.value == view.max_count.value


def test_running_thread_stays_referenced_after_restart(
        qtbot, window, view, thread_registry, run_gate, files, mocker):
    """
    The cancelled run is still running when the user restarts. The view
    must keep its thread alive: destroying a running QThread aborts the
    application. The threads here are only referenced weakly by the test.
    """
    open_measurement(window, files, WATER)
    weak_threads = []

    class WeakSpyThread(QThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            weak_threads.append(weakref.ref(self))

    mocker.patch.object(evaluation_view_module, 'QThread', WeakSpyThread)
    gate = run_gate()
    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    click_evaluate(qtbot, view)
    click_evaluate(qtbot, view)
    assert len(weak_threads) == 2

    gc.collect()
    first = weak_threads[0]()
    assert first is not None
    assert first.isRunning()

    gate.release()
    qtbot.waitUntil(lambda: not first.isRunning(), timeout=TIMEOUT)
    qtbot.waitUntil(lambda: not view.evaluation_running, timeout=TIMEOUT)
    # the view lets go of finished threads
    qtbot.waitUntil(lambda: len(view._runs) <= 1, timeout=TIMEOUT)


def test_finish_of_superseded_run_does_not_change_the_new_run(
        qtbot, window, view, thread_registry, run_gate, files, reference):
    """
    Cancel, restart, and let the cancelled run end while the new run is
    still going. The end of the old run must not touch progress, button,
    timer, or plot state of the new run.
    """
    session = open_measurement(window, files, WATER)
    first_gate = run_gate()
    click_evaluate(qtbot, view)
    qtbot.waitUntil(first_gate.reached.is_set, timeout=TIMEOUT)
    first_run = view._current_run
    click_evaluate(qtbot, view)

    second_gate = run_gate(LATE_HOLD)
    click_evaluate(qtbot, view)
    qtbot.waitUntil(second_gate.reached.is_set, timeout=TIMEOUT)
    second_run = view._current_run
    assert second_run is not first_run
    view.refresh_ui()
    count_before = view.count.value
    progress_before = view.evaluation_progress.value()

    first_gate.release()
    qtbot.waitUntil(lambda: first_run.finished, timeout=TIMEOUT)
    qtbot.wait(100)

    assert first_run.outcome in ('cancelled', 'superseded')
    # The second run is untouched
    assert not second_run.finished
    assert view.evaluation_running
    assert view.button_evaluate.text() == 'Cancel'
    assert view.evaluation_timer.isActive()
    assert not view.nrBrillouinPeaksGroup.isEnabled()
    assert view.count is second_run.count
    assert view.count.value == count_before
    assert view.evaluation_progress.value() == progress_before

    second_gate.release()
    wait_until_idle(qtbot, view, thread_registry)
    assert view.count.value == view.max_count.value == point_count(session)
    assert_same_snapshot(snapshot(session), reference[(WATER, '0')])


def test_end_of_a_run_is_handled_in_the_gui_thread(
        qtbot, window, view, thread_registry, files, mocker):
    """
    The slots that end a run touch widgets and matplotlib. They must run
    in the thread of the application, not in the worker thread.
    """
    open_measurement(window, files, WATER)
    threads = []
    for name in ('on_run_finished', '_forget_finished_runs'):
        original = getattr(evaluation_view_module.EvaluationView, name)

        def spy(self, *args, original=original):
            threads.append(threading.current_thread())
            return original(self, *args)

        mocker.patch.object(
            evaluation_view_module.EvaluationView, name, spy)

    run_evaluation(qtbot, view, thread_registry)

    assert threads
    assert all(t is threading.main_thread() for t in threads)


# ---------------------------------------------------------------------------
# Late results of a cancelled run (parallel evaluation)
# ---------------------------------------------------------------------------

def poison(value):
    """
    Make fit results of a worker recognizable and wrong: every number
    changes, also the NaN of failed fits.
    """
    if isinstance(value, dict):
        return {key: poison(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(poison(item) for item in value)
    if isinstance(value, np.ndarray) and value.dtype.kind == 'f':
        return np.where(np.isnan(value), 1.0e9, value * 3.0 + 1.0e6)
    return value


class LatePool:
    """
    A real pool of worker processes whose results are held back. The
    results of the workers are real, but they reach the run only when
    the test calls ``deliver``, and then with wrong numbers (``poison``).
    """

    def __init__(self, pool):
        self.pool = pool
        self.held = []
        self.terminated = False
        self._lock = threading.Lock()

    def apply_async(self, func, args=(), callback=None,
                    error_callback=None):
        def hold(results):
            with self._lock:
                self.held.append((callback, poison(results)))
        return self.pool.apply_async(
            func, args, callback=hold, error_callback=error_callback)

    def check_workers(self):
        self.pool.check_workers()

    def close(self):
        self.pool.close()

    def terminate(self):
        self.terminated = True
        self.pool.terminate()

    def join(self):
        self.pool.join()

    def deliver(self):
        with self._lock:
            held, self.held = self.held, []
        for callback, results in held:
            callback(results)
        return len(held)


@parallel_only
@pytest.mark.parametrize('flag_stays_set', [True, False],
                         ids=['abort-flag-stays-set', 'abort-flag-reset'])
def test_late_results_of_cancelled_parallel_run_do_not_reach_new_run(
        qtbot, window, view, thread_registry, run_gate, files, reference,
        child_processes, mocker, flag_stays_set):
    """
    The strong form of the late-result test. A parallel run is held in the
    extraction of a point while its workers have fitted point 0 for real.
    The user cancels and evaluates the same repetition again. The new run
    finishes. Only then the old run receives its results (with wrong
    numbers) and is allowed to go on. Nothing of that may appear in the
    results of the new run, and no worker of the old run may survive.

    In the variant "abort-flag-reset" the old run also does not see its
    cancel any more (its flag is reset, as the shared flag of the old
    implementation was by a restart), so only the run identity of bmlab
    stops it.
    """
    session = open_measurement(window, files, WATER)
    pools = []

    def create_pool(workers):
        pool = parallel_fit.create_spawn_pool(workers)
        if not pools:
            # the first run
            pool = LatePool(pool)
        pools.append(pool)
        return pool

    mocker.patch.object(
        EvaluationController, '_fit_execution',
        dataclasses.replace(PARALLEL_EXECUTION, pool_factory=create_pool))
    gate = run_gate()

    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    old_run = view._current_run
    late = pools[0]
    # A real worker has fitted a point of the old run, the result waits
    qtbot.waitUntil(lambda: bool(late.held), timeout=TIMEOUT)

    click_evaluate(qtbot, view)
    click_evaluate(qtbot, view)
    assert view._current_run is not old_run
    if not flag_stays_set:
        old_run.abort.value = False
    # The old run is still held, the new run runs on its own
    wait_for_run(qtbot, view)
    assert len(pools) == 2
    assert thread_registry.alive(thread_registry.threads[0])
    new_run_results = snapshot(session)
    assert_same_snapshot(new_run_results, reference[(WATER, '0')])

    # Late: the old results arrive and the old run goes on
    assert late.deliver() >= 1
    gate.release()
    wait_until_idle(qtbot, view, thread_registry)

    assert_same_snapshot(snapshot(session), new_run_results)
    assert_same_snapshot(snapshot(session), reference[(WATER, '0')])
    assert old_run.outcome in (('cancelled', 'superseded')
                               if flag_stays_set else ('superseded',))
    assert late.terminated
    wait_for_no_children(qtbot, child_processes)


def wait_for_run(qtbot, view):
    """Wait until the current run of the view has ended."""
    run = view._current_run
    qtbot.waitUntil(lambda: run.finished, timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# Failure of a run
# ---------------------------------------------------------------------------

class DyingPool(multiprocessing.pool.ThreadPool):
    """A pool that reports a dead worker, like a spawn pool does."""

    def check_workers(self):
        raise parallel_fit.WorkerDied(
            'A worker process of the evaluation ended unexpectedly '
            '(process 1, exit code 1). The evaluation was stopped. '
            'Start it again.')


@pytest.fixture
def failing_workers(mocker):
    """Every parallel run fails at once, as with a dead worker."""
    mocker.patch.object(
        EvaluationController, '_fit_execution',
        dataclasses.replace(PARALLEL_EXECUTION, pool_factory=DyingPool))


@pytest.fixture
def excepthook(mocker):
    return mocker.patch.object(sys, 'excepthook')


@parallel_only
def test_failed_run_resets_the_button_and_reports_the_error(
        qtbot, window, view, thread_registry, files, reference, execution,
        failing_workers, excepthook, caplog, mocker):
    """
    A run that ends with an error (here a dead worker) must not leave the
    button on Cancel for ever. Like every other unhandled error of the
    application, the error goes to the exception hook (the existing error
    dialog) and to the log. No new dialog or control is involved.
    """
    session = open_measurement(window, files, WATER)

    with caplog.at_level(logging.ERROR):
        run_evaluation(qtbot, view, thread_registry)

    assert view.button_evaluate.text() == 'Evaluate'
    assert not view.evaluation_running
    assert not view.evaluation_timer.isActive()
    assert view.nrBrillouinPeaksGroup.isEnabled()
    assert view.button_evaluate.isEnabled()
    assert view._current_run.outcome == 'failed'
    excepthook.assert_called_once()
    error_type, error, _ = excepthook.call_args.args
    assert error_type is parallel_fit.WorkerDied
    assert 'worker process' in str(error)
    assert any('The evaluation failed' in r.getMessage()
               for r in caplog.records)

    # The user can evaluate again, with healthy workers
    excepthook.reset_mock()
    mocker.patch.object(EvaluationController, '_fit_execution',
                        execution.fit_execution())
    run_evaluation(qtbot, view, thread_registry)
    excepthook.assert_not_called()
    assert_same_snapshot(snapshot(session), reference[(WATER, '0')])


@parallel_only
def test_failed_blocking_run_raises_to_the_caller(
        qtbot, window, view, thread_registry, files, failing_workers,
        excepthook):
    open_measurement(window, files, WATER)

    with pytest.raises(parallel_fit.WorkerDied):
        view.evaluate(blocking=True)

    assert not view.evaluation_running
    assert view.button_evaluate.text() == 'Evaluate'
    excepthook.assert_not_called()
    qtbot.waitUntil(lambda: not thread_registry.running(), timeout=TIMEOUT)


@parallel_only
def test_batch_marks_file_with_failed_evaluation_as_failed(
        qtbot, window, view, thread_registry, files, batch,
        failing_workers, excepthook):
    events = batch
    entry = batch_entry(files, WATER)
    window.batch_files = {'a': entry}

    window.run_batch_evaluation()

    assert entry['status'] == 'failed'
    # A file with a failed evaluation is not saved
    assert 'save' not in [e[0] for e in events]
    excepthook.assert_not_called()
    assert not thread_registry.running()


# ---------------------------------------------------------------------------
# File and repetition switching
# ---------------------------------------------------------------------------

def test_file_switching_gives_results_of_selected_file(
        qtbot, window, view, thread_registry, files, reference):
    session = open_measurement(window, files, WATER)
    run_evaluation(qtbot, view, thread_registry)
    water = snapshot(session)
    assert_same_snapshot(water, reference[(WATER, '0')])

    session = open_measurement(window, files, TWO_REPETITIONS)
    # The previous results are gone, nothing is mixed up
    assert session.evaluation_model().results['time'].size == 0
    run_evaluation(qtbot, view, thread_registry)
    two = snapshot(session)
    assert_same_snapshot(two, reference[(TWO_REPETITIONS, '0')])
    assert two['results']['time'].shape != water['results']['time'].shape

    # and back
    session = open_measurement(window, files, WATER)
    run_evaluation(qtbot, view, thread_registry)
    assert_same_snapshot(snapshot(session), water)


def test_switching_file_after_cancel_gives_clean_results(
        qtbot, window, view, thread_registry, run_gate, files, reference):
    open_measurement(window, files, WATER)
    gate = run_gate()
    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    click_evaluate(qtbot, view)
    gate.release()
    wait_until_idle(qtbot, view, thread_registry)

    session = open_measurement(window, files, TWO_REPETITIONS)
    run_evaluation(qtbot, view, thread_registry)

    assert_same_snapshot(snapshot(session),
                         reference[(TWO_REPETITIONS, '0')])


def test_repetition_switching_gives_results_of_selected_repetition(
        qtbot, window, view, thread_registry, files, reference):
    session = open_measurement(window, files, TWO_REPETITIONS, '0')
    run_evaluation(qtbot, view, thread_registry)
    first = snapshot(session)
    assert_same_snapshot(first, reference[(TWO_REPETITIONS, '0')])

    select_repetition(window, '1')
    # Repetition 1 has not been evaluated yet
    assert session.evaluation_model().results['time'].size == 0
    run_evaluation(qtbot, view, thread_registry)
    second = snapshot(session)
    assert_same_snapshot(second, reference[(TWO_REPETITIONS, '1')])

    # The repetitions differ, so the checks above can tell them apart
    assert not np.array_equal(first['results']['time'],
                              second['results']['time'])
    assert not np.array_equal(first['positions']['x'],
                              second['positions']['x'])

    # Repetition 0 still holds its own results
    select_repetition(window, '0')
    assert_same_snapshot(snapshot(session), first)
    # and evaluating it again gives the same results again
    run_evaluation(qtbot, view, thread_registry)
    assert_same_snapshot(snapshot(session), first)


def test_repetition_switching_after_cancel_gives_clean_results(
        qtbot, window, view, thread_registry, run_gate, files, reference):
    session = open_measurement(window, files, TWO_REPETITIONS, '0')
    gate = run_gate()
    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    click_evaluate(qtbot, view)
    gate.release()
    wait_until_idle(qtbot, view, thread_registry)

    select_repetition(window, '1')
    run_evaluation(qtbot, view, thread_registry)

    assert_same_snapshot(snapshot(session),
                         reference[(TWO_REPETITIONS, '1')])


def test_plot_follows_selected_repetition(
        qtbot, window, view, thread_registry, files, reference):
    open_measurement(window, files, TWO_REPETITIONS, '0')
    run_evaluation(qtbot, view, thread_registry)
    data_first = np.array(view.image_map.get_array())

    select_repetition(window, '1')
    run_evaluation(qtbot, view, thread_registry)
    data_second = np.array(view.image_map.get_array())

    expected = reference[(TWO_REPETITIONS, '1')]['results']
    assert not np.array_equal(data_first, data_second, equal_nan=True)
    np.testing.assert_array_equal(
        data_second,
        np.rot90(np.nanmean(expected['brillouin_shift_f'],
                            axis=tuple(range(3, 6)))[:, :, 0] * 1e-9))


# ---------------------------------------------------------------------------
# Plot refreshes and plot positions
# ---------------------------------------------------------------------------

def centered(values):
    return values - np.nanmean(values)


def test_repeated_plot_refreshes_keep_positions(
        qtbot, window, view, thread_registry, run_gate, files, reference):
    session = open_measurement(window, files, WATER)
    # Positions as read from the file, before anything was plotted
    original = reference[(WATER, '0')]['positions']
    expected_x = centered(original['x'])[:, 0, 0]
    gate = run_gate()

    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    # Refresh the plot over and over while the run is held
    for _ in range(5):
        view.refresh_plot()
        np.testing.assert_array_equal(view.image_map[0].get_xdata(),
                                      expected_x)
    gate.release()
    wait_until_idle(qtbot, view, thread_registry)
    for _ in range(5):
        view.refresh_plot()
        np.testing.assert_array_equal(view.image_map[0].get_xdata(),
                                      expected_x)

    # Refreshes must not leak into the stored positions either
    for key, values in session.get_payload_positions().items():
        np.testing.assert_array_equal(values, original[key])


def test_repeated_plot_refreshes_keep_image_extent(
        qtbot, window, view, thread_registry, files, reference):
    session = open_measurement(window, files, TWO_REPETITIONS, '0')
    # Positions as read from the file, before anything was plotted
    original = reference[(TWO_REPETITIONS, '0')]['positions']
    run_evaluation(qtbot, view, thread_registry)
    extent = tuple(view.image_map.get_extent())
    expected = (np.nanmin(centered(original['x'])),
                np.nanmax(centered(original['x'])),
                np.nanmin(centered(original['y'])),
                np.nanmax(centered(original['y'])))
    assert extent == pytest.approx(expected)

    for _ in range(5):
        view.refresh_plot()
        window.update_ui()
        assert tuple(view.image_map.get_extent()) == extent

    for key, values in session.get_payload_positions().items():
        np.testing.assert_array_equal(values, original[key])


def test_plot_positions_after_cancel_and_restart(
        qtbot, window, view, thread_registry, run_gate, files, reference):
    open_measurement(window, files, WATER)
    expected_x = centered(reference[(WATER, '0')]['positions']['x'])[:, 0, 0]
    gate = run_gate()

    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    click_evaluate(qtbot, view)
    gate.release()
    wait_until_idle(qtbot, view, thread_registry)
    run_evaluation(qtbot, view, thread_registry)

    np.testing.assert_array_equal(view.image_map[0].get_xdata(), expected_x)


# ---------------------------------------------------------------------------
# Batch evaluation with the blocking call
# ---------------------------------------------------------------------------

@pytest.fixture
def batch(window, mocker):
    """The batch dialog, without showing it, and a spy on save and close."""
    dialog = QtWidgets.QDialog(window)
    ref = main_module.resources.files('bmicro.gui') / 'batch_evaluation.ui'
    with main_module.resources.as_file(ref) as ui_file:
        uic.loadUi(ui_file, dialog)
    window.batch_dialog = dialog
    window.batch_config['evaluation']['evaluate'] = True
    window.batch_evaluation_running = True

    events = []
    original_save = BMicro.save_session
    original_close = BMicro.close_file

    def save_session():
        session = Session.get_instance()
        events.append(('save', processed_points(session),
                       point_count(session)))
        original_save()

    def close_file(self):
        events.append(('close',))
        original_close(self)

    mocker.patch.object(BMicro, 'save_session', staticmethod(save_session))
    mocker.patch.object(BMicro, 'close_file', close_file)
    return events


def batch_entry(files, name):
    return {'path': files / name, 'status': 'pending'}


def test_blocking_batch_evaluation_waits_for_calculation(
        qtbot, window, view, thread_registry, run_gate, files, batch,
        reference, child_processes):
    events = batch
    gate = run_gate()
    entry = batch_entry(files, WATER)
    checks = []

    def during_calculation():
        # Called from the event loop that evaluate(blocking=True) runs
        checks.append({
            'held': gate.reached.is_set(),
            'running': view.evaluation_running,
            'status': entry['status'],
            'events': list(events),
        })
        # Let a moment pass while the calculation is still held
        QtCore.QTimer.singleShot(300, check_still_waiting)

    def check_still_waiting():
        checks.append({
            'running': view.evaluation_running,
            'status': entry['status'],
            'events': list(events),
        })
        gate.release()

    # The gate is reached from the worker thread, the timer polls for it
    poll = QtCore.QTimer()
    poll.setInterval(10)

    def poll_gate():
        if gate.reached.is_set():
            poll.stop()
            during_calculation()

    poll.timeout.connect(poll_gate)
    poll.start()
    try:
        window.evaluate_batch_file(entry)
    finally:
        poll.stop()
        gate.release()

    assert len(checks) == 2
    for check in checks:
        # No file was advanced, saved, or closed during the calculation.
        # The only event so far is the close of the previously open file
        # while the batch opened this one.
        assert check['running']
        assert check['status'] == 'in-process'
        assert check['events'] == [('close',)]
    # Afterwards: saved once with complete results, then closed
    assert [e[0] for e in events] == ['close', 'save', 'close']
    _, processed, points = events[1]
    assert processed == points
    assert entry['status'] == 'success'
    assert not view.evaluation_running
    assert not thread_registry.running()

    # The saved session holds the complete, correct results
    session = Session.get_instance()
    session.clear()
    session.set_file(files / WATER)
    assert_same_snapshot(snapshot(session), reference[(WATER, '0')])
    wait_for_no_children(qtbot, child_processes)


def test_blocking_batch_evaluation_does_not_start_next_file_early(
        qtbot, window, view, thread_registry, run_gate, files, batch,
        child_processes):
    events = batch
    gate = run_gate()
    window.batch_files = {
        'a': batch_entry(files, WATER),
        'b': batch_entry(files, TWO_REPETITIONS),
    }
    opened = []
    original_open = BMicro.open_file

    def open_file(self, file_name=None):
        opened.append((pathlib.Path(file_name).name,
                       gate.reached.is_set(), gate.released.is_set()))
        return original_open(self, file_name)

    mocked = pytest.MonkeyPatch()
    mocked.setattr(BMicro, 'open_file', open_file)
    poll = QtCore.QTimer()
    poll.setInterval(10)

    def poll_gate():
        if gate.reached.is_set():
            poll.stop()
            QtCore.QTimer.singleShot(300, gate.release)

    poll.timeout.connect(poll_gate)
    poll.start()
    try:
        window.run_batch_evaluation()
    finally:
        poll.stop()
        gate.release()
        mocked.undo()

    # The second file was opened only after the first one was held and
    # released, and saved before that
    assert [name for name, _, _ in opened] == [WATER, TWO_REPETITIONS]
    assert opened[1][1] and opened[1][2]
    # Each file: close of the previous file on open, save, close
    assert [e[0] for e in events] == [
        'close', 'save', 'close', 'close', 'save', 'close']
    assert [f['status'] for f in window.batch_files.values()] \
        == ['success', 'success']
    wait_for_no_children(qtbot, child_processes)


# ---------------------------------------------------------------------------
# Application exit
# ---------------------------------------------------------------------------

def test_observer_detects_leftover_processes(child_processes):
    """
    The checks for surviving worker processes below are only worth
    something if the observer sees processes of every kind.
    """
    plain = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)'])
    spawned = multiprocessing.get_context('spawn').Process(
        target=time.sleep, args=(60,))
    spawned.start()
    try:
        assert {plain.pid, spawned.pid} <= set(child_processes.stray())
    finally:
        plain.kill()
        plain.wait()
        spawned.kill()
        spawned.join()
    assert child_processes.stray() == []


def quit_application():
    """
    What the application does when it leaves its event loop. The tests do
    not run the event loop of the application, so they emit the signal.
    """
    QtCore.QCoreApplication.instance().aboutToQuit.emit()


def test_quit_after_evaluation_leaves_nothing_running(
        qtbot, window, view, thread_registry, files, child_processes):
    open_measurement(window, files, WATER)
    run_evaluation(qtbot, view, thread_registry)

    quit_application()
    window.close()

    assert not view.evaluation_running
    assert not view.evaluation_timer.isActive()
    assert thread_registry.running() == []
    wait_for_no_children(qtbot, child_processes)


def test_quit_during_evaluation_cancels_and_leaves_nothing_running(
        qtbot, window, view, thread_registry, run_gate, files, execution,
        child_processes):
    """
    The application quits while the calculation runs. The run is cancelled
    and ended by the application itself: when quitting returns, no thread
    is running. A worker process of a parallel run is gone shortly after.
    The extraction held by the gate ends 0.3 s after the quit started.
    """
    session = open_measurement(window, files, WATER)
    gate = run_gate(LATE_HOLD)
    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    if execution.parallel:
        assert child_processes.stray()

    threading.Timer(0.3, gate.release).start()
    quit_application()

    assert thread_registry.running() == []
    assert not view.evaluation_timer.isActive()
    assert processed_points(session) < point_count(session)
    wait_for_no_children(qtbot, child_processes)


def test_quit_while_cancelled_run_is_shutting_down(
        qtbot, window, view, thread_registry, run_gate, files,
        child_processes):
    open_measurement(window, files, WATER)
    gate = run_gate()
    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)
    click_evaluate(qtbot, view)

    threading.Timer(0.3, gate.release).start()
    quit_application()

    assert thread_registry.running() == []
    assert not view.evaluation_timer.isActive()
    wait_for_no_children(qtbot, child_processes)


def test_quit_does_not_wait_for_ever(
        qtbot, window, view, thread_registry, run_gate, files):
    """
    A run that does not end must not keep the application from quitting.
    """
    open_measurement(window, files, WATER)
    gate = run_gate()
    click_evaluate(qtbot, view)
    qtbot.waitUntil(gate.reached.is_set, timeout=TIMEOUT)

    view.stop_evaluation(timeout_ms=200)

    # The held run is still there; the quit returned anyway
    assert len(thread_registry.running()) == 1
    gate.release()
    wait_until_idle(qtbot, view, thread_registry)


def run_application_and_watch_children(measurement, mode, execution,
                                       output_dir):
    """
    Run the application in its own process (see ``exit_probe.py``), close
    its window, and watch all processes it starts. Returns the probe report,
    the descendant processes that are still alive after the application
    exited, the exit code, and the standard error output. The output goes
    to files, not to pipes: a surviving worker would hold a pipe open and
    hide itself from the test.
    """
    probe = pathlib.Path(__file__).parent / 'exit_probe.py'
    stdout_path = output_dir / 'probe.out'
    stderr_path = output_dir / 'probe.err'
    with open(stdout_path, 'w') as out, open(stderr_path, 'w') as err:
        process = subprocess.Popen(
            [sys.executable, '-W', 'ignore', str(probe), str(measurement),
             mode, execution],
            stdout=out, stderr=err, stdin=subprocess.DEVNULL,
            env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'})
    parent = psutil.Process(process.pid)
    seen = {}
    deadline = time.monotonic() + 60
    try:
        while process.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError('The application did not exit.')
            try:
                for child in parent.children(recursive=True):
                    if not is_infrastructure(child):
                        seen[(child.pid, child.create_time())] = child
            except psutil.Error:
                pass
            time.sleep(0.02)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    # Give workers a moment to notice that the application is gone
    deadline = time.monotonic() + 5
    survivors = []
    while time.monotonic() < deadline:
        survivors = [c.pid for c in seen.values()
                     if c.is_running()
                     and c.status() != psutil.STATUS_ZOMBIE]
        if not survivors:
            break
        time.sleep(0.05)
    stdout = stdout_path.read_text()
    stderr = stderr_path.read_text()
    lines = [line for line in stdout.splitlines() if line.startswith('{')]
    assert lines, f'No report from the application.\n{stdout}\n{stderr}'
    return json.loads(lines[-1]), survivors, process.returncode, stderr


@pytest.mark.parametrize('execution_mode', ['serial', 'parallel'])
@pytest.mark.parametrize('mode', ['early', 'during', 'after'])
def test_application_exit_leaves_no_worker_processes(
        mode, execution_mode, files, tmp_path):
    """
    The real application exit: start the application, evaluate, close the
    window (during or after the calculation), and check that the process
    ends cleanly and none of its worker processes survive.
    """
    report, survivors, returncode, stderr = run_application_and_watch_children(
        files / WATER, mode, execution_mode, tmp_path)

    # The scenario happened as intended
    assert report['running_at_close'] is (mode != 'after')
    if mode == 'during' and execution_mode == 'parallel':
        # The workers were busy when the window was closed
        assert report['workers_at_close'] >= 2
    assert returncode == 0
    assert report['exit_code'] == 0
    # Qt complains on stderr if a running thread is destroyed
    assert 'while thread is still running' not in stderr
    assert survivors == []
