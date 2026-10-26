"""
Helper for the application exit tests in ``test_evaluation_lifecycle.py``.

Runs BMicro offscreen in its own process, opens a measurement, starts
Evaluate, and closes the main window, like a user quitting the application.
It prints one line of JSON to the standard output and exits with the exit
code of the Qt event loop.

Usage: ``python exit_probe.py MEASUREMENT.h5 MODE EXECUTION``

MODE is ``early``, ``during`` or ``after``, EXECUTION is ``serial`` or
``parallel``.

``early``:  close the window 50 ms after Evaluate was pressed. The workers
            of a parallel run are still starting.
``during``: close the window while the calculation is still running (after
            the first points are done, so the workers of a parallel run
            are busy).
``after``:  wait until the calculation has finished, then close the window.

``serial`` and ``parallel`` force the execution mode through the internal
switch ``EvaluationController._fit_execution``, as the other tests do.
"""
import json
import multiprocessing
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt6 import QtCore, QtWidgets  # noqa: E402

from bmlab import parallel_fit  # noqa: E402
from bmlab.controllers import EvaluationController  # noqa: E402


def slow_down_extraction(delay):
    original = EvaluationController.extract_spectra

    def slow(self, key):
        time.sleep(delay)
        return original(self, key)

    EvaluationController.extract_spectra = slow


def main(path, mode, execution):
    if execution == 'parallel':
        EvaluationController._fit_execution = parallel_fit.FitExecution(
            parallel=True, workers=2, batch_size=1, max_outstanding_jobs=2)
    else:
        EvaluationController._fit_execution = parallel_fit.serial_execution()
    app = QtWidgets.QApplication([])
    from bmicro.gui.main import BMicro

    window = BMicro()
    window.show()
    window.open_file(path)
    view = window.widget_evaluation_view
    report = {}

    def close_window():
        report['running_at_close'] = bool(view.evaluation_running)
        report['workers_at_close'] = len(multiprocessing.active_children())
        window.close()

    if mode == 'early':
        slow_down_extraction(0.3)
        view.evaluate()
        QtCore.QTimer.singleShot(50, close_window)
    elif mode == 'during':
        # About three seconds for the whole run, plus the start of the
        # workers. Closed when the first points are done.
        slow_down_extraction(0.3)
        view.evaluate()
        poll = QtCore.QTimer()
        poll.setInterval(20)

        def close_when_started():
            if view.count.value >= 2:
                poll.stop()
                close_window()

        poll.timeout.connect(close_when_started)
        poll.start()
    else:
        view.evaluate(blocking=True)
        QtCore.QTimer.singleShot(0, close_window)

    code = app.exec()
    report['exit_code'] = code
    print(json.dumps(report), flush=True)
    sys.exit(code)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
