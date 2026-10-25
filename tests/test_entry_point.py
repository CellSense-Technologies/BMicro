"""
The entry point of the application and multiprocessing.

Evaluation starts worker processes with the ``spawn`` method (bmlab).
A spawned process imports the main module of its parent again, and a
frozen application (PyInstaller) starts the executable again. The entry
module must therefore be safe to import without starting the application,
and must call ``freeze_support`` before it does anything else.
"""
import ast
import pathlib
import runpy

import bmicro.__main__ as entry_module

ENTRY_FILE = pathlib.Path(entry_module.__file__)


def test_entry_module_does_not_start_the_application_when_imported():
    """What a spawned worker does with the main module of its parent."""
    namespace = runpy.run_module('bmicro.__main__', run_name='__mp_main__')

    assert callable(namespace['main'])


def called_names(statements):
    names = []
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                function = node.func
                names.append(getattr(function, 'id',
                                     getattr(function, 'attr', None)))
    return names


def test_entry_point_calls_freeze_support_before_the_application():
    tree = ast.parse(ENTRY_FILE.read_text())
    guards = [node for node in tree.body
              if isinstance(node, ast.If)
              and isinstance(node.test, ast.Compare)
              and getattr(node.test.left, 'id', None) == '__name__']
    assert len(guards) == 1

    names = called_names(guards[0].body)

    assert 'freeze_support' in names
    assert 'main' in names
    assert names.index('freeze_support') < names.index('main')
    # Nothing but the guard starts the application at import time
    assert 'main' not in called_names(
        [node for node in tree.body if node is not guards[0]])
