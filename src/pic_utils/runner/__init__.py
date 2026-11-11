"""Process-isolated batch runner (architecture §8).

The runner owns the worker protocol, shared-memory inputs, the supervised
process pool, the memory guard, and the run service. Only ``service.py`` may
import Qt (the ``QObject`` facade); worker and pool code must stay Qt-free.
Import this package's submodules directly; ``__init__`` stays free of
feature code so importing :mod:`pic_utils.runner` never pulls in Qt.
"""
