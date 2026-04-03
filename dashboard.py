"""Standard Streamlit entrypoint for the demo app.

This keeps the repo aligned with the other local projects that use
`dashboard.py` as the primary UI launch file while preserving the existing
`app.py` module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
RELAUNCH_FLAG = 'BASKETBALL_DASHBOARD_RELAUNCHED'

if EXPECTED_PYTHON.exists():
    current_python = Path(sys.executable).resolve()
    expected_python = EXPECTED_PYTHON.resolve()
    if current_python != expected_python:
        if os.environ.get(RELAUNCH_FLAG) != '1':
            os.environ[RELAUNCH_FLAG] = '1'
            os.execv(
                str(expected_python),
                [
                    str(expected_python),
                    '-m',
                    'streamlit',
                    'run',
                    str(ROOT / 'dashboard.py'),
                    *sys.argv[1:],
                ],
            )
        raise RuntimeError(
            'The app tried to relaunch itself with the project virtualenv Python, but that handoff did not complete.\n\n'
            f'Current interpreter: {current_python}\n'
            f'Expected interpreter: {expected_python}\n\n'
            'Run this command directly:\n'
            r'  .\.venv\Scripts\python.exe -m streamlit run dashboard.py'
        )

from app import *  # noqa: F401,F403
