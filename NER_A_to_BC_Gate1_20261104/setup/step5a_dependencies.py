"""Colab cell: guard Python compatibility, show installation progress, retain pins."""
import importlib
import importlib.metadata as metadata
import os
import signal
import subprocess
import sys

REQUIRED = {'datasets':'4.0.0','numpy':'2.0.2','pandas':'2.2.2','seqeval':'1.2.2',
            'scikit-learn':'1.6.1','huggingface-hub':'0.36.2'}
print('Python:', sys.version.split()[0], '| executable:', sys.executable, flush=True)
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise RuntimeError(
        'STOP BEFORE INSTALL: this project pins numpy==2.0.2 and pandas==2.2.2. '
        'Use a Python 3.12 runtime (for example the runtime used successfully for Step 4). '
        'Restarting this Python 3.13+ session will not change its Python version. '
        'Connect to a compatible runtime, then run 5a.1 again. No package installation was started.')

missing = []
for package, version in REQUIRED.items():
    try:
        current = metadata.version(package)
    except metadata.PackageNotFoundError:
        current = None
    print(f'{package}: installed={current or "missing"}; required={version}', flush=True)
    if current != version:
        missing.append(package)

if missing:
    print('Installing fixed requirements; mismatched packages:', ', '.join(missing), flush=True)
    print('Live pip output follows. After successful installation, restart Python and rerun 5a.1.', flush=True)
    command = [sys.executable, '-u', '-m', 'pip', 'install', '--use-pep517',
               '--only-binary=numpy,pandas,scipy,scikit-learn,pyarrow',
               '--progress-bar', 'off', '--disable-pip-version-check', '--no-input',
               '--timeout', '30', '--retries', '2',
               *[f'{p}=={v}' for p,v in REQUIRED.items()]]
    # Compiled scientific packages require wheels: fail clearly instead of an
    # unexpected source build. seqeval's small source distribution is allowed.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=(os.name == 'posix'))
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        return_code = process.wait()
    except KeyboardInterrupt:
        # Stop only this installer and its child build processes, not other jobs.
        if process.poll() is None:
            if os.name == 'posix':
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                else:
                    process.kill()
                process.wait()
        print('\nInstallation interrupted. Stop here; send the last pip output before continuing.', flush=True)
        raise
    finally:
        process.stdout.close()
    if return_code:
        raise RuntimeError(f'pip failed (exit {return_code}). Send the last pip lines; do not run 5a.2 yet.')
    raise RuntimeError('Installation finished successfully. Restart the Python session, then rerun 5a.1.')

for package, version in REQUIRED.items():
    module = importlib.import_module({'scikit-learn':'sklearn','huggingface-hub':'huggingface_hub'}.get(package,package))
    imported = getattr(module,'__version__',None)
    if imported is not None and imported != version:
        raise RuntimeError('Stale imported package: restart the Python session, then rerun 5a.1.')
import numpy as np
import pandas as pd
print('Dependencies ready:', REQUIRED, flush=True)
