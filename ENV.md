# ENV.md

- Python: Python 3.14.3
- OS: Microsoft Windows 11 Home Single Language 10.0.26200 (64-bit)
- CPU: AMD Ryzen 7 5700U with Radeon Graphics         , 8 cores, 16 logical processors
- RAM: 13.8 GB
- Free disk on C: at Phase 0: 251.9 GB
- Project root: C:\dev\DBPJT
- Packages: pinned in requirements.txt (pip freeze)
- Shell: Windows PowerShell; tasks.py replaces the Makefile

User-declared (provisional): compute_budget_core_hours 288 (user said "standard"; revisited at the Phase 7 dry-run).
Old repository: not yet provided; its audit is separate from this build.

## Frozen environment (Phase 13, written by scripts/07_freeze.py at 2026-10-25T06:42:36+00:00)

- Python (main environment): 3.14.3
- Platform: Windows-11-10.0.26200-SP0
- Logical processors seen by Python: 16

### pip freeze, main environment (.venv)

```
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.4.0
cycler==0.12.1
fonttools==4.67.0
formulaic==1.2.2
iniconfig==2.3.1
interface_meta==2.0.1
joblib==1.6.0
kiwisolver==1.5.1
lightgbm==4.7.0
matplotlib==3.11.2
narwhals==2.27.1
numpy==2.5.4
packaging==26.3
pandas==3.1.0
patsy==1.0.3
pillow==13.0.0
pluggy==1.7.0
pyarrow==26.0.0
Pygments==2.21.0
pyparsing==3.3.3
pytest==9.1.1
python-dateutil==3.0.0
PyYAML==6.0.3
scikit-learn==1.9.1
scipy==1.18.1
six==1.17.0
statsmodels==0.15.0
threadpoolctl==3.7.0
tqdm==4.70.1
typing_extensions==4.16.0
tzdata==2026.5
wrapt==2.5.2rc1
zstandard==0.25.0
```

### pip freeze, Chronos environment (.venv_chronos)

```
accelerate==1.15.0
annotated-doc==0.0.5
anyio==4.15.1
chronos-forecasting==2.3.2
click==8.5.0
colorama==0.4.6
einops==0.8.2
filelock==4.2.4
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.7.0
httpcore2==2.16.0
httpx2==2.16.0
huggingface_hub==2.3.0
idna==3.20
Jinja2==3.1.6
markdown-it-py==4.2.0
MarkupSafe==3.0.4
mdurl==0.1.2
mpmath==1.3.0
networkx==3.7
numpy==2.5.4
packaging==26.3
pandas==3.1.0
psutil==7.2.2
Pygments==2.21.0
python-dateutil==3.0.0
PyYAML==6.0.3
regex==2026.9.29
rich==15.0.0
safetensors==0.8.0
setuptools==84.2.1
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
tokenizers==0.23.3
torch==2.14.1
tqdm==4.70.1
transformers==5.20.2
truststore==0.10.5
typer==0.27.3
typing_extensions==4.16.0
tzdata==2026.5
```
