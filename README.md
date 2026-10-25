# DBPJT (Project A)

Forecast-driven partition placement under flash crowds. Local execution only: no git, no Docker.
Specification: Project_A_Execution_Guide_LLM.docx (single source of truth).
Project root is this directory; "projectA/" in the guide means this directory.

Setup (Windows PowerShell, from this directory):

    py -3.14 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Run: python tasks.py <test|data|forecast|tune|grid|analyze|freeze|report|all|smoke>

Status: Phase 0 complete. Later phases add configs, data, source, tests and results.
Clean-room rerun outcome is recorded here in Phase 16.
