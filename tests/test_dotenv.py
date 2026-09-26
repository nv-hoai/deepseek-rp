"""Unit tests for .env configuration support."""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _example_keys() -> set[str]:
    keys = set()
    for line in (REPO_ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0].strip())
    return keys


def test_env_example_covers_all_settings():
    """Every env var the server reads must be documented in .env.example."""
    used = set()
    for source in ["dsk/openai_server.py", "dsk/token_store.py",
                   "dsk/auth.py", "dsk/config.py"]:
        text = (REPO_ROOT / source).read_text()
        used.update(re.findall(r'os\.getenv\("([A-Z_]+)"', text))
        used.update(re.findall(r"os\.getenv\('([A-Z_]+)'", text))
    used.discard("TERM")
    assert used, "no env vars found; regex out of date?"
    assert used <= _example_keys(), f"undocumented: {used - _example_keys()}"


def test_dotenv_loaded_on_import(tmp_path, monkeypatch):
    """Simulates `uvicorn dsk.openai_server:app`: plain import reads .env."""
    (tmp_path / ".env").write_text("DSK_DOTENV_PROBE=hello-123\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DSK_DOTENV_PROBE", raising=False)
    code = ("import os, dsk.openai_server; "
            "print(os.getenv('DSK_DOTENV_PROBE'))")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=tmp_path, timeout=120, env=env)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "hello-123"


def test_real_env_wins_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DSK_DOTENV_PROBE=from-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DSK_DOTENV_PROBE", "from-env")
    code = ("import os, dsk.openai_server; "
            "print(os.getenv('DSK_DOTENV_PROBE'))")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=tmp_path, timeout=120, env=env)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "from-env"
