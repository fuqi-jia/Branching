from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("z3")


def test_demo_runs_and_reports_optimum():
    out = subprocess.run(
        [sys.executable, "-m", "examples.z3_demo"],
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert "optimum" in out.stdout.lower()


def test_rl_lra_smoke_reports_bool_accuracy():
    out = subprocess.run(
        [sys.executable, "-m", "examples.rl_LRA",
         "--train", "6", "--test", "6", "--min-vars", "10", "--max-vars", "10",
         "--iters", "1", "--epochs", "3", "--max-steps", "40"],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr
    assert "bool 分支准确率" in out.stdout or "bool-head" in out.stdout
