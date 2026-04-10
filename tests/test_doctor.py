"""Tests for the environment doctor."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_doctor_json_output_contains_expected_keys():
    output = subprocess.check_output([sys.executable, "scripts/doctor.py", "--json"], cwd=ROOT, text=True)
    payload = json.loads(output)

    assert "project_root" in payload
    assert "commands" in payload
    assert "paths" in payload
    assert "optional_modules" in payload
    assert isinstance(payload["notes"], list)

