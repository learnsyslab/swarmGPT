import os
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_frontend_install_runs_during_pixi_activation():
    pyproject = _pyproject()

    activation_scripts = pyproject["tool"]["pixi"]["activation"]["scripts"]

    assert "tools/frontend_install.sh" in activation_scripts


def test_api_task_builds_frontend_before_launch():
    pyproject = _pyproject()

    api_task = pyproject["tool"]["pixi"]["tasks"]["api"]
    web_build = pyproject["tool"]["pixi"]["tasks"]["web-build"]

    assert "web-build" in api_task["depends-on"]
    assert web_build["cmd"] == "npm --prefix web run build"


def test_frontend_install_skips_cleanly_when_web_package_is_missing(tmp_path: Path):
    npm_log = tmp_path / "npm.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    npm = bin_dir / "npm"
    npm.write_text(f"#!/usr/bin/env bash\necho \"$@\" >> {npm_log}\n")
    npm.chmod(0o755)
    env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", str(ROOT / "tools/frontend_install.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "web/package.json not found" in result.stdout
    assert not npm_log.exists()
