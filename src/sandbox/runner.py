"""
Sandbox Runner module for executing code inside isolated Docker containers.
"""

from dataclasses import dataclass
import os
import subprocess
import tempfile
import uuid
from typing import Optional

from src.sandbox.compile_step import prepare_sandbox_execution


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    compile_error: bool = False
    compile_stderr: str = ""


def _run_subprocess_with_timeout(cmd_list: list, timeout_sec: float, container_name: Optional[str] = None) -> tuple[str, str, int, bool]:
    """Helper to run a subprocess list with hard wall-clock timeout and immediate container cleanup."""
    proc = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return stdout, stderr, proc.returncode, False
    except subprocess.TimeoutExpired:
        if container_name:
            subprocess.run(["docker", "kill", container_name], capture_output=True)
        proc.kill()
        proc.communicate()
        return "", "Execution timed out", -1, True


def run_in_sandbox(
    code: str,
    lang: str,
    stdin_input: Optional[str] = None,
    timeout_sec: float = 5.0,
    memory_limit: str = "256m",
    image_name: str = "rosetta-sandbox:latest"
) -> ExecutionResult:
    """
    Compiles and executes source code inside a Docker container with strict sandbox security limits:
    - No network access (`--network none`)
    - Memory cap (`--memory 256m`)
    - CPU limit (`--cpus=1.0`)
    - Hard wall-clock timeout enforced via process subprocess timeout
    """
    prep = prepare_sandbox_execution(code, lang)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content in prep["files"].items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
                
        if stdin_input is not None:
            stdin_path = os.path.join(temp_dir, "stdin.txt")
            with open(stdin_path, "w", encoding="utf-8") as f:
                f.write(stdin_input)

        temp_dir_abs = os.path.abspath(temp_dir)
        container_name = f"sandbox_run_{uuid.uuid4().hex[:8]}"

        # Step 1: Compilation stage (for Java & C++)
        if prep["is_compiled"] and prep["compile_cmd"]:
            compile_container_name = f"sandbox_compile_{uuid.uuid4().hex[:8]}"
            compile_cmd = [
                "docker", "run", "--name", compile_container_name, "--rm",
                "-v", f"{temp_dir_abs}:/sandbox",
                "--network", "none",
                "--memory", memory_limit,
                image_name,
                "bash", "-c", prep["compile_cmd"]
            ]
            
            c_stdout, c_stderr, c_code, c_timed_out = _run_subprocess_with_timeout(
                compile_cmd, timeout_sec, compile_container_name
            )
            
            if c_timed_out or c_code != 0:
                return ExecutionResult(
                    stdout="",
                    stderr="",
                    exit_code=c_code,
                    timed_out=c_timed_out,
                    compile_error=True,
                    compile_stderr=c_stderr or c_stdout
                )

        # Step 2: Runtime execution stage
        bash_exec = prep["run_cmd"]
        if stdin_input is not None:
            bash_exec = f"{bash_exec} < stdin.txt"

        run_cmd = [
            "docker", "run", "--name", container_name, "--rm",
            "--network", "none",
            "--memory", memory_limit,
            "--cpus=1.0",
            "-v", f"{temp_dir_abs}:/sandbox",
            image_name,
            "bash", "-c", bash_exec
        ]

        r_stdout, r_stderr, r_code, r_timed_out = _run_subprocess_with_timeout(
            run_cmd, timeout_sec, container_name
        )
        
        return ExecutionResult(
            stdout=r_stdout,
            stderr=r_stderr,
            exit_code=r_code,
            timed_out=r_timed_out,
            compile_error=False,
            compile_stderr=""
        )
