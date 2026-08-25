"""
Safety unit tests for Sandbox security enforcement:
1. Infinite loop timeout kill
2. Network access blocking (--network none)
3. Filesystem write containment outside working directory
"""

import pytest
from src.sandbox.runner import run_in_sandbox, ExecutionResult


def test_infinite_loop_timeout():
    """Assert that infinite loop is killed by wall-clock timeout and timed_out=True."""
    code = """import time
while True:
    time.sleep(0.1)
"""
    res: ExecutionResult = run_in_sandbox(code, lang="python", timeout_sec=2.0)
    
    print("\n--- SAFETY TEST: Infinite Loop Timeout ---")
    print(f"Timed Out: {res.timed_out}")
    print(f"Exit Code: {res.exit_code}")
    print(f"Stderr: {res.stderr.strip()}")
    
    assert res.timed_out is True, "Expected timed_out=True for infinite loop"
    assert res.exit_code == -1, f"Expected exit_code=-1, got {res.exit_code}"


def test_network_access_blocked():
    """Assert that network requests fail due to --network none."""
    code = """import urllib.request
try:
    urllib.request.urlopen("http://8.8.8.8", timeout=2)
    print("SUCCESS_NETWORK")
except Exception as e:
    print(f"NETWORK_FAILED: {e}")
    raise
"""
    res: ExecutionResult = run_in_sandbox(code, lang="python", timeout_sec=5.0)
    
    print("\n--- SAFETY TEST: Network Call Blocking ---")
    print(f"Exit Code: {res.exit_code}")
    print(f"Stdout: {res.stdout.strip()}")
    print(f"Stderr: {res.stderr.strip()}")
    
    assert res.exit_code != 0 or "NETWORK_FAILED" in res.stdout, "Network call should have been blocked"
    assert "SUCCESS_NETWORK" not in res.stdout, "Network call unexpectedly succeeded under --network none"


def test_filesystem_write_containment():
    """Assert that writes outside working directory (/etc/hack.txt) are blocked by permissions/sandbox."""
    code = """try:
    with open('/etc/hack.txt', 'w') as f:
        f.write('hacked')
    print('WRITE_SUCCESS')
except Exception as e:
    print(f'WRITE_BLOCKED: {e}')
    raise
"""
    res: ExecutionResult = run_in_sandbox(code, lang="python", timeout_sec=5.0)
    
    print("\n--- SAFETY TEST: Filesystem Write Containment ---")
    print(f"Exit Code: {res.exit_code}")
    print(f"Stdout: {res.stdout.strip()}")
    print(f"Stderr: {res.stderr.strip()}")
    
    assert res.exit_code != 0 or "WRITE_BLOCKED" in res.stdout, "Write outside working dir should be blocked"
    assert "WRITE_SUCCESS" not in res.stdout, "Write to /etc/ succeeded unexpectedly"
