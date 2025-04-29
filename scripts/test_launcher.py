"""
Test script to verify the unified launcher system.

This script tests all three launcher modes to ensure they work correctly.
"""

import os
import sys
import subprocess
import time


def test_launcher_mode(mode):
    """Test a specific launcher mode."""
    print(f"\nTesting {mode} mode...")
    env = os.environ.copy()
    env["AI_PLATFORM_LAUNCHER_MODE"] = mode

    # Start the game process
    process = subprocess.Popen(
        [sys.executable, "-m", "ai_platform_trainer.main"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    # Let it run for a short time
    time.sleep(2)

    # Terminate the process
    process.terminate()

    # Get output
    stdout, stderr = process.communicate(timeout=5)

    # Check for errors
    if "Error" in stderr or "error" in stderr or "ERROR" in stderr:
        print(f"  FAILED: {mode} mode encountered errors")
        print(f"  Errors: {stderr}")
        return False
    else:
        print(f"  SUCCESS: {mode} mode started successfully")
        return True


def main():
    """Test all launcher modes."""
    print("Unified Launcher Test\n=====================")

    # Test all modes
    modes = ["STANDARD", "DI", "STATE_MACHINE"]
    results = {}

    for mode in modes:
        results[mode] = test_launcher_mode(mode)

    # Print summary
    print("\nTest Summary\n============")
    for mode, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{mode}: {status}")

    # Check if all tests passed
    if all(results.values()):
        print("\nAll tests passed! The unified launcher is working correctly.")
        return 0
    else:
        print("\nSome tests failed. Please check the logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
