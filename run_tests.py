#!/usr/bin/env python
"""
Simple script to run the test suite.
"""

import subprocess
import sys


def main():
    """Run the test suite with coverage."""
    print("Running vizra test suite...\n")
    
    # Run tests with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=vizra",
        "--cov-report=term-missing",
        "-v"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())