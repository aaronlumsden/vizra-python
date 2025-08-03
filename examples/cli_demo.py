#!/usr/bin/env python
"""
Demo script showing Vizra CLI usage examples.

This script demonstrates the various CLI commands available.
Run this after installing vizra with: pip install -e .
"""

import subprocess
import sys


def run_command(cmd):
    """Run a CLI command and display the output."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode


def main():
    print("🎯 Vizra CLI Demo")
    print("This demonstrates the Vizra command-line interface.\n")
    
    # Show version
    print("\n1. Check Vizra version:")
    run_command(['vizra', '--version'])
    
    # Show help
    print("\n2. Show general help:")
    run_command(['vizra', '--help'])
    
    # Show status
    print("\n3. Check installation status:")
    run_command(['vizra', 'status'])
    
    # List evaluations
    print("\n4. List available evaluations:")
    run_command(['vizra', 'eval', 'list'])
    
    # Show eval help
    print("\n5. Show evaluation command help:")
    run_command(['vizra', 'eval', '--help'])
    
    # List trainings
    print("\n6. List available training routines:")
    run_command(['vizra', 'train', 'list'])
    
    # Show train help  
    print("\n7. Show training command help:")
    run_command(['vizra', 'train', '--help'])
    
    print("\n" + "="*60)
    print("✅ CLI Demo Complete!")
    print("\nExample usage:")
    print("  vizra eval run chord_identifier_eval -v")
    print("  vizra train run chord_identifier_training -i 10")
    print("  vizra eval run chord_identifier_eval -o results.json")
    print("\nNote: The commands will work once you have agents and evaluations defined.")


if __name__ == '__main__':
    main()