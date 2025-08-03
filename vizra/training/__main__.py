"""
CLI interface for running RL training.

Usage:
    python -m vizra.training list
    python -m vizra.training run <training_name>
    python -m vizra.training run-all
"""

import sys
import argparse
from .runner import TrainingRunner


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vizra Agent RL Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m vizra.training list
  python -m vizra.training run chord_identifier_training
  python -m vizra.training run-all --report training_report.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available training routines')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a specific training routine')
    run_parser.add_argument('training_name', help='Name of the training to run')
    run_parser.add_argument('--report', help='Path to save report', default=None)
    
    # Run-all command
    run_all_parser = subparsers.add_parser('run-all', help='Run all training routines')
    run_all_parser.add_argument('--report', help='Path to save report', default=None)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize runner
    runner = TrainingRunner()
    
    if args.command == 'list':
        # List all training routines
        trainings = runner.list_trainings()
        
        if not trainings:
            print("No training routines found.")
            return 0
        
        print("\n📋 Available Training Routines:")
        print("-" * 60)
        
        for train_info in trainings:
            print(f"\n• {train_info['name']}")
            print(f"  Description: {train_info['description']}")
            print(f"  Agent: {train_info['agent_name']}")
            print(f"  Algorithm: {train_info['algorithm']}")
            print(f"  Class: {train_info['class']}")
        
        print("\n" + "-" * 60)
        print(f"Total: {len(trainings)} training routines\n")
        
    elif args.command == 'run':
        # Run specific training
        try:
            results = runner.run_training(args.training_name)
            
            if args.report:
                runner.generate_report(results, args.report)
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
    
    elif args.command == 'run-all':
        # Run all training routines
        try:
            results = runner.run_all_trainings()
            
            if args.report:
                runner.generate_report(results, args.report)
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())