"""
CLI interface for running evaluations.

Usage:
    python -m vizra.evaluation list
    python -m vizra.evaluation run <evaluation_name>
    python -m vizra.evaluation run-all
"""

import sys
import argparse
from .runner import EvaluationRunner


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vizra Agent Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m vizra.evaluation list
  python -m vizra.evaluation run chord_identifier_eval
  python -m vizra.evaluation run-all --report output.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available evaluations')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a specific evaluation')
    run_parser.add_argument('evaluation_name', help='Name of the evaluation to run')
    run_parser.add_argument('--report', help='Path to save report', default=None)
    
    # Run-all command
    run_all_parser = subparsers.add_parser('run-all', help='Run all evaluations')
    run_all_parser.add_argument('--report', help='Path to save report', default=None)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize runner
    runner = EvaluationRunner()
    
    if args.command == 'list':
        # List all evaluations
        evaluations = runner.list_evaluations()
        
        if not evaluations:
            print("No evaluations found.")
            return 0
        
        print("\n📋 Available Evaluations:")
        print("-" * 60)
        
        for eval_info in evaluations:
            print(f"\n• {eval_info['name']}")
            print(f"  Description: {eval_info['description']}")
            print(f"  Agent: {eval_info['agent_name']}")
            print(f"  Class: {eval_info['class']}")
        
        print("\n" + "-" * 60)
        print(f"Total: {len(evaluations)} evaluations\n")
        
    elif args.command == 'run':
        # Run specific evaluation
        try:
            results = runner.run_evaluation(args.evaluation_name)
            
            if args.report:
                runner.generate_report(results, args.report)
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
    
    elif args.command == 'run-all':
        # Run all evaluations
        try:
            results = runner.run_all_evaluations()
            
            if args.report:
                runner.generate_report(results, args.report)
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())