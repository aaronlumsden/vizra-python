#!/usr/bin/env python
"""
Demo script showing how to use the Vizra evaluation and training framework.
"""

from vizra.evaluation import EvaluationRunner
from vizra.training import TrainingRunner


def main():
    print("🎯 Vizra Evaluation & Training Framework Demo\n")
    
    # Run evaluation
    print("=" * 60)
    print("1. Running Evaluation")
    print("=" * 60)
    
    eval_runner = EvaluationRunner()
    
    # List available evaluations
    evaluations = eval_runner.list_evaluations()
    if evaluations:
        print(f"\nFound {len(evaluations)} evaluation(s):")
        for eval_info in evaluations:
            print(f"  - {eval_info['name']}: {eval_info['description']}")
        
        # Run the first evaluation
        first_eval = evaluations[0]['name']
        print(f"\nRunning evaluation: {first_eval}")
        eval_results = eval_runner.run_evaluation(first_eval)
        
        # Generate report
        eval_runner.generate_report(eval_results, 'evaluation_report.txt')
    else:
        print("No evaluations found. Make sure example agents are registered.")
    
    # Run training
    print("\n" + "=" * 60)
    print("2. Running RL Training")
    print("=" * 60)
    
    train_runner = TrainingRunner()
    
    # List available training routines
    trainings = train_runner.list_trainings()
    if trainings:
        print(f"\nFound {len(trainings)} training routine(s):")
        for train_info in trainings:
            print(f"  - {train_info['name']}: {train_info['description']}")
        
        # Run the first training (with reduced iterations for demo)
        if trainings:
            first_training = trainings[0]['name']
            print(f"\nRunning training: {first_training}")
            
            # Note: In a real scenario, you'd run the full training
            # For demo, we'll just show the setup
            print("(Training would run here with full iterations)")
            
            # Generate report placeholder
            train_results = {
                'training_name': first_training,
                'status': 'demo',
                'message': 'Full training would run with actual agent'
            }
            train_runner.generate_report(train_results, 'training_report.txt')
    else:
        print("No training routines found.")
    
    print("\n✅ Demo complete! Check the generated report files.")


if __name__ == '__main__':
    main()