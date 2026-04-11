#!/usr/bin/env python3
"""
RAG Evaluation Script
Demonstrates usage of RAGEvaluator with sample data
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag_evaluator import RAGEvaluator, EvaluationSample
from configs.logger import GLOBAL_LOGGER


def load_samples_from_json(filepath: str) -> list:
    """Load evaluation samples from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('samples', [])


def run_evaluation(samples_file: str = "scripts/evaluation_samples.json"):
    """Run RAG evaluation on sample data"""

    GLOBAL_LOGGER.info("Starting RAG evaluation", samples_file=samples_file)

    try:
        # Load samples
        samples_data = load_samples_from_json(samples_file)
        GLOBAL_LOGGER.info(f"Loaded {len(samples_data)} evaluation samples")

        # Create evaluator
        evaluator = RAGEvaluator()

        # Add samples
        for sample_data in samples_data:
            sample = EvaluationSample(
                question=sample_data['question'],
                answer=sample_data['answer'],
                contexts=sample_data['contexts'],
                ground_truth=sample_data.get('ground_truth')
            )
            evaluator.add_sample(sample)

        GLOBAL_LOGGER.info(f"Added {len(evaluator.samples)} samples to evaluator")

        # Run evaluation
        GLOBAL_LOGGER.info("Running evaluation on all samples...")
        results = evaluator.evaluate_all()

        GLOBAL_LOGGER.info(f"Evaluation complete. Evaluated {len(results)} samples")

        # Print results
        print("\n" + "="*80)
        print("RAG EVALUATION RESULTS")
        print("="*80 + "\n")

        for idx, result in enumerate(results, 1):
            print(f"Sample {idx}: {result.question[:60]}...")
            print(f"  Answer: {result.answer[:60]}...")
            print("  Metrics:")
            for metric_name, metric_value in result.metrics.items():
                print(f"    {metric_name}: {metric_value:.4f}")
            print()

        # Get summary metrics
        summary = evaluator.get_summary_metrics()

        print("="*80)
        print("SUMMARY METRICS")
        print("="*80 + "\n")
        print(f"Total Samples Evaluated: {len(results)}\n")

        for metric_name, metric_value in summary.items():
            if "mean" in metric_name:
                metric_type = metric_name.replace("_mean", "")
                min_val = summary.get(f"{metric_type}_min", 0)
                max_val = summary.get(f"{metric_type}_max", 0)
                print(f"{metric_type.upper()}:")
                print(f"  Mean: {metric_value:.4f}")
                print(f"  Min:  {min_val:.4f}")
                print(f"  Max:  {max_val:.4f}\n")

        # Export results
        print("="*80)
        print("EXPORTING RESULTS")
        print("="*80 + "\n")

        json_file = "evaluation_results.json"
        csv_file = "evaluation_results.csv"

        evaluator.export_to_json(json_file)
        GLOBAL_LOGGER.info(f"Exported results to {json_file}")
        print(f"✓ Exported JSON results to {json_file}")

        evaluator.export_to_csv(csv_file)
        GLOBAL_LOGGER.info(f"Exported results to {csv_file}")
        print(f"✓ Exported CSV results to {csv_file}")

        # Convert to DataFrame
        df = evaluator.to_dataframe()
        print(f"\n✓ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        print(f"\nDataFrame columns: {', '.join(df.columns.tolist())}\n")

        print("="*80)
        print("EVALUATION COMPLETE")
        print("="*80)

        return evaluator, results, summary

    except FileNotFoundError:
        GLOBAL_LOGGER.error(f"Samples file not found: {samples_file}")
        print(f"Error: Could not find {samples_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        GLOBAL_LOGGER.error("Invalid JSON in samples file")
        print("Error: Invalid JSON format in samples file")
        sys.exit(1)
    except Exception as e:
        GLOBAL_LOGGER.error(f"Evaluation failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Check for custom samples file argument
    samples_file = sys.argv[1] if len(sys.argv) > 1 else "scripts/evaluation_samples.json"

    evaluator, results, summary = run_evaluation(samples_file)

    # Print sample metrics details
    print("\nDETAILED METRICS BREAKDOWN:")
    print("-" * 80)

    metric_names = ["faithfulness", "answer_relevancy", "context_relevancy",
                    "context_precision", "context_recall"]

    for metric in metric_names:
        values = [r.metrics.get(metric, 0) for r in results]
        avg = sum(values) / len(values) if values else 0
        print(f"{metric:20s}: avg={avg:.4f}, min={min(values):.4f}, max={max(values):.4f}")
