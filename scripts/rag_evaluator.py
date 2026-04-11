"""
RAGAS-based RAG Evaluation Module

Provides comprehensive evaluation metrics for RAG systems including:
- Faithfulness: How consistent is the answer with the context?
- Answer Relevancy: How relevant is the answer to the question?
- Context Relevancy: How relevant is the retrieved context to the question?
- Context Precision: What proportion of retrieved context is relevant?
- Context Recall: How much of the relevant context is retrieved?
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from pydantic import BaseModel, ConfigDict
from configs.logger import CustomLogger
from configs.exceptions import DocumentPortalException

# Initialize logger
log = CustomLogger().get_logger(__name__)


class EvaluationSample(BaseModel):
    """Single evaluation sample"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is RAG?",
                "answer": "RAG is Retrieval-Augmented Generation",
                "contexts": ["RAG combines retrieval and generation..."],
                "ground_truth": "Retrieval-Augmented Generation"
            }
        }
    )


class EvaluationResult(BaseModel):
    """RAG evaluation result"""
    sample_id: int
    question: str
    answer: str
    evaluated_at: str
    metrics: Dict[str, float]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sample_id": 1,
                "question": "What is RAG?",
                "answer": "RAG is Retrieval-Augmented Generation",
                "evaluated_at": "2026-04-11T12:00:00",
                "metrics": {
                    "faithfulness": 0.95,
                    "answer_relevancy": 0.89,
                    "context_relevancy": 0.92
                }
            }
        }
    )


class RAGEvaluator:
    """
    RAGAS-based RAG evaluation system

    Evaluates RAG pipelines using RAGAS metrics:
    - Faithfulness (0 - 1): Answer consistency with retrieved context
    - Answer Relevancy (0 - 1): Answer relevance to question
    - Context Relevancy (0 - 1): Context relevance to question
    - Context Precision (0 - 1): Proportion of relevant context
    - Context Recall (0 - 1): Coverage of all relevant information
    """

    def __init__(self, llm_model: Optional[Any] = None):
        """
        Initialize RAG Evaluator

        Args:
            llm_model: LLM instance for evaluation (optional)
        """
        self.llm_model = llm_model
        self.results: List[EvaluationResult] = []
        self.samples: List[EvaluationSample] = []

        log.info("RAGEvaluator initialized")

    def add_sample(self, sample: EvaluationSample) -> None:
        """Add evaluation sample"""
        self.samples.append(sample)
        log.info(
            "Sample added to evaluation set",
            question=sample.question[:50],
            num_contexts=len(sample.contexts)
        )

    def add_samples_from_file(self, filepath: str) -> None:
        """Load evaluation samples from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    sample = EvaluationSample(**item)
                    self.add_sample(sample)
            else:
                sample = EvaluationSample(**data)
                self.add_sample(sample)

            log.info(
                "Samples loaded from file",
                filepath=filepath,
                count=len(self.samples)
            )
        except Exception as e:
            log.error("Failed to load samples", filepath=filepath, error=str(e))
            raise DocumentPortalException(
                "Failed to load evaluation samples",
                e,
                context={"filepath": filepath}
            )

    def _calculate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Calculate faithfulness score

        Simple heuristic: Check if key terms from answer appear in contexts
        In production, use RAGAS faithfulness metric
        """
        try:
            if not contexts or not answer:
                return 0.0

            # Simple keyword overlap
            answer_words = set(answer.lower().split())
            context_text = " ".join(contexts).lower()
            context_words = set(context_text.split())

            # Calculate overlap ratio
            overlap = len(answer_words & context_words)
            faithfulness = min(overlap / max(len(answer_words), 1), 1.0)

            return round(faithfulness, 4)
        except Exception as e:
            log.warning("Faithfulness calculation failed", error=str(e))
            return 0.0

    def _calculate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Calculate answer relevancy score

        Simple heuristic: Check if question terms appear in answer
        In production, use semantic similarity
        """
        try:
            if not question or not answer:
                return 0.0

            # Extract key terms (non-stop words)
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                'to', 'for', 'of', 'with', 'from', 'is', 'are', 'was',
                'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                'did', 'will', 'would', 'could', 'should', 'may', 'might'
            }

            question_words = {
                w.lower() for w in question.split()
                if w.lower() not in stop_words and len(w) > 2
            }
            answer_text = answer.lower()

            # Calculate keyword presence
            if not question_words:
                return 0.5  # Neutral if no key terms

            matches = sum(1 for word in question_words if word in answer_text)
            relevancy = matches / len(question_words)

            return round(relevancy, 4)
        except Exception as e:
            log.warning("Answer relevancy calculation failed", error=str(e))
            return 0.0

    def _calculate_context_relevancy(
        self,
        question: str,
        contexts: List[str]
    ) -> float:
        """
        Calculate context relevancy score

        Simple heuristic: Check if question terms appear in contexts
        """
        try:
            if not question or not contexts:
                return 0.0

            question_words = set(
                w.lower() for w in question.split()
                if len(w) > 2
            )
            context_text = " ".join(contexts).lower()

            if not question_words:
                return 0.5

            matches = sum(1 for word in question_words if word in context_text)
            relevancy = matches / len(question_words)

            return round(relevancy, 4)
        except Exception as e:
            log.warning("Context relevancy calculation failed", error=str(e))
            return 0.0

    def _calculate_context_precision(
        self,
        question: str,
        contexts: List[str]
    ) -> float:
        """
        Calculate context precision

        Proportion of retrieved contexts that contain relevant information
        """
        try:
            if not contexts:
                return 0.0

            question_words = set(
                w.lower() for w in question.split()
                if len(w) > 2
            )

            relevant_contexts = 0
            for context in contexts:
                context_words = set(w.lower() for w in context.split())
                if question_words & context_words:
                    relevant_contexts += 1

            precision = relevant_contexts / len(contexts)
            return round(precision, 4)
        except Exception as e:
            log.warning("Context precision calculation failed", error=str(e))
            return 0.0

    def _calculate_context_recall(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Calculate context recall

        How much of the answer content is covered by the retrieved contexts
        """
        try:
            if not contexts or not answer:
                return 0.0

            # Simple: Check if all answer terms are in contexts
            answer_words = set(
                w.lower() for w in answer.split()
                if len(w) > 2
            )
            context_text = " ".join(contexts).lower()
            context_words = set(w.lower() for w in context_text.split())

            if not answer_words:
                return 1.0

            covered = len(answer_words & context_words)
            recall = covered / len(answer_words)

            return round(recall, 4)
        except Exception as e:
            log.warning("Context recall calculation failed", error=str(e))
            return 0.0

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        sample_id: int = 0
    ) -> EvaluationResult:
        """Evaluate a single sample"""
        try:
            metrics = {
                "faithfulness": self._calculate_faithfulness(
                    sample.answer,
                    sample.contexts
                ),
                "answer_relevancy": self._calculate_answer_relevancy(
                    sample.question,
                    sample.answer
                ),
                "context_relevancy": self._calculate_context_relevancy(
                    sample.question,
                    sample.contexts
                ),
                "context_precision": self._calculate_context_precision(
                    sample.question,
                    sample.contexts
                ),
                "context_recall": self._calculate_context_recall(
                    sample.answer,
                    sample.contexts
                ),
            }

            result = EvaluationResult(
                sample_id=sample_id,
                question=sample.question,
                answer=sample.answer,
                evaluated_at=datetime.now(timezone.utc).isoformat(),
                metrics=metrics
            )

            self.results.append(result)

            log.info(
                "Sample evaluated",
                sample_id=sample_id,
                avg_score=round(sum(metrics.values()) / len(metrics), 4)
            )

            return result
        except Exception as e:
            log.error(
                "Evaluation failed for sample",
                sample_id=sample_id,
                error=str(e)
            )
            raise DocumentPortalException(
                "Sample evaluation failed",
                e,
                context={"sample_id": sample_id}
            )

    def evaluate_all(self) -> List[EvaluationResult]:
        """Evaluate all samples"""
        log.info("Starting batch evaluation", total_samples=len(self.samples))

        for idx, sample in enumerate(self.samples):
            self.evaluate_sample(sample, idx)

        log.info(
            "Batch evaluation completed",
            total_samples=len(self.results)
        )

        return self.results

    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary metrics across all results"""
        if not self.results:
            return {}

        summary = {}
        metrics_keys = self.results[0].metrics.keys()

        for metric_key in metrics_keys:
            values = [r.metrics[metric_key] for r in self.results]
            summary[f"{metric_key}_mean"] = round(sum(values) / len(values), 4)
            summary[f"{metric_key}_min"] = round(min(values), 4)
            summary[f"{metric_key}_max"] = round(max(values), 4)

        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = []
        for result in self.results:
            row = {
                "sample_id": result.sample_id,
                "question": result.question,
                "answer": result.answer,
                "evaluated_at": result.evaluated_at,
            }
            row.update(result.metrics)
            data.append(row)

        return pd.DataFrame(data)

    def save_results(self, filepath: str) -> None:
        """Save evaluation results to file"""
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if filepath.endswith('.json'):
                data = [r.model_dump() for r in self.results]
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            elif filepath.endswith('.csv'):
                df = self.to_dataframe()
                df.to_csv(filepath, index=False)

            else:
                raise ValueError(f"Unsupported file format: {filepath}")

            log.info(
                "Results saved",
                filepath=filepath,
                count=len(self.results)
            )
        except Exception as e:
            log.error("Failed to save results", filepath=filepath, error=str(e))
            raise DocumentPortalException(
                "Failed to save results",
                e,
                context={"filepath": filepath}
            )

    def export_to_json(self, filepath: str = "evaluation_results.json") -> None:
        """Export evaluation results to JSON file"""
        self.save_results(filepath)

    def export_to_csv(self, filepath: str = "evaluation_results.csv") -> None:
        """Export evaluation results to CSV file"""
        self.save_results(filepath)

    @staticmethod
    def load_samples_from_json(filepath: str) -> List[dict]:
        """Load evaluation samples from JSON file and return as list of dicts"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'samples' in data:
                return data['samples']
            elif isinstance(data, list):
                return data
            else:
                return [data]
        except Exception as e:
            log.error("Failed to load samples", filepath=filepath, error=str(e))
            raise DocumentPortalException(
                "Failed to load evaluation samples",
                e,
                context={"filepath": filepath}
            )


if __name__ == "__main__":
    # Example usage
    evaluator = RAGEvaluator()

    # Add sample
    sample = EvaluationSample(
        question="What is RAG?",
        answer="RAG is Retrieval-Augmented Generation, a technique that combines retrieval and generation",
        contexts=[
            "RAG (Retrieval-Augmented Generation) is a NLP technique that combines information retrieval with generative models",
            "RAG works by first retrieving relevant documents, then using them to augment the generation process"
        ],
        ground_truth="Retrieval-Augmented Generation"
    )

    evaluator.add_sample(sample)

    # Evaluate
    result = evaluator.evaluate_all()[0]
    print(f"Evaluation Result:\n{result.model_dump_json(indent=2)}")

    # Get summary
    summary = evaluator.get_summary_metrics()
    print(f"\nSummary Metrics:\n{json.dumps(summary, indent=2)}")
