import pytest
from scripts.rag_evaluator import RAGEvaluator, EvaluationSample, EvaluationResult


@pytest.fixture
def evaluator():
    """Create RAG evaluator instance"""
    return RAGEvaluator()


@pytest.fixture
def sample_data():
    """Sample evaluation data"""
    return EvaluationSample(
        question="What is RAG?",
        answer="RAG is Retrieval-Augmented Generation, a technique combining retrieval and generation",
        contexts=[
            "RAG combines information retrieval with generative models",
            "RAG retrieves relevant documents then augments generation"
        ],
        ground_truth="Retrieval-Augmented Generation"
    )


class TestRAGEvaluator:
    """Tests for RAG Evaluator"""

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator is not None
        assert evaluator.results == []
        assert evaluator.samples == []

    def test_add_sample(self, evaluator, sample_data):
        """Test adding evaluation sample"""
        evaluator.add_sample(sample_data)
        assert len(evaluator.samples) == 1
        assert evaluator.samples[0].question == sample_data.question

    def test_add_multiple_samples(self, evaluator):
        """Test adding multiple samples"""
        for i in range(3):
            sample = EvaluationSample(
                question=f"Question {i}",
                answer=f"Answer {i}",
                contexts=[f"Context {i}"]
            )
            evaluator.add_sample(sample)

        assert len(evaluator.samples) == 3

    def test_evaluate_single_sample(self, evaluator, sample_data):
        """Test evaluating single sample"""
        result = evaluator.evaluate_sample(sample_data, sample_id=0)

        assert isinstance(result, EvaluationResult)
        assert result.sample_id == 0
        assert "faithfulness" in result.metrics
        assert "answer_relevancy" in result.metrics
        assert "context_relevancy" in result.metrics
        assert "context_precision" in result.metrics
        assert "context_recall" in result.metrics

    def test_faithfulness_score(self, evaluator):
        """Test faithfulness score calculation"""
        answer = "RAG combines retrieval and generation"
        contexts = ["RAG is a technique using retrieval and generation"]

        score = evaluator._calculate_faithfulness(answer, contexts)
        assert 0 <= score <= 1
        assert score > 0  # Should have some overlap

    def test_answer_relevancy_score(self, evaluator):
        """Test answer relevancy score"""
        question = "What is machine learning?"
        answer = "Machine learning is a subset of AI that enables systems to learn from data"

        score = evaluator._calculate_answer_relevancy(question, answer)
        assert 0 <= score <= 1
        assert score > 0  # Should be relevant

    def test_context_relevancy_score(self, evaluator):
        """Test context relevancy score"""
        question = "What is RAG?"
        contexts = [
            "RAG is Retrieval-Augmented Generation",
            "Neural networks are deep learning models"
        ]

        score = evaluator._calculate_context_relevancy(question, contexts)
        assert 0 <= score <= 1

    def test_context_precision_score(self, evaluator):
        """Test context precision score"""
        question = "What is machine learning?"
        contexts = [
            "Machine learning is a branch of AI",
            "Another relevant context about learning",
            "Unrelated context about cooking"
        ]

        score = evaluator._calculate_context_precision(question, contexts)
        assert 0 <= score <= 1

    def test_context_recall_score(self, evaluator):
        """Test context recall score"""
        answer = "RAG combines retrieval and generation"
        contexts = ["RAG combines retrieval and generation techniques"]

        score = evaluator._calculate_context_recall(answer, contexts)
        assert 0 <= score <= 1
        assert score > 0.5  # Should have good coverage

    def test_evaluate_all(self, evaluator):
        """Test batch evaluation"""
        samples = [
            EvaluationSample(
                question="Q1",
                answer="Answer to Q1",
                contexts=["Context for Q1"]
            ),
            EvaluationSample(
                question="Q2",
                answer="Answer to Q2",
                contexts=["Context for Q2"]
            )
        ]

        for sample in samples:
            evaluator.add_sample(sample)

        results = evaluator.evaluate_all()
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_get_summary_metrics(self, evaluator):
        """Test summary metrics generation"""
        sample = EvaluationSample(
            question="What is AI?",
            answer="AI is Artificial Intelligence",
            contexts=["AI is a field of computer science"]
        )
        evaluator.add_sample(sample)
        evaluator.evaluate_all()

        summary = evaluator.get_summary_metrics()
        assert "faithfulness_mean" in summary
        assert "answer_relevancy_mean" in summary
        assert "context_relevancy_mean" in summary
        assert all(isinstance(v, float) for v in summary.values())

    def test_to_dataframe(self, evaluator):
        """Test conversion to DataFrame"""
        sample = EvaluationSample(
            question="What is RAG?",
            answer="RAG is Retrieval-Augmented Generation",
            contexts=["RAG combines retrieval and generation"]
        )
        evaluator.add_sample(sample)
        evaluator.evaluate_all()

        df = evaluator.to_dataframe()
        assert len(df) == 1
        assert "sample_id" in df.columns
        assert "question" in df.columns
        assert "faithfulness" in df.columns

    def test_empty_sample_handling(self, evaluator):
        """Test handling of empty samples"""
        score = evaluator._calculate_faithfulness("", [])
        assert score == 0.0

        score = evaluator._calculate_answer_relevancy("", "")
        assert score == 0.0

    def test_evaluate_sample_with_different_content_lengths(self, evaluator):
        """Test evaluation with different content sizes"""
        short_sample = EvaluationSample(
            question="Q?",
            answer="A",
            contexts=["C"]
        )

        long_sample = EvaluationSample(
            question="What is a comprehensive question about machine learning systems?",
            answer="A comprehensive answer about machine learning systems and their applications",
            contexts=[
                "Machine learning is a complex field with many applications",
                "Systems learn from data through various algorithms"
            ]
        )

        evaluator.add_sample(short_sample)
        evaluator.add_sample(long_sample)

        results = evaluator.evaluate_all()
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)


class TestEvaluationSample:
    """Tests for EvaluationSample model"""

    def test_sample_creation(self):
        """Test sample creation"""
        sample = EvaluationSample(
            question="What is RAG?",
            answer="RAG is Retrieval-Augmented Generation",
            contexts=["Context 1", "Context 2"],
            ground_truth="Retrieval-Augmented Generation"
        )

        assert sample.question == "What is RAG?"
        assert sample.answer == "RAG is Retrieval-Augmented Generation"
        assert len(sample.contexts) == 2
        assert sample.ground_truth == "Retrieval-Augmented Generation"

    def test_sample_optional_ground_truth(self):
        """Test sample with optional ground truth"""
        sample = EvaluationSample(
            question="Q",
            answer="A",
            contexts=["C"]
        )

        assert sample.ground_truth is None


class TestEvaluationResult:
    """Tests for EvaluationResult model"""

    def test_result_creation(self):
        """Test result creation"""
        result = EvaluationResult(
            sample_id=1,
            question="Q",
            answer="A",
            evaluated_at="2026-04-11T12:00:00",
            metrics={"faithfulness": 0.9, "answer_relevancy": 0.85}
        )

        assert result.sample_id == 1
        assert result.question == "Q"
        assert result.answer == "A"
        assert result.metrics["faithfulness"] == 0.9
