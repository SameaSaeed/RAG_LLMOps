#!/usr/bin/env python3
"""
LangSmith Evaluation Script for RAG System

This script runs evaluations on the AI Engineering Report RAG system using
custom correctness evaluator and LangSmith.

Usage:
    python run_evaluations.py --dataset AgenticAIReportGoldens
    python run_evaluations.py --dataset AgenticAIReportGoldens --evaluator correctness
    python run_evaluations.py --dataset AgenticAIReportGoldens --evaluator all
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.index import ChatIngestor
from src.RAG import ConversationalRAG

# Load environment variables
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


# ============================================================================
# File Adapter and RAG Function
# ============================================================================

class LocalFileAdapter:
    """Adapter for local file paths to work with ChatIngestor."""

    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.name = self.path.name

    def getbuffer(self) -> bytes:
        return self.path.read_bytes()


def answer_ai_report_question(
    inputs: dict,
    data_path: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 5
) -> dict:

    if data_path is None:
        data_path = str(PROJECT_ROOT / "data" / "The 2025 AI Engineering Report.txt")

    try:
        question = inputs.get("question", "")
        if not question:
            return {"answer": "No question provided"}

        if not Path(data_path).exists():
            return {"answer": f"Data file not found: {data_path}"}

        # ✅ Astra Init
        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        db_id = os.getenv("ASTRA_DB_ID")

        if not token or not db_id:
            return {"answer": "Missing AstraDB credentials"}

        import cassio
        cassio.init(token=token, database_id=db_id)

        # ✅ Load + Split
        from utils.docs_loader import load_documents
        from utils.model_loader import ModelLoader

        file_adapter = LocalFileAdapter(data_path)

        temp_path = Path("data/eval_temp")
        temp_path.mkdir(parents=True, exist_ok=True)

        temp_file = temp_path / file_adapter.name
        temp_file.write_bytes(file_adapter.getbuffer())

        docs = load_documents([temp_file])

        ingestor = ChatIngestor(use_session_dirs=False)
        chunks = ingestor._split(docs, chunk_size, chunk_overlap)

        # ✅ Embeddings
        embeddings = ModelLoader().load_embeddings()

        # ✅ Astra Vector Store
        from langchain_community.vectorstores.cassandra import Cassandra

        table_name = "qa_mini_demo"

        vectorstore = Cassandra(
            embedding=embeddings,
            table_name=table_name,
        )

        # ✅ Add documents (IMPORTANT)
        vectorstore.add_documents(chunks)

        # ✅ Retriever
        retriever = vectorstore.as_retriever()

        # ✅ RAG
        rag = ConversationalRAG(
            session_id="eval_session",
            retriever=retriever
        )

        rag._build_lcel_chain()

        answer = rag.invoke(question, chat_history=[])

        return {"answer": answer}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


# ============================================================================
# Custom Correctness Evaluator
# ============================================================================

def correctness_evaluator(run: Run, example: Example) -> dict:
    """
    Custom LLM-as-a-Judge evaluator for correctness.

    Correctness means how well the actual model output matches the reference output
    in terms of factual accuracy, coverage, and meaning.

    Args:
        run: The Run object containing the actual outputs
        example: The Example object containing the expected outputs

    Returns:
        dict with 'score' (1 for correct, 0 for incorrect) and 'reasoning'
    """
    # Extract actual and expected outputs
    actual_output = run.outputs.get("answer", "")
    expected_output = example.outputs.get("answer", "")
    input_question = example.inputs.get("question", "")

    # Define the evaluation prompt
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluator whose job is to judge correctness.

Correctness means how well the actual model output matches the reference output in terms of factual accuracy, coverage, and meaning.

- If the actual output matches the reference output semantically (even if wording differs), it should be marked correct.
- If the output misses key facts, introduces contradictions, or is factually incorrect, it should be marked incorrect.

Do not penalize for stylistic or formatting differences unless they change meaning."""),
        ("human", """<example>
<input>
{input}
</input>

<output>
Expected Output: {expected_output}

Actual Output: {actual_output}
</output>
</example>

Please grade the following agent run given the input, expected output, and actual output.
Focus only on correctness (semantic and factual alignment).

Respond with:
1. A brief reasoning (1-2 sentences)
2. A final verdict: either "CORRECT" or "INCORRECT"

Format your response as:
Reasoning: [your reasoning]
Verdict: [CORRECT or INCORRECT]""")
    ])

    # Initialize LLM (using Gemini as shown in your config)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0
    )

    # Create chain and invoke
    chain = eval_prompt | llm

    try:
        response = chain.invoke({
            "input": input_question,
            "expected_output": expected_output,
            "actual_output": actual_output
        })

        response_text = response.content

        # Parse the response
        reasoning = ""
        verdict = ""

        for line in response_text.split('\n'):
            if line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
            elif line.startswith("Verdict:"):
                verdict = line.replace("Verdict:", "").strip()

        # Convert verdict to score (1 for correct, 0 for incorrect)
        score = 1 if "CORRECT" in verdict.upper() else 0

        return {
            "key": "correctness",
            "score": score,
            "reasoning": reasoning,
            "comment": f"Verdict: {verdict}"
        }

    except Exception as e:
        return {
            "key": "correctness",
            "score": 0,
            "reasoning": f"Error during evaluation: {str(e)}"
        }


# ============================================================================
# Main Evaluation Function
# ============================================================================

def run_evaluation(
    dataset_name: str = "AgenticAIReportGoldens",
    evaluator_type: str = "correctness",
    experiment_prefix: Optional[str] = None,
    description: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 5
):
    """
    Run evaluation on the RAG system.

    Args:
        dataset_name: Name of the dataset in LangSmith
        evaluator_type: Type of evaluator to use ('correctness', 'cot_qa', 'all')
        experiment_prefix: Prefix for the experiment name
        description: Description for the experiment
        chunk_size: Chunk size for document splitting
        chunk_overlap: Overlap between chunks
        k: Number of documents to retrieve
    """
    print(f"\n{'='*80}")
    print(f"Running Evaluation on Dataset: {dataset_name}")
    print(f"Evaluator Type: {evaluator_type}")
    print(f"{'='*80}\n")

    # Select evaluators based on type
    evaluators = []

    if evaluator_type == "correctness":
        evaluators = [correctness_evaluator]
        exp_prefix = experiment_prefix or "agenticAIReport-correctness"
        desc = description or "Evaluating RAG system with custom correctness evaluator (LLM-as-a-Judge)"

    elif evaluator_type == "cot_qa":
        evaluators = [LangChainStringEvaluator("cot_qa")]
        exp_prefix = experiment_prefix or "agenticAIReport-cot-qa"
        desc = description or "Evaluating RAG system with Chain-of-Thought QA evaluator"

    elif evaluator_type == "all":
        evaluators = [
            correctness_evaluator,
            LangChainStringEvaluator("cot_qa")
        ]
        exp_prefix = experiment_prefix or "agenticAIReport-multi-eval"
        desc = description or "Evaluating RAG system with multiple evaluators (correctness + cot_qa)"

    else:
        print(f"Error: Unknown evaluator type '{evaluator_type}'")
        print("Available types: correctness, cot_qa, all")
        return None

    # Prepare metadata
    metadata = {
        "variant": "RAG with FAISS and AI Engineering Report",
        "evaluator_type": evaluator_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k": k,
    }

    if evaluator_type == "correctness" or evaluator_type == "all":
        metadata["llm_judge_model"] = "gemini-2.5-pro"

    print("Starting evaluation...")

    # Format evaluator names for display
    evaluator_names = []
    for e in evaluators:
        if isinstance(e, str):
            evaluator_names.append(e)
        elif hasattr(e, '__name__'):
            evaluator_names.append(e.__name__)
        elif hasattr(e, '__class__'):
            evaluator_names.append(e.__class__.__name__)
        else:
            evaluator_names.append(str(e))

    print(f"Evaluators: {evaluator_names}")
    print(f"Metadata: {metadata}\n")

    # Run evaluation
    try:
        experiment_results = evaluate(
            answer_ai_report_question,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=exp_prefix,
            description=desc,
            metadata=metadata,
        )

        print("\n" + "="*80)
        print("Evaluation Completed Successfully!")
        print("="*80)

        # Print summary if available
        if hasattr(experiment_results, 'experiment_name'):
            print(f"\nExperiment Name: {experiment_results.experiment_name}")

        print("\nCheck the LangSmith UI for detailed results:")
        print("https://smith.langchain.com/")

        return experiment_results

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Error during evaluation: {str(e)}")
        print(f"{'='*80}\n")
        raise


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main function to run evaluations from command line."""
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluations on the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with correctness evaluator
  python run_evaluations.py --dataset AgenticAIReportGoldens --evaluator correctness

  # Run with chain-of-thought QA evaluator
  python run_evaluations.py --dataset AgenticAIReportGoldens --evaluator cot_qa

  # Run with all evaluators
  python run_evaluations.py --dataset AgenticAIReportGoldens --evaluator all

  # Run with custom parameters
  python run_evaluations.py --dataset AgenticAIReportGoldens --evaluator correctness --chunk-size 500 --k 10
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="AgenticAIReportGoldens",
        help="Name of the dataset in LangSmith (default: AgenticAIReportGoldens)"
    )

    parser.add_argument(
        "--evaluator",
        type=str,
        choices=["correctness", "cot_qa", "all"],
        default="correctness",
        help="Type of evaluator to use (default: correctness)"
    )

    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default=None,
        help="Custom prefix for experiment name"
    )

    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Custom description for the experiment"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for document splitting (default: 1000)"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )

    args = parser.parse_args()

    # Check for required environment variables
    required_env_vars = ["LANGSMITH_API_KEY", "GOOGLE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"\nError: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment.\n")
        sys.exit(1)

    # Run evaluation
    try:
        run_evaluation(
            dataset_name=args.dataset,
            evaluator_type=args.evaluator,
            experiment_prefix=args.experiment_prefix,
            description=args.description,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            k=args.k
        )
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
