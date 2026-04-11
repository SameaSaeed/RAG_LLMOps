"""
RAG Evaluation API Endpoints
Provides endpoints for evaluating RAG system performance
"""

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
import json

from scripts.rag_evaluator import RAGEvaluator, EvaluationSample
from configs.logger import GLOBAL_LOGGER


router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])
evaluator = RAGEvaluator()


class EvaluateSampleRequest(BaseModel):
    """Request model for evaluating single sample"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class EvaluateRequest(BaseModel):
    """Request model for batch evaluation"""
    samples: List[EvaluateSampleRequest]


class MetricsResponse(BaseModel):
    """Response model for evaluation metrics"""
    sample_id: int
    question: str
    answer: str
    metrics: dict
    evaluated_at: str


class SummaryMetricsResponse(BaseModel):
    """Response model for summary metrics"""
    total_samples: int
    total_evaluated: int
    metrics_summary: dict


class ExportResponse(BaseModel):
    """Response model for export"""
    format: str
    filename: str
    message: str


@router.post("/sample", response_model=MetricsResponse)
async def evaluate_sample(request: EvaluateSampleRequest):
    """
    Evaluate a single RAG sample

    Args:
        request: Evaluation sample with question, answer, and contexts

    Returns:
        Evaluation metrics for the sample
    """
    try:
        sample = EvaluationSample(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth
        )

        result = evaluator.evaluate_sample(sample, sample_id=len(evaluator.results))
        evaluator.results.append(result)

        GLOBAL_LOGGER.info(
            "Evaluated RAG sample",
            sample_id=result.sample_id,
            question=request.question[:50],
            metrics=result.metrics
        )

        return MetricsResponse(
            sample_id=result.sample_id,
            question=result.question,
            answer=result.answer,
            metrics=result.metrics,
            evaluated_at=result.evaluated_at
        )

    except Exception as e:
        GLOBAL_LOGGER.error("Failed to evaluate sample", error=str(e))
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {str(e)}")


@router.post("/batch", response_model=List[MetricsResponse])
async def evaluate_batch(request: EvaluateRequest):
    """
    Evaluate multiple RAG samples in batch

    Args:
        request: List of evaluation samples

    Returns:
        List of evaluation metrics for each sample
    """
    try:
        results = []

        for idx, sample_req in enumerate(request.samples):
            sample = EvaluationSample(
                question=sample_req.question,
                answer=sample_req.answer,
                contexts=sample_req.contexts,
                ground_truth=sample_req.ground_truth
            )

            result = evaluator.evaluate_sample(
                sample,
                sample_id=len(evaluator.results) + idx
            )
            evaluator.results.append(result)

            results.append(MetricsResponse(
                sample_id=result.sample_id,
                question=result.question,
                answer=result.answer,
                metrics=result.metrics,
                evaluated_at=result.evaluated_at
            ))

        GLOBAL_LOGGER.info(
            "Evaluated batch of RAG samples",
            batch_size=len(results),
            total_evaluated=len(evaluator.results)
        )

        return results

    except Exception as e:
        GLOBAL_LOGGER.error("Batch evaluation failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Batch evaluation failed: {str(e)}")


@router.get("/summary", response_model=SummaryMetricsResponse)
async def get_summary():
    """
    Get summary metrics for all evaluations

    Returns:
        Summary statistics across all evaluated samples
    """
    try:
        summary = evaluator.get_summary_metrics()

        return SummaryMetricsResponse(
            total_samples=len(evaluator.samples),
            total_evaluated=len(evaluator.results),
            metrics_summary=summary
        )

    except Exception as e:
        GLOBAL_LOGGER.error("Failed to get summary metrics", error=str(e))
        raise HTTPException(status_code=400, detail=f"Failed to get summary: {str(e)}")


@router.get("/results")
async def get_results(limit: int = 100, offset: int = 0):
    """
    Get evaluation results with pagination

    Args:
        limit: Maximum number of results to return
        offset: Number of results to skip

    Returns:
        List of evaluation results
    """
    try:
        results = evaluator.results[offset:offset + limit]

        return {
            "total": len(evaluator.results),
            "limit": limit,
            "offset": offset,
            "results": [
                {
                    "sample_id": r.sample_id,
                    "question": r.question,
                    "answer": r.answer,
                    "metrics": r.metrics,
                    "evaluated_at": r.evaluated_at
                }
                for r in results
            ]
        }

    except Exception as e:
        GLOBAL_LOGGER.error("Failed to get results", error=str(e))
        raise HTTPException(status_code=400, detail=f"Failed to get results: {str(e)}")


@router.post("/upload")
async def upload_evaluation_file(file: UploadFile = File(...)):
    """
    Upload JSON file with evaluation samples

    Expected JSON format:
    {
        "samples": [
            {
                "question": "...",
                "answer": "...",
                "contexts": ["...", "..."],
                "ground_truth": "..." (optional)
            }
        ]
    }

    Args:
        file: JSON file with evaluation samples

    Returns:
        Evaluation results for uploaded samples
    """
    try:
        content = await file.read()
        data = json.loads(content.decode())

        samples = data.get("samples", [])
        results = []

        for idx, sample_data in enumerate(samples):
            sample = EvaluationSample(
                question=sample_data["question"],
                answer=sample_data["answer"],
                contexts=sample_data["contexts"],
                ground_truth=sample_data.get("ground_truth")
            )

            result = evaluator.evaluate_sample(
                sample,
                sample_id=len(evaluator.results) + idx
            )
            evaluator.results.append(result)
            results.append(result)

        GLOBAL_LOGGER.info(
            "Uploaded evaluation file",
            filename=file.filename,
            samples_count=len(samples),
            total_evaluated=len(evaluator.results)
        )

        return {
            "filename": file.filename,
            "samples_loaded": len(samples),
            "total_evaluated": len(evaluator.results),
            "results": [
                {
                    "sample_id": r.sample_id,
                    "question": r.question,
                    "metrics": r.metrics
                }
                for r in results
            ]
        }

    except json.JSONDecodeError:
        GLOBAL_LOGGER.error("Invalid JSON file uploaded")
        raise HTTPException(status_code=400, detail="File must be valid JSON")
    except KeyError as e:
        GLOBAL_LOGGER.error("Missing required field in JSON", field=str(e))
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except Exception as e:
        GLOBAL_LOGGER.error("File upload failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")


@router.post("/export/{format}")
async def export_results(format: str = "json"):
    """
    Export evaluation results in specified format

    Args:
        format: Export format (json, csv)

    Returns:
        Export confirmation
    """
    try:
        if format.lower() not in ["json", "csv"]:
            raise ValueError("Format must be 'json' or 'csv'")

        if format.lower() == "json":
            filename = "evaluation_results.json"
            evaluator.export_to_json(filename)
        else:
            filename = "evaluation_results.csv"
            evaluator.export_to_csv(filename)

        GLOBAL_LOGGER.info(
            "Exported evaluation results",
            format=format,
            filename=filename,
            samples=len(evaluator.results)
        )

        return ExportResponse(
            format=format,
            filename=filename,
            message=f"Successfully exported {len(evaluator.results)} results to {filename}"
        )

    except Exception as e:
        GLOBAL_LOGGER.error("Export failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Export failed: {str(e)}")


@router.delete("/clear")
async def clear_results():
    """
    Clear all evaluation results and samples

    Returns:
        Confirmation message
    """
    try:
        evaluator.results.clear()
        evaluator.samples.clear()

        GLOBAL_LOGGER.info("Cleared all evaluation results")

        return {
            "message": "All evaluation results and samples cleared",
            "total_cleared": len(evaluator.results)
        }

    except Exception as e:
        GLOBAL_LOGGER.error("Failed to clear results", error=str(e))
        raise HTTPException(status_code=400, detail=f"Clear failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Check evaluator service health"""
    return {
        "status": "healthy",
        "total_samples": len(evaluator.samples),
        "total_results": len(evaluator.results)
    }
