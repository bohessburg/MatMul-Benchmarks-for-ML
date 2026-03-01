from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")
    kernel: str = Field(..., description="Which GEMM kernel is being used in the prediction forward pass")
    model_version: str = Field(..., description="MLP model version")


class PredictResponse(BaseModel):
    language: str = Field(..., description="Predicted language code")
    scores: dict[str, float] = Field(..., description="Per-language probabilities")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    kernel: str = Field(..., description="Which GEMM kernel is being used in the prediction forward pass")
    model_version: str = Field(..., description="MLP model version")


class PredictionRecord(BaseModel):
    id: int
    input_hash: str
    language: str
    scores: dict[str, float]
    latency_ms: float
    kernel: str
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True}
