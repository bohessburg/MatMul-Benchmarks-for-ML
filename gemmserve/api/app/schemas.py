from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictResponse(BaseModel):
    language: str = Field(..., description="Predicted language code")
    scores: dict[str, float] = Field(..., description="Per-language probabilities")
