"""Response models — these define the exact JSON contract the assignment asks for.

Using Literal types means the service can *only* emit allowed values. If any
code path tried to return an off-contract string, validation would fail here
rather than shipping malformed JSON to the caller.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GenderLabel = Literal["male", "female", "unknown"]
AgeBracketLabel = Literal["18-30", "31-45", "46-60", "60+", "unknown"]
AudioQualityLabel = Literal["good", "degraded", "insufficient"]


class GenderResult(BaseModel):
    prediction: GenderLabel
    confidence: float = Field(ge=0.0, le=1.0)


class AgeResult(BaseModel):
    prediction: AgeBracketLabel
    confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    contact_id: str
    gender: GenderResult
    age_bracket: AgeResult
    processing_ms: int
    audio_quality: AudioQualityLabel

    model_config = {
        "json_schema_extra": {
            "example": {
                "contact_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
                "gender": {"prediction": "male", "confidence": 0.87},
                "age_bracket": {"prediction": "31-45", "confidence": 0.63},
                "processing_ms": 142,
                "audio_quality": "good",
            }
        }
    }
