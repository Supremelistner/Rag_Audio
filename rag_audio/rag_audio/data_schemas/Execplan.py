from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ExecutionStep(BaseModel):
    step_no: int = Field(..., description="Execution order starting from 0")

    tool: Literal[
        "metadata_retriever",
        "chunk_retriever",
        "voice_modifier",
        "audio_constructor",
        "similarity_search"
    ]

    description: str = Field(
        ...,
        description="Purpose of this step."
    )

    inputs: List[str] = Field(
        default_factory=list,
        description="State keys required by this tool."
    )

    outputs: List[str] = Field(
        default_factory=list,
        description="State keys produced by this tool."
    )

    parameters: dict = Field(
        default_factory=dict,
        description="Tool-specific configuration."
    )


class ExecutionPlan(BaseModel):

    workflow: Literal[
        "voice_change",
        "stem_change",
        "mashup",
        "similarity_score"
    ]

    description: str

    required_inputs: List[str] = Field(
        default_factory=list
    )

    required_output: str

    song_name: List[str] = Field(
        default_factory=list
    )

    stems: List[
        Literal[
            "vocals",
            "drums",
            "bass",
            "guitar",
            "piano",
            "other"
        ]
    ] = Field(default_factory=list)

    modify_stems: List[
        Literal[
            "vocals",
            "drums",
            "bass",
            "guitar",
            "piano",
            "other"
        ]
    ] = Field(default_factory=list)

    constraints: List[str] = Field(
        default_factory=list
    )

    steps: List[ExecutionStep]
