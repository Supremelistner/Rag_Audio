from pydantic import BaseModel
from typing import Literal

class ExecutionPlan(BaseModel):
    task=str
    description:str
    query_type: Literal[
            "audio",
            "text",
            "hybrid"
        ]
    song_name: str | None
    query_audio: bool    
    required_output=str
    tools=list[str]
    steps=list[str]
    step_no=int
    stems=list[str]
    mod_stem=list[str]
   
class stem_plan(BaseModel):
    song_id:list[str]
    required_stem:list[str]