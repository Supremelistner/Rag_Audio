from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..data_schemas.Execplan import ExecutionPlan
from ..data_schemas.schema_song import AudioMetadata
from .tools import toollist

LIGHTWEIGHT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class AgentState(TypedDict, total=False):
    command: str
    input_song_id: Optional[str]
    audio_name: Optional[str]
    query_audio_path: Optional[str]
    input_audio_path: Optional[str]
    messages: list[Any]
    input_song_metadata: Any
    execution_plan: ExecutionPlan
    songs_metadata: list[Any]
    retrieved_chunks: dict[str, Any]
    modified_stems: dict[str, Any]
    audio_path: Optional[str]
    similarity_query_song_id: Optional[str]
    similarity_results: list[dict[str, Any]]
    current_step_index: int
    final_song_metadata: dict[str, Any]
    final_output: dict[str, Any]


class AudioAgent:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("plan", self._plan)
        workflow.add_node("execute_step", self._execute_step)
        workflow.add_node("finalize", self._finalize)
        workflow.add_edge(START, "plan")
        workflow.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "execute_step": "execute_step",
                "finalize": "finalize",
            },
        )
        workflow.add_conditional_edges(
            "execute_step",
            self._route_after_step,
            {
                "execute_step": "execute_step",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def _plan(self, state: AgentState):
        planning_command = state["command"]
        input_song_id = state.get("input_song_id")
        input_song_metadata = None

        if state.get("audio_name"):
            planning_command = f"Query audio_name={state['audio_name']}. {planning_command}"
        if state.get("query_audio_path"):
            planning_command = f"Query audio_path={state['query_audio_path']}. {planning_command}"
        if state.get("input_audio_path"):
            planning_command = f"Input audio_path={state['input_audio_path']}. {planning_command}"

        if input_song_id:
            input_song_metadata = AudioMetadata.get_by_id(input_song_id)
            if not input_song_metadata:
                raise ValueError(f"Song with id '{input_song_id}' was not found.")
            planning_command = (
                f"Song 1 is fixed and must always be treated as the first song. "
                f"song_id={input_song_metadata.id}, song_title={input_song_metadata.title}. "
                f"{planning_command}"
            )

        plan_result = toollist.audio_intent.invoke(
            {
                "raw_state": {
                    "messages": state.get("messages", []),
                },
                "command": planning_command,
            }
        )
        execution_plan = plan_result["execution_plan"]

        if input_song_metadata:
            ordered_song_names = [song for song in execution_plan.song_name if song != input_song_metadata.title]
            ordered_song_names.insert(0, input_song_metadata.title)
            execution_plan = execution_plan.model_copy(
                update={"song_name": ordered_song_names}
            )

        return {
            "input_song_metadata": input_song_metadata,
            "execution_plan": execution_plan,
            "current_step_index": 0,
        }

    def _execute_step(self, state: AgentState):
        execution_plan = state["execution_plan"]
        ordered_steps = sorted(execution_plan.steps, key=lambda item: item.step_no)
        step = ordered_steps[state["current_step_index"]]

        if step.tool == "metadata_retriever":
            result = toollist.metadata_agent.invoke({"raw_state": state})
        elif step.tool == "chunk_retriever":
            result = toollist.retrieval_agent.invoke({"raw_state": state})
        elif step.tool == "voice_modifier":
            result = toollist.voice_modifier_agent.invoke({"raw_state": state})
        elif step.tool == "audio_constructor":
            result = toollist.construct_audio_agent.invoke({"raw_state": state})
        elif step.tool == "similarity_search":
            result = toollist.similarity_search_agent.invoke({"raw_state": state})
        else:
            raise ValueError(f"Unsupported tool '{step.tool}' in execution plan.")

        result["current_step_index"] = state["current_step_index"] + 1
        return result

    def _finalize(self, state: AgentState):
        execution_plan = state["execution_plan"]
        input_song_metadata = state.get("input_song_metadata")
        songs_metadata = state.get("songs_metadata", [])
        primary_song = input_song_metadata or (songs_metadata[0] if songs_metadata else None)

        final_song_metadata = {
            "workflow": execution_plan.workflow,
            "description": execution_plan.description,
            "required_output": execution_plan.required_output,
            "song_id": str(primary_song.id) if primary_song else None,
            "song_name": primary_song.title if primary_song else (execution_plan.song_name[0] if execution_plan.song_name else None),
            "stems": execution_plan.stems,
            "modify_stems": execution_plan.modify_stems,
            "constraints": execution_plan.constraints,
        }

        return {
            "final_song_metadata": final_song_metadata,
            "final_output": {
                "final_song_metadata": final_song_metadata,
                "audio_path": state.get("audio_path"),
                "similarity_query_song_id": state.get("similarity_query_song_id"),
                "similarity_results": state.get("similarity_results", []),
            },
        }

    def _route_after_plan(self, state: AgentState):
        if state["execution_plan"].steps:
            return "execute_step"
        return "finalize"

    def _route_after_step(self, state: AgentState):
        if state["current_step_index"] < len(state["execution_plan"].steps):
            return "execute_step"
        return "finalize"

    def invoke(
        self,
        command: str,
        song_id: Optional[str] = None,
        messages: Optional[list[Any]] = None,
        audio_name: Optional[str] = None,
        query_audio_path: Optional[str] = None,
        input_audio_path: Optional[str] = None,
    ):
        result = self.graph.invoke(
            {
                "command": command,
                "input_song_id": song_id,
                "messages": messages or [],
                "audio_name": audio_name,
                "query_audio_path": query_audio_path,
                "input_audio_path": input_audio_path,
            }
        )
        return result["final_output"]
