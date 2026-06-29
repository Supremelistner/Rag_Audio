from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from ..data_schemas.Execplan import ExecutionPlan,stem_plan
from typing import Annotated,dict
from langchain_huggingface import HuggingFacePipeline

WORKFLOWS = [
    "similarity_search",
    "voice_change",
    "stem_change",
    "mashup"
]
WORKFLOW_MAP = {
    "voice_change":[
        "metadata_retriever",
        "chunk_retriever",
        "voice_modifier",
        "audio_constructor"    ],
    "stem_change":[
        "metadata_retriever",
        "chunk_retriever",
        "audio_constructor"    ],
    "mashup":[
        "metadata_retriever",
        "chunk_retriever",
        "audio_constructor"
    ],
    "similarity_score":[
        "metadata_retriever",
        "chunk_retriever"
]}


Planner=HuggingFacePipeline(repo_id="google/gemma-2-2b",task="text-generation")
class toollist:

    @tool("intent",description="Responsible for understanding the request.")
    def audio_intent(raw_state: Annotated[dict, InjectedState()],command:str)->ExecutionPlan:
        chat_history = raw_state.get("messages", [])
        history_context = ""
        if chat_history:
            for msg in chat_history:
                role = "User" if msg.type == "human" else "Assistant"
                history_context += f"{role}: {msg.content}\n"
       
        tool_manifest = ["audio_intent","stem_selector","retrieve_chunks","retrieve metadata","reconstruct_audio","voice_conversion"]
        
        plan=ExecutionPlan
        prompt=f"""You are an execution planner for an Audio Retrieval Agent.
                Your task is ONLY to create an execution plan.
                define the required inputs and check it 
                against available stems and tools.
                define required output of whole process
                start with step_no = 0
                define the task and describe it in detail 
                Do NOT answer the user.
                Only select tasks from available tasks. 
                Do NOT generate new tasks.
                Do NOT explain.
                Use ONLY the available tools.
                Do NOT generate steps for tools that are not available.
                Do NOT invent new tools.
                Return ONLY an ExecutionPlan.
                If information is missing,
                add it to required_inputs.
                Never invent tools.
                Never invent stems.
                Use Worflow_map provided to plan the tasks.
                Never add unnecessary tasks which are not referred in Workflow_map
                Available stems:
                - vocals
                - guitar
                - bass
                - drums
                - piano
                - other
            History:\n{history_context}\n\nAvailable Tools:\n{tool_manifest}\n\nCommand: {command}\n\ntasks:{WORKFLOWS}\n\n"""
        model=Planner.with_structured_output(plan)
        response=model.invoke(prompt)
        return response
        


    @tool("chunk_retriever")
    def retrival_agent():
        pass

    
    @tool("metadata_retreiver")
    def metadata_agent(command:str):
        songs=" ".split(command)
        for song in songs:
            
        

    
    
    @tool("construct_audio")
    def construct():
        pass

    
    
    
    @tool("vioce_modifier")
    def vconverter():
        pass