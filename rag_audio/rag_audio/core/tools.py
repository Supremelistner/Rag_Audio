from pathlib import Path
from uuid import uuid4
from typing import Annotated
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


from ..data_schemas.Execplan import ExecutionPlan
from ..data_schemas.schema_chunks import ChunkMetaData
from ..data_schemas.schema_song import AudioMetadata
from ..Ingestion.ingest import Audio_loader
from ..embedders.vector_db import Vector_db

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
        "similarity_search"
]}


pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=512,
    temperature=0.1,
    return_full_text=False
)

Planner = HuggingFacePipeline(pipeline=pipe)



class toollist:

    @tool("intent",description="Responsible for understanding the request.")
    def audio_intent(raw_state: Annotated[dict, InjectedState()],command:str)->ExecutionPlan:
        chat_history = raw_state.get("messages", [])
        history_context = ""
        if chat_history:
            for msg in chat_history:
                role = "User" if msg.type == "human" else "Assistant"
                history_context += f"{role}: {msg.content}\n"
       
        tool_manifest = [
            "metadata_retriever",
            "chunk_retriever",
            "voice_modifier",
            "audio_constructor",
            "similarity_search",
        ]
        
        prompt=f"""You are an execution planner for an Agentic Audio Retrieval and Editing System.
            Your task is to analyze the user's request and generate a valid ExecutionPlan.
            Your responsibilities:
            1. Determine the workflow.
            2. Identify the songs involved.
            3. Determine the stems required.
            4. Determine which stems should be modified.
            5. Determine any additional required inputs.
            6. Generate the execution steps using ONLY the available tools.
            7. Specify the desired output.
            SUPPORTED WORKFLOWS
            - voice_change
            - stem_change
            - mashup
            - similarity_score
            Do NOT invent new workflows.
            AVAILABLE STEMS
            - vocals
            - drums
            - bass
            - guitar
            - piano
            - other
            Never invent new stem names.
            AVAILABLE TOOLS
            1. metadata_retriever
            Resolves song names into internal metadata.
            2. chunk_retriever
            Retrieves required audio chunks from the database.
            3. voice_modifier
            Modifies only the requested vocal stem.
            4. audio_constructor
            Constructs the final audio using retrieved and modified assets.
            5. similarity_search
            Ingests query audio when needed and returns top similar songs from Qdrant.
            Do NOT invent tools.
            WORKFLOW GUIDELINES
            voice_change
            - Requires one song.
            - Requires a voice sample.
            - Retrieve every stem.
            - Modify only vocals.
            - Final tool must be audio_constructor.
            stem_change
            - Requires one song.
            - Requires a replacement audio.
            - Retrieve only the requested stem unless preserving the full song is necessary.
            - Final tool must be audio_constructor.
            mashup
            - Requires two or more songs.
            - Retrieve only the requested stems from each song.
            - Final tool must be audio_constructor.
            similarity_score
            - Requires one query audio.
            - Uses similarity_search.
            - Does NOT use audio_constructor.
            Take help of WORKFLOW_MAP for reference before finalizing

            RULES
            Do not answer the user.
            Do not explain your reasoning.
            Do not generate natural language.
            Do not invent workflows.
            Do not invent tools.
            Do not invent stems.
            If information is missing,
            add it to required_inputs.
            Always return a valid ExecutionPlan.
            Return ONLY the ExecutionPlan and validate it against the ExecutionPlan schema.
            User_Query: {command}\n\nHistory: {history_context}\n\nWORKFLOW_MAP: {WORKFLOW_MAP}\n\nExecutionPlan Schema: {ExecutionPlan.model_json_schema()}\n\nAvailable Tools: {tool_manifest}"""
        
        response = Planner.invoke(prompt)

        # Extract JSON if the model adds extra text
        start = response.find("{")
        end = response.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(f"Model did not return JSON:\n{response}")

        json_text = response[start:end]

        plan = ExecutionPlan.model_validate_json(json_text)

        return {"execution_plan": plan}
        


    @tool("chunk_retriever",description="Responsible for retrieving the required audio chunks from the database.")
    def retrieval_agent(raw_state: Annotated[dict, InjectedState()]):
        songs = raw_state.get("songs_metadata", [])
        plan = raw_state.get("execution_plan")
        required_chunks = {}
        for song in songs:
            required_chunks[str(song.id)] = {}
            for stem in plan.stems:
                chunks = ChunkMetaData.get_for_song_and_stem(str(song.id), stem)
                if not chunks:
                    raise ValueError(f"No chunks found for song '{song.title}' with stem type '{stem}'.")
                required_chunks[str(song.id)][stem] = chunks
        return {"retrieved_chunks": required_chunks}


    @tool("similarity_search", description="Responsible for searching similar songs based on audio features.")
    def similarity_search_agent(raw_state: Annotated[dict, InjectedState()]):
        plan = raw_state.get("execution_plan")
        if not plan:
            raise ValueError("Execution plan is required before similarity search.")

        step = None
        for execution_step in sorted(plan.steps, key=lambda item: item.step_no):
            if execution_step.tool == "similarity_search":
                step = execution_step
                break
        if step is None:
            raise ValueError("No execution step found for tool 'similarity_search'.")

        song_id = (
            raw_state.get("input_song_id")
            or step.parameters.get("song_id")
            or step.parameters.get("query_song_id")
        )
        if not song_id:
            audio_name = (
                raw_state.get("audio_name")
                or step.parameters.get("audio_name")
            )
            query_audio_path = (
                raw_state.get("query_audio_path")
                or raw_state.get("input_audio_path")
                or step.parameters.get("query_audio_path")
                or step.parameters.get("input_audio_path")
            )
            if not audio_name and query_audio_path:
                audio_name = Path(query_audio_path).name
            if not audio_name:
                raise ValueError("Similarity search requires input_song_id, audio_name, or query_audio_path.")
            song_id = Audio_loader().audio_loader(audio_name)

        stems = step.parameters.get("stems", plan.stems or ["vocals"])
        top_k = int(step.parameters.get("top_k", 5))
        results = Vector_db().get_top_songs(song_id=song_id, stems=stems, top_k=top_k)
        return {
            "similarity_query_song_id": song_id,
            "similarity_results": results,
        }



    
    @tool("metadata_retriever", description="Responsible for retrieving metadata for the specified songs.")
    def metadata_agent(raw_state: Annotated[dict, InjectedState()]):
        plan=raw_state["execution_plan"]
        songs=plan.song_name
        audio_data = []
        for song in songs:
            audio = AudioMetadata.get_by_title(song)
            if audio:  # Ensures the song was actually found
                audio_data.append(audio)
        return {"songs_metadata": audio_data}


            
        

    
        
    @tool("audio_constructor", description="Responsible for constructing the final audio using retrieved and modified assets.")
    def construct_audio_agent(raw_state: dict):
        plan = raw_state.get("execution_plan")
        songs = raw_state.get("songs_metadata", [])
        retrieved_chunks = raw_state.get("retrieved_chunks", {})
        modified_stems = raw_state.get("modified_stems", {})

        if not plan:
            raise ValueError("Execution plan is required before constructing audio.")
        if not songs:
            raise ValueError("Song metadata is required before constructing audio.")
        if not retrieved_chunks:
            raise ValueError("Retrieved chunks are required before constructing audio.")

        step = None
        for execution_step in sorted(plan.steps, key=lambda item: item.step_no):
            if execution_step.tool == "audio_constructor":
                step = execution_step
                break
        if step is None:
            raise ValueError("No execution step found for tool 'audio_constructor'.")

        output_dir = Path(step.parameters.get("output_dir", Path("data") / "generated_audio"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # build_assets()
        assets = {}
        replacement_audio_path = (
            step.parameters.get("replacement_audio_path")
            or step.parameters.get("provided_audio_path")
            or raw_state.get("replacement_audio_path")
            or raw_state.get("provided_audio_path")
            or raw_state.get("input_audio_path")
        )
        for song in songs:
            song_id = str(song.id)
            stem_chunks = retrieved_chunks.get(song_id, {})
            if not stem_chunks:
                raise ValueError(f"No retrieved chunks available for song id '{song_id}'.")

            stems_to_use = step.parameters.get("stems", plan.stems)
            for stem_name in stems_to_use:
                modified_entry = modified_stems.get(song_id, {}).get(stem_name)
                if (
                    plan.workflow == "stem_change"
                    and stem_name in plan.modify_stems
                    and replacement_audio_path
                ):
                    stem_waveform, stem_sr = torchaudio.load(str(replacement_audio_path))
                elif modified_entry and modified_entry.get("output_path"):
                    stem_waveform, stem_sr = torchaudio.load(modified_entry["output_path"])
                else:
                    chunks = stem_chunks.get(stem_name, [])
                    if not chunks:
                        raise ValueError(f"No chunks found for stem '{stem_name}' in song id '{song_id}'.")
                    ordered_chunks = sorted(chunks, key=lambda chunk: chunk.chunk_number)
                    audio_segments = []
                    stem_sr = ordered_chunks[0].sample_rate
                    for chunk in ordered_chunks:
                        waveform, chunk_sr = torchaudio.load(chunk.chunk_path)
                        if chunk_sr != stem_sr:
                            waveform = torchaudio.functional.resample(waveform, chunk_sr, stem_sr)
                        audio_segments.append(waveform)
                    stem_waveform = torch.cat(audio_segments, dim=1)

                # Normalize to (channels, samples) explicitly so downstream code
                # never has to guess dimensionality.
                if stem_waveform.dim() == 1:
                    stem_waveform = stem_waveform.unsqueeze(0)

                if stem_name not in assets:
                    assets[stem_name] = []
                assets[stem_name].append(
                    {
                        "song_id": song_id,
                        "song_title": song.title,
                        "stem_name": stem_name,
                        "waveform": stem_waveform,
                        "sample_rate": stem_sr,
                    }
                )

        if not assets:
            raise ValueError("No song audio was assembled from the execution plan.")

        # execute_operations()
        for stem_name, stem_assets in assets.items():
            for asset in stem_assets:
                operation_parameters = {}
                operation_parameters.update(step.parameters)
                operation_parameters.update(step.parameters.get("stem_operations", {}).get(stem_name, {}))
                operation_parameters.update(step.parameters.get("song_operations", {}).get(asset["song_id"], {}))
                operation_parameters.update(step.parameters.get("song_operations", {}).get(asset["song_title"], {}))

                # Keep as (channels, samples) — no squeeze(0)/unsqueeze(0) round-trip.
                audio = asset["waveform"].detach().cpu().numpy()

                if operation_parameters.get("pitch_shift") is not None:
                    audio = librosa.effects.pitch_shift(
                        audio,
                        sr=asset["sample_rate"],
                        n_steps=float(operation_parameters["pitch_shift"]),
                    )
                if operation_parameters.get("time_stretch") not in (None, 1, 1.0):
                    audio = librosa.effects.time_stretch(
                        audio,
                        rate=float(operation_parameters["time_stretch"]),
                    )
                if operation_parameters.get("gain_db") is not None:
                    audio = audio * (10 ** (float(operation_parameters["gain_db"]) / 20.0))

                asset["waveform"] = torch.tensor(audio, dtype=asset["waveform"].dtype)

        # align_tracks()
        final_sample_rate = None
        aligned_stems = {}
        for stem_name, stem_assets in assets.items():
            max_length = 0
            aligned_assets = []
            for asset in stem_assets:
                if final_sample_rate is None:
                    final_sample_rate = asset["sample_rate"]
                if asset["sample_rate"] != final_sample_rate:
                    asset["waveform"] = torchaudio.functional.resample(
                        asset["waveform"],
                        asset["sample_rate"],
                        final_sample_rate,
                    )
                    asset["sample_rate"] = final_sample_rate
                if asset["waveform"].shape[1] > max_length:
                    max_length = asset["waveform"].shape[1]
                aligned_assets.append(asset)
            if not aligned_assets:
                raise ValueError("No aligned stems were available for rendering.")
            for asset in aligned_assets:
                if asset["waveform"].shape[1] < max_length:
                    asset["waveform"] = torch.nn.functional.pad(
                        asset["waveform"],
                        (0, max_length - asset["waveform"].shape[1]),
                    )
            aligned_stems[stem_name] = aligned_assets

        # render()
        rendered_stems = []
        song_weights = step.parameters.get("song_weights", {})
        stem_weights = step.parameters.get("stem_weights", {})
        for stem_name, stem_assets in aligned_stems.items():
            stem_render = torch.zeros_like(stem_assets[0]["waveform"])
            total_weight = 0.0
            for asset in stem_assets:
                weight = song_weights.get(asset["song_id"], song_weights.get(asset["song_title"], 1.0))
                stem_render = stem_render + (asset["waveform"] * float(weight))
                total_weight += float(weight)
            if total_weight > 0:
                stem_render = stem_render / total_weight
            stem_render = stem_render * float(stem_weights.get(stem_name, 1.0))
            rendered_stems.append(stem_render)

        if not rendered_stems:
            raise ValueError("No rendered stems were available for export.")

        target_length = max(tensor.shape[1] for tensor in rendered_stems)
        merged = torch.zeros_like(
            rendered_stems[0]
            if rendered_stems[0].shape[1] == target_length
            else torch.nn.functional.pad(rendered_stems[0], (0, target_length - rendered_stems[0].shape[1]))
        )
        for tensor in rendered_stems:
            if tensor.shape[1] < target_length:
                tensor = torch.nn.functional.pad(tensor, (0, target_length - tensor.shape[1]))
            merged = merged + tensor

        peak = merged.abs().max()
        if peak > 1:
            merged = merged / peak

        # export()
        # merged is (channels, samples) at this point — normalize any stray
        # extra dims, then transpose to (samples, channels) for soundfile.
        out = np.squeeze(merged.detach().cpu().numpy())
        if out.ndim == 1:
            pass  # mono, write as-is
        elif out.ndim == 2:
            out = out.T  # (channels, samples) -> (samples, channels)
        else:
            raise ValueError(f"Unexpected audio shape before export: {out.shape}")

        output_path = output_dir / f"{plan.workflow}_{uuid4().hex}.wav"
        sf.write(str(output_path), out, final_sample_rate or 44100)
        return {"audio_path": str(output_path)}
    
    
        
    @tool("voice_modifier", description="Responsible for modifying the vocal stem of a song based on specified parameters.")
    def voice_modifier_agent(raw_state: dict):
        plan = raw_state.get("execution_plan")
        retrieved_chunks = raw_state.get("retrieved_chunks", {})

        if not plan:
            raise ValueError("Execution plan is required before modifying vocals.")
        if not retrieved_chunks:
            raise ValueError("Retrieved chunks are required before modifying vocals.")

        if plan.workflow != "voice_change":
            return {"audio_path": None}

        if "vocals" not in plan.modify_stems:
            raise ValueError("Voice change workflow must include 'vocals' in modify_stems.")

        step = None
        for execution_step in sorted(plan.steps, key=lambda item: item.step_no):
            if execution_step.tool == "voice_modifier":
                step = execution_step
                break
        if step is None:
            raise ValueError("No execution step found for tool 'voice_modifier'.")

        output_dir = Path(step.parameters.get("output_dir", Path("data") / "modified_stems"))
        output_dir.mkdir(parents=True, exist_ok=True)

        all_modified = {}

        for song_id, stems in retrieved_chunks.items():
            vocal_chunks = stems.get("vocals")
            if not vocal_chunks:
                raise ValueError(f"No vocal chunks found for song id '{song_id}'.")

            # build_vocal_asset(...)
            ordered_chunks = sorted(vocal_chunks, key=lambda chunk: chunk.chunk_number)
            audio_segments = []
            sample_rate = ordered_chunks[0].sample_rate
            for chunk in ordered_chunks:
                waveform, chunk_sr = torchaudio.load(chunk.chunk_path)
                if chunk_sr != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, chunk_sr, sample_rate)
                audio_segments.append(waveform)
            vocal_asset = torch.cat(audio_segments, dim=1)  # (channels, samples)

            # modify(...)
            operation_parameters = {}
            operation_parameters.update(step.parameters)
            operation_parameters.update(step.parameters.get("song_operations", {}).get(song_id, {}))

            modified_audio = vocal_asset.detach().cpu().numpy()  # keep (channels, samples)

            if operation_parameters.get("pitch_shift") is not None:
                modified_audio = librosa.effects.pitch_shift(
                    modified_audio,
                    sr=sample_rate,
                    n_steps=float(operation_parameters["pitch_shift"]),
                )
            if operation_parameters.get("time_stretch") not in (None, 1, 1.0):
                modified_audio = librosa.effects.time_stretch(
                    modified_audio,
                    rate=float(operation_parameters["time_stretch"]),
                )
            if operation_parameters.get("gain_db") is not None:
                modified_audio = modified_audio * (10 ** (float(operation_parameters["gain_db"]) / 20.0))

            modified = torch.tensor(modified_audio, dtype=vocal_asset.dtype)  # still (channels, samples)
            peak = modified.abs().max()
            if peak > 1:
                modified = modified / peak

            # export() — normalize shape right before writing
            out = np.squeeze(modified.detach().cpu().numpy())
            if out.ndim == 1:
                pass  # mono
            elif out.ndim == 2:
                out = out.T  # (channels, samples) -> (samples, channels)
            else:
                raise ValueError(f"Unexpected audio shape before export: {out.shape}")

            output_path = output_dir / f"{song_id}_vocals_{uuid4().hex}.wav"
            sf.write(str(output_path), out, sample_rate)

            all_modified[song_id] = {
                "vocals": {
                    "output_path": str(output_path),
                    "sample_rate": sample_rate,
                }
            }

        return {"modified_stems": all_modified}