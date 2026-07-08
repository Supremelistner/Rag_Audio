from pathlib import Path
import importlib.util
import sys
import types

import streamlit as st

from rag_audio.Ingestion.ingest import Audio_loader
from rag_audio.data_schemas.schema_song import AudioMetadata
from rag_audio.embedders.chunking import load_chunks
from rag_audio.embedders.stem import stem_loader
from rag_audio.embedders.vector_db import Vector_db
from rag_audio.core.agent import AudioAgent

ROOT_DIR = Path(__file__).parent
INPUT_DIR = ROOT_DIR / "data" / "input_data"
GENERATED_DIR = ROOT_DIR / "data" / "generated_audio"





@st.cache_resource
def get_agent():
    return AudioAgent()


def save_uploaded_audio(uploaded_file):
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    file_path = INPUT_DIR / safe_name
    file_path.write_bytes(uploaded_file.getbuffer())
    return safe_name, file_path


def ingest_song(audio_name, chunk_size=10, overlap=2):
    loader = Audio_loader()
    song_id = loader.audio_loader(audio_name)

    stemmer = stem_loader()
    stem_path = stemmer.separate_audio_6stems(song_id)
    stemmer.create_metadata(Path(stem_path), song_id)

    chunker = load_chunks(chunk_size=chunk_size, overlap=overlap)
    chunker.create_chunks(song_id)

    Vector_db().insert_song(song_id)
    return song_id


def render_song_table():
    songs = AudioMetadata.list_documents()
    rows = [
        {
            "song_id": str(song.id),
            "name": song.title,
            "duration": song.duration,
            "artist": song.artist,
        }
        for song in songs
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No songs are currently stored.")


def render_agent_output(result):
    similarity_results = result.get("similarity_results") or []
    audio_path = result.get("audio_path")

    if similarity_results:
        rows = []
        for item in similarity_results:
            metadata = item.get("metadata") or {}
            rows.append(
                {
                    "song_id": item.get("song_id"),
                    "title": metadata.get("title"),
                    "artist": metadata.get("artist"),
                    "score": item.get("score"),
                    "matched_stems": ", ".join(item.get("stems", [])),
                    "matches": item.get("matches"),
                }
            )
        st.subheader("Top matches")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if audio_path:
        path = Path(audio_path)
        st.subheader("Generated song")
        st.write(str(path))
        if path.exists():
            st.audio(str(path))
            st.download_button(
                "Download generated song",
                data=path.read_bytes(),
                file_name=path.name,
                mime="audio/wav",
            )
        else:
            st.warning("The agent returned an audio path, but the file was not found.")

    if not similarity_results and not audio_path:
        st.info("The agent completed, but no song path or similarity results were returned.")


st.set_page_config(page_title="RAG Audio", layout="wide")
st.title("RAG Audio")

section = st.sidebar.radio(
    "Navigate",
    [
        "Ingest song",
        "Stored songs",
        "Chatbot query",
    ],
)

if section == "Ingest song":
    st.header("Ingest song")
    uploaded_song = st.file_uploader("Upload audio", type=["mp3", "wav", "flac", "ogg", "m4a"])
    chunk_size = st.number_input("Chunk size seconds", min_value=1, max_value=60, value=10)
    overlap = st.number_input("Overlap seconds", min_value=0, max_value=30, value=2)

    if st.button("Ingest", disabled=uploaded_song is None):
        try:
            audio_name, _ = save_uploaded_audio(uploaded_song)
            with st.status("Ingesting song", expanded=True) as status:
                st.write("Saving metadata")
                st.write("Separating stems")
                st.write("Creating chunks")
                st.write("Embedding chunks into Qdrant")
                song_id = ingest_song(audio_name, chunk_size=chunk_size, overlap=overlap)
                status.update(label="Ingestion complete", state="complete")
            st.success(f"song_id: {song_id}")
        except Exception as exc:
            st.error(str(exc))

elif section == "Stored songs":
    st.header("Stored songs")
    if st.button("Refresh"):
        st.rerun()
    render_song_table()

else:
    st.header("Chatbot query")
    query = st.text_area("Query", placeholder="Find songs similar to this audio or create a mashup...")

    with st.sidebar:
        st.subheader("Query audio")
        input_song_id = st.text_input("Existing song id")
        uploaded_query_song = st.file_uploader(
            "Upload song for this query",
            type=["mp3", "wav", "flac", "ogg", "m4a"],
            key="query_song",
        )
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    if st.button("Run query", disabled=not query.strip()):
        try:
            query_song_id = input_song_id.strip() or None
            if uploaded_query_song is not None:
                audio_name, _ = save_uploaded_audio(uploaded_query_song)
                with st.status("Preparing uploaded query song", expanded=True) as status:
                    st.write("Ingesting and indexing uploaded audio")
                    query_song_id = ingest_song(audio_name)
                    status.update(label="Uploaded query song is ready", state="complete")
                st.success(f"query song_id: {query_song_id}")

            agent = get_agent()
            result = agent.invoke(
                command=f"{query.strip()} top_k={top_k}",
                song_id=query_song_id,
            )
            st.subheader("Final metadata")
            st.json(result.get("final_song_metadata", {}))
            render_agent_output(result)
        except Exception as exc:
            st.error(str(exc))
