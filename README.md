# RAG Audio(Use Branch master for more stable and latest version)
### Agentic Retrieval-Augmented Audio Editing and Reconstruction Framework

> A modular AI system that ingests songs, decomposes them into semantic audio assets, indexes them into a vector database, and reconstructs new audio compositions through natural language instructions.

---

## Overview

RAG Audio is an experimental framework for intelligent audio editing using Retrieval-Augmented Generation (RAG) principles.

Instead of treating a song as one continuous waveform, the system converts every song into reusable audio assets consisting of:

- Individual stems
- Temporal audio chunks
- Semantic embeddings
- Metadata
- Vector representations

These assets are later retrieved and recombined through an AI planning agent capable of executing complex audio editing workflows.

The goal is to transform songs into reusable building blocks rather than static recordings.

---

# Features

- Audio ingestion pipeline
- Automatic audio normalization
- Six-stem source separation using Demucs
- Temporal chunk generation with overlap
- Audio embedding generation
- Qdrant vector search
- MongoDB metadata storage
- Agentic workflow planning
- Natural language audio editing
- Modular audio reconstruction engine

Supported workflows include:

- Voice modification
- Stem replacement
- Mashups
- Similarity search
- Audio reconstruction

---

# System Architecture

```
                User Audio
                     │
                     ▼
             Audio Ingestion
                     │
                     ▼
             Audio Normalization
                     │
                     ▼
             Stem Separation
         (Vocals, Piano, Bass...)
                     │
                     ▼
             Chunk Generation
                     │
                     ▼
             Audio Embeddings
                     │
                     ▼
            Qdrant Vector Store
                     │
             MongoDB Metadata
                     │
                     ▼
             AI Planning Agent
                     │
                     ▼
        Asset Retrieval & Construction
                     │
                     ▼
             Generated Audio
```

---

# Project Structure

```
rag_audio/

│
├── app.py
│
├── Ingestion/
│   └── ingest.py
│
├── embedders/
│   ├── embeds.py
│   ├── chunking.py
│   ├── stem.py
│   └── vector_db.py
│
├── core/
│   ├── agent.py
│   └── tools.py
│
├── data_schemas/
│   ├── schema_song.py
│   ├── schema_stem.py
│   ├── schema_chunks.py
│   └── ExecPlan.py
│
└── data/
    ├── input_data/
    ├── processed/
    ├── normalized/
    ├── stem_data/
    ├── chunks/
    └── generated_audio/
```

---

# Workflow

## 1. Audio Ingestion

Responsible for validating and preparing uploaded audio.

Operations

- Audio validation
- WAV conversion
- Sample rate normalization
- Loudness normalization
- Metadata extraction
- SHA hashing

Output

- Processed WAV
- Normalized WAV
- MongoDB Song Metadata

---

## 2. Stem Separation

Uses Facebook Research's Demucs model.

Generated stems

- Vocals
- Drums
- Bass
- Guitar
- Piano
- Other

Output

- Six independent audio tracks
- Stem metadata

---

## 3. Chunk Generation

Each stem is divided into overlapping temporal chunks.

Purpose

- Fine-grained retrieval
- Partial reconstruction
- Semantic indexing

Output

```
Song
 ├── Vocals
 │      ├── chunk_001.wav
 │      ├── chunk_002.wav
 │      └── ...
 │
 ├── Piano
 └── ...
```

---

## 4. Audio Embeddings

Each chunk is embedded using pretrained transformer models.

Current models

- WavLM
- MERT

Output

768-dimensional semantic vectors

---

## 5. Vector Database

Backend

- Qdrant

Collections

```
vocals
drums
bass
guitar
piano
other
```

Each point stores

- Embedding
- Song ID
- Stem ID
- Chunk ID
- Timing metadata

---

## 6. Metadata Database

Backend

MongoDB

Collections

- Songs
- Stems
- Chunks

Stores

- Audio metadata
- File locations
- Relationships
- Chunk references

---

## 7. Agentic Planning

Natural language requests are converted into structured execution plans.

Example

```
Replace the piano from Song A
with the piano from Song B.
```

↓

Execution Plan

```
Retrieve Song A
Retrieve Song B
Extract Piano Stem
Replace Piano
Reconstruct Audio
```

---

## 8. Audio Constructor

Unlike traditional mixers, the constructor operates on reusable assets.

Capabilities

- Stem replacement
- Stem blending
- Melody fusion
- Weighted reconstruction
- Voice modification
- Mashups

This module is designed as an extensible audio constructor rather than a fixed audio mixer.

---

# Streamlit Interface

The application currently provides three primary endpoints.

## Ingest Song

Uploads a song into the retrieval system.

Pipeline

```
Upload

↓

Normalize

↓

Separate Stems

↓

Chunk

↓

Embed

↓

Store
```

---

## Stored Songs

Displays indexed songs currently available.

Shows

- Title
- Artist
- Duration
- Song ID

---

## Chatbot Query

Natural language interface.

Example requests

```
Find songs similar to this recording.

Replace the piano with a violin recording.

Mix the vocals of Song A with the melody of Song B.

Increase vocal pitch.

Generate a mashup using only piano and bass.
```

---

# Technology Stack

### Backend

- Python
- Streamlit
- MongoEngine
- MongoDB

### AI

- LangGraph
- LangChain
- HuggingFace Transformers
- WavLM
- MERT

### Audio

- FFmpeg
- Torchaudio
- Librosa
- Demucs

### Vector Database

- Qdrant

---

# Future Improvements

- Diffusion-based audio generation
- Intelligent stem alignment
- Beat synchronization
- BPM detection
- Key detection
- Semantic music editing
- Multi-track timeline editing
- Plugin architecture
- Real-time reconstruction

---

# Outcomes

This project demonstrates:

- Retrieval-Augmented Generation for audio
- Agentic workflow planning
- Audio source separation
- Vector similarity search
- Semantic audio retrieval
- AI-driven reconstruction
- Modular audio processing pipelines
- Production-oriented backend architecture

---

# Copyright & Attribution

© 2026 Manish Rana

This project was developed by **Manish Rana** as part of ongoing research and development in Retrieval-Augmented Audio Processing and Agentic AI systems.

You are welcome to:

- Learn from the code
- Use portions of the implementation in personal or academic projects
- Modify and extend the project
- Reference it in your own work

If you use substantial parts of this project or build upon it, please provide appropriate credit by linking back to this repository or mentioning the original author.

Contributions, suggestions, and improvements are always welcome.
