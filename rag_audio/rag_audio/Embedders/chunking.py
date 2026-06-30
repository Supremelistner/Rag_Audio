from mongoengine import connect,Document,StringField,IntField
import mutagen
from pathlib import Path
from ..data_schemas.schema_stem import stemData
from ..data_schemas.schema_chunks import ChunkMetaData
from ..data_schemas.schema_song import AudioMetadata
import torchaudio
import os
connect(db="metadata_chunks",host="mongodb://localhost:27017/Audio_rag")

class load_chunks:
    def __init__(self,chunk_size,overlap):
        self.chunk_size = chunk_size
        self.overlap = overlap
            
    def create_chunks(self, song_id):
        audio_file = AudioMetadata.objects(id=song_id).first()
        stem_data = audio_file.stem_data

        for stem in stem_data:
            waveform, sr = torchaudio.load(stem.stem_path)

            chunk_samples   = int(self.chunk_size * sr)
            overlap_samples = int(self.overlap * sr)
            hop_samples     = chunk_samples - overlap_samples  
            total_samples   = waveform.size(1)

            chunk_number = 0
            start_sample = 0

            while start_sample + chunk_samples <= total_samples:  
                end_sample    = start_sample + chunk_samples
                chunk_waveform = waveform[:, start_sample:end_sample]

                chunk_path = (
                    Path(__file__).parent.parent.parent
                    / "data" / "chunks"
                    / str(song_id) / str(stem.id)
                    / f"chunk_{chunk_number}.wav"
                )
                os.makedirs(chunk_path.parent, exist_ok=True)
                torchaudio.save(str(chunk_path), chunk_waveform, sr)

                chunk_metadata = ChunkMetaData(
                    song_id     = song_id,
                    stem_id     = str(stem.id),
                    stem_type   = stem.type,
                    chunk_number= chunk_number,
                    start_time  = int(start_sample / sr * 1000),  # ms
                    end_time    = int(end_sample   / sr * 1000),
                    duration    = int(chunk_samples / sr * 1000),
                    sample_rate = sr,
                    chunk_path  = str(chunk_path),
                )
                chunk = chunk_metadata.save()
                stem.chunks.append(chunk)

                start_sample += hop_samples  
                chunk_number += 1

            stem.save() 
a=load_chunks(10,3)
a.create_chunks("6523f1f4cf4ec9880d9f809d9e43e589b75481d28eb52afd3d08312a2c531fff487f2481447d65b9d206f09d732e8962577a5b05f891b9e07ddb02f2f767b971")