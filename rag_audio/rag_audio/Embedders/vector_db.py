from qdrant_client import QdrantClient
from ..data_schemas.schema_chunks import ChunkMetaData
from ..data_schemas.schema_song import AudioMetadata
from ..data_schemas.schema_stem import stemData
from .embeds import embeds
import uuid



wavdim=768

client = QdrantClient(url="http://localhost:6333")
COLLECTIONS:list[str]=["vocals","drums","bass","piano","guitar","other"]

class Vector_db:
    def __init__(self):
        self.embedder=embeds()
        existing = {collection.name
                    for collection in client.get_collections().collections}
        for c in COLLECTIONS:
            if c not in existing:
                client.create_collection(
                    collection_name=c,
                    vectors_config={
                        "size": wavdim,
                        "distance": "Cosine"
                    }
                )
    def insert_song(self,song_id):
        song=AudioMetadata.get_by_id(song_id)
        for stem in song.stem_data:
            stem_data:stemData=stem
            chunk:ChunkMetaData
            for chunk in stem_data.chunks:
                point_id=str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_DNS,
                                    f"{song_id}:{stem_data.type}:{chunk.chunk_number}"
                                )
                            )
            
                client.upsert(
                    collection_name=stem_data.type,
                    points=[
                        {
                            "id": point_id,
                            "vector": self.embedder.embed(chunk.chunk_path, stem_data.type),
                            "payload": {
                                        "song_id":song_id,
                                        "stem_id":chunk.stem_id,
                                        "stem_type":chunk.stem_type,
                                        "chunk_id":str(chunk.id),
                                        "chunk_number":chunk.chunk_number,
                                        "duration":chunk.duration,
                                        "start_time":chunk.start_time,
                                        "end_time":chunk.end_time
                            }
                        }
                    ]
                )

    
    def query(self, query_audio, stem_type, top_k=5):
        results = client.query_points(
            collection_name=stem_type,
            query=self.embedder.embed(query_audio, stem_type),
            limit=top_k,
        )
        return results.points

    def get_top_songs(self, song_id, stems=None, top_k=5):
        song = AudioMetadata.get_by_id(song_id)
        if not song:
            raise ValueError(f"Song with id '{song_id}' was not found.")

        stems = stems or COLLECTIONS
        scores = {}
        for stem_type in stems:
            results = self.query(song.normalized_path, stem_type, top_k=top_k)
            for result in results:
                matched_song_id = result.payload.get("song_id")
                if not matched_song_id or matched_song_id == song_id:
                    continue
                if matched_song_id not in scores:
                    scores[matched_song_id] = {
                        "song_id": matched_song_id,
                        "score": 0.0,
                        "matches": 0,
                        "stems": set(),
                    }
                scores[matched_song_id]["score"] += float(result.score)
                scores[matched_song_id]["matches"] += 1
                scores[matched_song_id]["stems"].add(stem_type)

        ranked = sorted(
            scores.values(),
            key=lambda item: item["score"] / item["matches"],
            reverse=True,
        )[:top_k]
        for item in ranked:
            matched_song = AudioMetadata.get_by_id(item["song_id"])
            item["score"] = item["score"] / item["matches"]
            item["stems"] = sorted(item["stems"])
            item["metadata"] = {
                "title": matched_song.title,
                "artist": matched_song.artist,
                "album": matched_song.album,
                "year": matched_song.year,
                "normalized_path": matched_song.normalized_path,
            } if matched_song else None
        return ranked
print(client.count(
    collection_name="vocals",
    exact=True
))