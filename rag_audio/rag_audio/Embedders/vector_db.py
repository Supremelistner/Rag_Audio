from qdrant_client import QdrantClient
from ..data_schemas.schema_chunks import ChunkMetaData
from ..data_schemas.schema_song import AudioMetadata
from ..data_schemas.schema_stem import stemData
from .embeds import embeds
from mongoengine import connect
import uuid

connect(db="metadata_stem", host="mongodb://localhost:27017/Audio_rag")



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
        song=AudioMetadata.objects(id=song_id).first()
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
        results = client.search(
            collection_name=stem_type,
            query_vector=self.embedder.embed(query_audio, stem_type),
            limit=top_k,
        )
        return results

       
collection_info = client.get_collection(collection_name="vocals")
print(f"Total entries: {collection_info.points_count}")