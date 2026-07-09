from mongoengine import Document, StringField, IntField
from . import connect_audio_rag

class ChunkMetaData(Document):
    song_id=StringField(required=True)
    stem_id=StringField(required=True)
    stem_type=StringField(required=True)
    chunk_number=IntField(required=True)
    start_time=IntField(required=True)
    end_time=IntField(required=True)
    duration=IntField(required=True)
    sample_rate=IntField(required=True)
    chunk_path=StringField(required=True)

    @classmethod
    def create_document(cls, metadata: dict):
        connect_audio_rag()
        return cls(**metadata).save()

    @classmethod
    def get_by_id(cls, chunk_id: str):
        connect_audio_rag()
        return cls.objects(id=chunk_id).first()

    @classmethod
    def get_for_song_and_stem(cls, song_id: str, stem_type: str):
        connect_audio_rag()
        return list(
            cls.objects(
                song_id=song_id,
                stem_type=stem_type,
            ).order_by("chunk_number")
        )
