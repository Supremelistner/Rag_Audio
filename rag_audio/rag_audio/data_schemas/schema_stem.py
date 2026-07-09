from mongoengine import ListField, ReferenceField, Document, StringField, IntField
from ..data_schemas.schema_chunks import ChunkMetaData
from . import connect_audio_rag


class stemData(Document):
    song_id=StringField(required=True)
    stem_path=StringField(required=True)
    type=StringField(required=True)
    duration=IntField(required=True)
    sample_rate=IntField(required=True)
    channels=IntField(required=True)
    chunks=ListField(ReferenceField(ChunkMetaData))

    @classmethod
    def create_document(cls, metadata: dict):
        connect_audio_rag()
        return cls(**metadata).save()

    @classmethod
    def get_by_id(cls, stem_id: str):
        connect_audio_rag()
        return cls.objects(id=stem_id).first()

    @classmethod
    def get_for_song(cls, song_id: str):
        connect_audio_rag()
        return list(cls.objects(song_id=song_id))

    @classmethod
    def append_chunk(cls, stem, chunk):
        connect_audio_rag()
        stem.chunks.append(chunk)
        stem.save()
        return stem
