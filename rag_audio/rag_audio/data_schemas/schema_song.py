from mongoengine import ListField, ReferenceField, Document, StringField, IntField
from ..data_schemas.schema_stem import stemData
from . import connect_audio_rag

class AudioMetadata(Document):
    id = StringField(primary_key=True)
    title = StringField(required=True)
    artist = StringField(required=True)
    album = StringField(required=True)
    year = IntField(required=True)
    bit_path = StringField(required=True)
    genre = StringField(required=True)
    duration = IntField(required=True)
    sample_rate = IntField(required=True)
    channels = IntField(required=True)
    bit_depth = IntField(required=True)
    codec = StringField(required=True)
    stem_data=ListField(ReferenceField(stemData))
    normalized_path = StringField(required=True)
    normalized_hash = StringField(required=True)

    @classmethod
    def create_document(cls, metadata: dict):
        connect_audio_rag()
        existing = cls.objects(id=metadata["id"]).first()
        if existing:
            return existing
        return cls(**metadata).save()

    @classmethod
    def get_by_id(cls, song_id: str):
        connect_audio_rag()
        return cls.objects(id=song_id).first()

    @classmethod
    def get_by_title(cls, title: str):
        connect_audio_rag()
        return cls.objects(title=title).first()

    @classmethod
    def list_documents(cls):
        connect_audio_rag()
        return list(cls.objects())

    @classmethod
    def add_stem(cls, song_id: str, stem):
        connect_audio_rag()
        return cls.objects(id=song_id).update_one(push__stem_data=stem)
