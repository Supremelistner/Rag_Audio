from mongoengine import connect,ListField, ReferenceField,EmbeddedDocumentField,Document, StringField, IntField
from ..data_schemas.schema_chunks import ChunkMetaData
class stemData(Document):
    song_id=StringField(required=True)
    stem_path=StringField(required=True)
    type=StringField(required=True)
    duration=IntField(required=True)
    sample_rate=IntField(required=True)
    channels=IntField(required=True)
    chunks=ListField(ReferenceField(ChunkMetaData))