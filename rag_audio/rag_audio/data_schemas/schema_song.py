from mongoengine import ListField,EmbeddedDocument,ReferenceField,EmbeddedDocumentField,Document, StringField, IntField
from ..data_schemas.schema_stem import stemData

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
