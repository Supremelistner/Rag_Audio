from mongoengine import connect, EmbeddedDocumentField,Document, StringField, IntField

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