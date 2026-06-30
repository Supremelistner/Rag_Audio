from mongoengine import connect

MONGO_URI = "mongodb://localhost:27017/Audio_rag"
MONGO_DB = "Audio_rag"
_CONNECTED = False


def connect_audio_rag():
    global _CONNECTED
    if not _CONNECTED:
        connect(db=MONGO_DB, host=MONGO_URI, alias="default")
        _CONNECTED = True
