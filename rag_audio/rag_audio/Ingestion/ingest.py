from pathlib import Path
import magic
import uuid
import mutagen
import hashlib
import subprocess
from mongoengine import connect, Document, StringField, IntField

connect(db="metadata_audio", host="mongodb://localhost:27017/Audio_rag")
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
    normalized_path = StringField(required=True)
    normalized_hash = StringField(required=True)

class Audio_loader:


    def get_audio_path(self,audio_name):
        return Path(__file__).parent.parent.parent/"data"/"input_data"/audio_name

    def load_audio(self,audio_name):
        """Verifies the file is audio using python-magic, then converts it to bits."""
        path = self.get_audio_path(audio_name)

        # 1. Verify file existence
        if not path.is_file():
            raise FileNotFoundError(f"No file found at {path}")

        # 2. Check the MIME type using python-magic
        mime_type = magic.from_file(str(path), mime=True)
        if not mime_type.startswith("audio/"):
            raise ValueError(f"File is not a valid audio format. Detected: {mime_type}")

    def decoder(self, path):
        # 3. Read and convert to bits
        id_ = str(uuid.uuid4())
        bit_path=Path(__file__).parent.parent.parent/"data"/"processed"/f"{id_}.wav"
        bit_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",                  
            "-i", str(path),       # Input file path
            "-vn",                  # Block video streams entirely
            "-acodec", "pcm_s16le", # Forces 16-bit audio depth
            "-ar", "16000",         # Forces 16 kHz sampling rate
            "-ac", "1",             
            str(bit_path)            
        ]
        subprocess.run(command,check=True,capture_output=True)
        return bit_path

    def get_hash(self, file_path):
        """Generates a SHA256 hash for the given file."""
        hasher = hashlib.sha512()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                hasher.update(byte_block)
        return hasher.hexdigest()      

    def normalise_audio(self, bit_path): 
        """Normalizes the audio file to a standard volume level."""
        normalized_path = bit_path.parent.parent/"normalized" / f"normalized_{bit_path.name}"
        command = [
            "ffmpeg",
            "-y",                  
            "-i", str(bit_path),   
            "-filter:a", "loudnorm", 
            str(normalized_path)    
        ]
        subprocess.run(command)
        return normalized_path
    
    def save_metadata(self,bit_path,audio_path,hss,normalized_path ,nhss):
        audio_file = mutagen.File(str(audio_path), easy=True)
        try:
            year=int(audio_file.get("date", ["Unknown"])[0])
        except ValueError:
            year= 0 
        # Example metadata (replace with actual metadata extraction logic)
        metadata = {
            "id":str(hss),
            "title": audio_file.get("title", ["Unknown"])[0],
            "artist": audio_file.get("artist", ["Unknown"])[0],
            "album": audio_file.get("album", ["Unknown"])[0],
            "year": year,
            "bit_path": str(bit_path),
            "genre": audio_file.get("genre", ["Unknown"])[0],
            "duration": int(audio_file.info.length),
            "sample_rate": audio_file.info.sample_rate,
            "channels": audio_file.info.channels,
            "bit_depth": 16,  # Assuming 16-bit depth after conversion
            "codec": audio_file.mime[0] if audio_file.mime else "Unknown",
            "normalized_path": str(normalized_path),
            "normalized_hash": str(nhss)}
        audio_metadata = AudioMetadata(**metadata)
        audio_metadata.save()

    def audio_loader(self, audio_name):
        path = self.get_audio_path(audio_name)
        self.load_audio(audio_name)
        bit_path = self.decoder(path)
        hss=self.get_hash(bit_path)
        normalized_path = self.normalise_audio(bit_path)
        nhss=self.get_hash(normalized_path)
        self.save_metadata(bit_path,path,hss,normalized_path,nhss)
    
def main():
    audio_name = "test.mp3"  # Replace with your actual audio file name
    loader = Audio_loader()
    loader.audio_loader(audio_name)
    print(f"Audio metadata for {audio_name} has been saved to the database.")
if __name__ == "__main__":
    main()