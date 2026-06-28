from mongoengine import connect,Document,StringField,IntField
import subprocess
from pathlib import Path


connect(db="metadata_audio", host="mongodb://localhost:27017/Audio_rag")


class stemData(Document):
    id=StringField(primary_key=True)
    song_id=StringField(required=True)
    stem_path=StringField(required=True)
    type=StringField(required=True)
    duration=IntField(required=True)
    sample_rate=IntField(required=True)
    channels=IntField(required=True)
    hash=StringField(required=True)

class stem_loader:
    def separate_audio(self,input_path,stems=None,mp3=False):
        out_path = Path(__file__).parent.parent.parent/"data"/"stem_data"/input_path.name+".wav"
        command = ["demucs", "-o", out_path]
        
        if mp3:
            command.append("--wav")
            
        # Add specific stem isolation if requested (e.g., 'vocals')
        if stems:
            command.append(f"--two-stems={stems}")
            
        # Append the target file
        command.append(input_file)
        
        print(f"Running command: {' '.join(command)}")
        
        # Run Demucs and stream the output to your python console
        result = subprocess.run(command, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("Separation completed successfully!")
        else:
            print("An error occurred during separation.")

    # Example Usage
    separate_audio("my_song.mp3", stems="vocals")
