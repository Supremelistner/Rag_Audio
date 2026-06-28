from mongoengine import connect,Document,StringField,IntField
import subprocess, os
import mutagen
from pathlib import Path
from ..data_schemas.schema_stem import stemData
from ..data_schemas.schema_song import AudioMetadata



connect(db="metadata_stem", host="mongodb://localhost:27017/Audio_rag")




class stem_loader:


    def separate_audio_6stems(self, song_id:str):
        """
        Separates a .wav file into 6 stems (vocals, drums, bass, guitar, piano, other)
        using the htdemucs_6s model.
        """
        # Convert paths to string and ensure the file exists
        input_wav=AudioMetadata.objects(id=song_id).first().normalized_path
        input_wav_path = input_wav
        output_dir = Path(__file__).parent.parent.parent/"data"/"stem_data"
        
        if not os.path.exists(input_wav_path):
            raise FileNotFoundError(f"The file {input_wav_path} could not be found.")


       # Build the Demucs command line instruction
        # -n: specifies the 6-stem model
        # -o: specifies output folder destination
        # Change "cpu" to "cuda" if you have a configured NVIDIA GPU
        command = [
            "demucs",
            "-n", "htdemucs_6s", 
            "-o", str(output_dir),
            "-d", "cpu", 
            input_wav_path
        ]

        try:
            # Execute the process and capture console output in real-time
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True
            )
            
            for line in process.stdout:
                print(line, end="")   
            process.wait()
            
            if process.returncode == 0:
                track_name = Path(input_wav_path).stem
                final_output_path = os.path.join(output_dir, "htdemucs_6s", track_name)
                print(f"\n Separation complete! Your 6 stems are saved in:\n-> {final_output_path}")
            else:
                print(f"\nDemucs failed with exit code: {process.returncode}")

        except Exception as e:
            print(f"An error occurred while running Demucs: {e}")
        return output_dir

    def create_metadata(self,stem_path:Path,song_id):
        for files in stem_path.iterdir():
            if files.is_file():
                audio_file = mutagen.File(str(files), easy=True)
                metadata = {
                    "song_id": song_id,
                    "stem_path": str(files),
                    "type": files.stem,
                    "duration": int(audio_file.info.length),
                    "sample_rate": audio_file.info.sample_rate,
                    "channels": audio_file.info.channels,
                }
                stem=stemData(**metadata).save()
                song=AudioMetadata.objects(id=song_id).update_one(push__stem_data=stem)

a= stem_loader()
a.create_metadata(stem_path=Path("data/stem_data/htdemucs_6s/normalized_0e70f59a-5b0f-4f11-8793-6f7a826efd20"),song_id="f9b585355cea187f8bc969d1d58109df25f998eb9d50b0a28348aff336ef0c857643a21f545a2603d5fa91d66ae616299ed6a09618e7d8b5e5229d1017411408")