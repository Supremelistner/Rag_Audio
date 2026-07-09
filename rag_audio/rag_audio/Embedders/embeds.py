from transformers import AutoProcessor, AutoModel,AutoFeatureExtractor
import torch
import torch.nn.functional as F
import librosa





class embeds:
    def __init__(self):

        self.wprocessor = AutoFeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus"
        )

        self.mprocessor = AutoFeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M",
            trust_remote_code=True
        )

        self.wavlm = AutoModel.from_pretrained(
            "microsoft/wavlm-base-plus"
        )

        self.mert = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M",
            trust_remote_code=True
        )        
        self.wavlm.eval()
        self.mert.eval()

    def _mean_pool(self, hidden_states):
        return hidden_states.mean(dim=1)
                
    def _normalize(self, embeddings):
        """L2-normalize so all vectors sit on the unit sphere."""
        return F.normalize(embeddings, p=2, dim=-1)

    def _pool_and_normalize(self, hidden_states):
        pooled = self._mean_pool(hidden_states)
        return self._normalize(pooled).squeeze(0).cpu().numpy()


    def get_wavlm_embeds(self, audio, sampling_rate=16000):
        inputs = self.wprocessor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            output = self.wavlm(**inputs)
        # last_hidden_state: [batch, time, hidden]
        return self._pool_and_normalize(output.last_hidden_state)

    def get_mert_embeds(self, audio, sampling_rate=24000):
        inputs = self.mprocessor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            output = self.mert(**inputs)
        # last_hidden_state: [batch, time, hidden]
        return self._pool_and_normalize(output.last_hidden_state)
    
    def embed(self, audio_path, stem_type):
        if stem_type == "vocals":
            audio, sr = librosa.load(audio_path, sr=16000)
            return self.get_wavlm_embeds(audio, sr)

        audio, sr = librosa.load(audio_path, sr=24000)
        return self.get_mert_embeds(audio, sr)