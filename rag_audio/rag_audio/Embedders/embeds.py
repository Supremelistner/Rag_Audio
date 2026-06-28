from transformers import AutoProcessor, AutoModel
import torch
import torch.nn.functional as F



class embeds:
    def __init__(self):
        self.wprocessor = AutoProcessor.from_pretrained("microsoft/wavlm-base-plus-sd")
        self.mprocessor = AutoProcessor.from_pretrained("m-a-p/MERT-v1-95M")
        self.wavelm = AutoModel.from_pretrained("microsoft/wavlm-base-plus-sd")
        self.wavelm.eval()
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True, dtype="auto")
        self.mert.eval()

    def _mean_pool(self, hidden_states, attention_mask=None):
        """Average across the time dimension."""
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.shape).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)
        return pooled                               
    def _normalize(self, embeddings):
        """L2-normalize so all vectors sit on the unit sphere."""
        return F.normalize(embeddings, p=2, dim=-1)

    def _pool_and_normalize(self, hidden_states, attention_mask=None):
        pooled     = self._mean_pool(hidden_states, attention_mask)
        normalized = self._normalize(pooled)
        return normalized.squeeze(0).cpu().numpy()


    def get_wavlm_embeds(self, audio, sampling_rate=16000):
        inputs = self.wprocessor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            output = self.wavlm(**inputs)
        # last_hidden_state: [batch, time, hidden]
        return self._pool_and_normalize(output.last_hidden_state, inputs.get("attention_mask"))

    def get_mert_embeds(self, audio, sampling_rate=24000):
        inputs = self.mprocessor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            output = self.mert(**inputs)
        # last_hidden_state: [batch, time, hidden]
        return self._pool_and_normalize(output.last_hidden_state, inputs.get("attention_mask"))
    
    def embed(self, audio, stem_type):
        if stem_type == "vocals":
            return self.get_wavlm_embeds(audio)
        return self.get_mert_embeds(audio)