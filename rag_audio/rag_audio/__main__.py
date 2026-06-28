from transformers import AutoModel

model = AutoModel.from_pretrained(
    "m-a-p/MERT-v1-95M",
    trust_remote_code=True
)
print(model.__class__.__name__)

