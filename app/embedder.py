from transformers import AutoTokenizer, AutoModel
import torch

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

layoutlm_tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
layoutlm_model = AutoModel.from_pretrained("microsoft/layoutlm-base-uncased")

def embed_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


