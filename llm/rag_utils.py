import os
# avoid torchvision compatibility issues when importing torch
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["DISABLE_TORCHVISION"] = "1"
import torch
from tqdm import tqdm
from llm.embeddings import get_openai_embedding, set_openai_key


def get_embeddings(inputs, mode='context'):
    """Generate sentence/query vectors via the OpenAI (over OpenRouter) text-embedding API, with L2 normalization.

    (The historical dpr/contriever/dragon local-encoder backends are no longer used and were removed; only OpenAI embedding is used now.)
    The mode arg is kept for signature compatibility and makes no difference on the OpenAI path.
    """
    set_openai_key()
    all_embeddings = []
    batch_size = 24
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size)):
            vecs = get_openai_embedding(inputs[i:(i + batch_size)])
            embeddings = torch.tensor(vecs, dtype=torch.float32, device=device)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0).cpu().numpy()
