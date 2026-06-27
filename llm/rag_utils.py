import os
# avoid torchvision compatibility issues when importing torch
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["DISABLE_TORCHVISION"] = "1"
import torch
from tqdm import tqdm
from llm.embeddings import get_embedding, EMBEDDING_BATCH_SIZE


def get_embeddings(inputs, mode='context'):
    """Generate sentence/query vectors via the configured embedding backend.

    The mode arg is kept for signature compatibility.
    """
    all_embeddings = []
    batch_size = EMBEDDING_BATCH_SIZE
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size)):
            vecs = get_embedding(inputs[i:(i + batch_size)], batch_size=batch_size)
            embeddings = torch.tensor(vecs, dtype=torch.float32, device=device)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings)
    if not all_embeddings:
        return torch.empty((0, 0), dtype=torch.float32).numpy()
    return torch.cat(all_embeddings, dim=0).cpu().numpy()
