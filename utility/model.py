import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value=None):
        key_value = query if key_value is None else key_value
        attended, _ = self.attention(
            self.norm_q(query), self.norm_kv(key_value), self.norm_kv(key_value),
            need_weights=False,
        )
        query = query + self.dropout(attended)
        return query + self.dropout(self.feed_forward(self.norm_ff(query)))


class SetUtilityModel(nn.Module):
    """Four attention blocks for one shared WRITE/RETRIEVE utility."""

    def __init__(self, input_dim, hidden_dim=384, num_heads=6, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.operation_embedding = nn.Embedding(2, hidden_dim)
        self.empty_set = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.set_blocks = nn.ModuleList(
            [AttentionBlock(hidden_dim, num_heads, dropout) for _ in range(2)]
        )
        self.candidate_to_set = AttentionBlock(hidden_dim, num_heads, dropout)
        self.candidate_to_context = AttentionBlock(hidden_dim, num_heads, dropout)
        self.utility_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_items, selected_set, context, operation):
        candidate = self.input_projection(candidate_items).mean(dim=1, keepdim=True)
        return self._score(candidate, selected_set, context, operation).squeeze(1)

    def score_candidates(self, candidate_vectors, selected_set, context, operation):
        candidate = self.input_projection(candidate_vectors)
        return self._score(candidate, selected_set, context, operation)

    def _score(self, candidate, selected_set, context, operation):
        candidate = candidate + self.operation_embedding(operation).unsqueeze(1)

        if selected_set.shape[1] == 0:
            selected = self.empty_set.expand(candidate.shape[0], -1, -1)
        else:
            selected = self.input_projection(selected_set)
        for block in self.set_blocks:
            selected = block(selected)

        candidate = self.candidate_to_set(candidate, selected)
        candidate = self.candidate_to_context(
            candidate, self.input_projection(context)
        )
        return self.utility_head(candidate).squeeze(-1)

    def checkpoint_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }
