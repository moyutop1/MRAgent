import argparse
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from utility.model import SetUtilityModel


def forward_record(model, record, device):
    return model(
        torch.as_tensor(record["candidate"], device=device).unsqueeze(0),
        torch.as_tensor(record["selected"], device=device).unsqueeze(0),
        torch.as_tensor(record["context"], device=device).unsqueeze(0),
        torch.tensor([record["operation"]], dtype=torch.long, device=device),
    )


def mean_loss(model, records, device):
    model.eval()
    with torch.inference_mode():
        losses = [
            F.binary_cross_entropy_with_logits(
                forward_record(model, record, device),
                torch.tensor([record["label"]], dtype=torch.float32, device=device),
            ).item()
            for record in records
        ]
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="checkpoints/set_utility.pt")
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--accumulate", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    records = torch.load(args.data, weights_only=False)
    samples = sorted({record["sample_id"] for record in records})
    random.shuffle(samples)
    validation_samples = set(samples[:max(1, len(samples) // 10)])
    train_records = [record for record in records if record["sample_id"] not in validation_samples]
    validation_records = [record for record in records if record["sample_id"] in validation_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SetUtilityModel(
        train_records[0]["candidate"].shape[1],
        args.hidden_dim,
        args.heads,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_records)
        model.train()
        optimizer.zero_grad()
        for step, record in enumerate(train_records, start=1):
            label = torch.tensor([record["label"]], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(
                forward_record(model, record, device), label
            ) / args.accumulate
            loss.backward()
            if step % args.accumulate == 0 or step == len(train_records):
                optimizer.step()
                optimizer.zero_grad()
        validation_loss = mean_loss(model, validation_records, device)
        print(f"epoch={epoch} validation_loss={validation_loss:.6f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({
                "config": model.checkpoint_config(),
                "model_state": model.state_dict(),
                "validation_loss": validation_loss,
            }, output)


if __name__ == "__main__":
    main()
