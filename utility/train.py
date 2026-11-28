import argparse
import math
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


def evaluate(model, records, device):
    model.eval()
    with torch.inference_mode():
        rows = []
        for record in records:
            label = torch.tensor([record["label"]], dtype=torch.float32, device=device)
            logit = forward_record(model, record, device)
            rows.append((
                F.binary_cross_entropy_with_logits(logit, label).item(),
                int((torch.sigmoid(logit) >= 0.5).item() == record["label"]),
                record["operation"],
            ))
    metrics = {
        "loss": sum(row[0] for row in rows) / len(rows),
        "accuracy": sum(row[1] for row in rows) / len(rows),
    }
    for operation, name in ((0, "write"), (1, "retrieve")):
        operation_rows = [row for row in rows if row[2] == operation]
        if operation_rows:
            metrics[f"{name}_accuracy"] = (
                sum(row[1] for row in operation_rows) / len(operation_rows)
            )
    return metrics


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
    if "split" in records[0]:
        train_records = [record for record in records if record["split"] == "train"]
        validation_records = [record for record in records if record["split"] == "validation"]
        test_records = [record for record in records if record["split"] == "test"]
    else:
        samples = sorted({record["sample_id"] for record in records})
        random.shuffle(samples)
        validation_samples = set(samples[:max(1, len(samples) // 10)])
        train_records = [record for record in records if record["sample_id"] not in validation_samples]
        validation_records = [record for record in records if record["sample_id"] in validation_samples]
        test_records = validation_records

    write_records = [record for record in train_records if record["operation"] == 0]
    retrieve_records = [record for record in train_records if record["operation"] == 1]
    write_repeats = math.ceil(len(retrieve_records) / len(write_records))
    train_records = retrieve_records + write_records * write_repeats
    positives = sum(record["label"] for record in train_records)
    pos_weight = torch.tensor(
        [(len(train_records) - positives) / positives], dtype=torch.float32
    )

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
    pos_weight = pos_weight.to(device)
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
                forward_record(model, record, device), label, pos_weight=pos_weight
            ) / args.accumulate
            loss.backward()
            if step % args.accumulate == 0 or step == len(train_records):
                optimizer.step()
                optimizer.zero_grad()
        validation = evaluate(model, validation_records, device)
        print(
            f"epoch={epoch} validation_loss={validation['loss']:.6f} "
            f"validation_accuracy={validation['accuracy']:.4f}"
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save({
                "config": model.checkpoint_config(),
                "model_state": model.state_dict(),
                "validation": validation,
            }, output)

    checkpoint = torch.load(output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, test_records, device)
    print("test " + " ".join(f"{key}={value:.4f}" for key, value in test.items()))


if __name__ == "__main__":
    main()
