"""Train the fresh DeepONet macro-element strain operator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .autograd import strain_jacobian_wrt_q
from .data import SyntheticConfig, SyntheticMacroDataset
from .models import MacroDeepONet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--basis", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/fresh_deeponet"))
    parser.add_argument("--no-j-features", action="store_true")
    return parser.parse_args()


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_count = 0
    for batch_cpu in loader:
        batch = to_device(batch_cpu, device)
        pred = model(batch["q_b"], batch["x_nodes"], batch["xi"])
        loss = nn.functional.mse_loss(pred, batch["epsilon"])
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * pred.shape[0]
        total_count += pred.shape[0]
    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = SyntheticMacroDataset(SyntheticConfig(num_samples=args.samples, seed=args.seed))
    n_train = int(0.85 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = MacroDeepONet(
        basis_dim=args.basis,
        hidden_dim=args.hidden,
        depth=args.depth,
        use_jacobian_features=not args.no_j_features,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_val_mse": best_val,
                },
                args.out_dir / "best.pt",
            )
        print(f"epoch {epoch:04d} train_mse={train_loss:.6e} val_mse={val_loss:.6e}")

    batch = next(iter(val_loader))
    batch = to_device(batch, device)
    eps, b_macro = strain_jacobian_wrt_q(
        model,
        batch["q_b"][:4],
        batch["x_nodes"][:4],
        batch["xi"][:4],
    )
    print(f"autograd epsilon shape: {tuple(eps.shape)}")
    print(f"autograd B shape: {tuple(b_macro.shape)}")
    print(f"best checkpoint: {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

