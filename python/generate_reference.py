from pathlib import Path

import torch
from tiny_transformer import TinyTransformer
from export_weights import export_weights


def write_matrix(f, name, tensor):
    # tensor is expected to be [rows, cols]
    rows, cols = tensor.shape
    f.write(f"{name} {rows} {cols}\n")
    flat = tensor.reshape(-1).tolist()
    for i in range(0, len(flat), 8):
        f.write(" ".join(f"{v:.6f}" for v in flat[i:i + 8]) + "\n")


def main():
    out_dir = Path(__file__).parent
    weight_path = out_dir / "tiny_model.bin"
    reference_path = out_dir / "reference_outputs.txt"

    torch.manual_seed(42)
    model = TinyTransformer(
        vocab_size=1000,
        hidden_dim=128,
        num_heads=8,
        num_layers=2,
        max_seq_len=16,
    )

    tokens = torch.tensor([10, 20, 30], dtype=torch.long)

    # Export weights for C++
    export_weights(model, str(weight_path))

    # Collect forward activations
    with torch.no_grad():
        x = model.embedding(tokens)  # [T, hidden]
        embedding_out = x.transpose(0, 1).contiguous()

        block_outputs = []
        for blk in model.blocks:
            x = blk(x)
            block_outputs.append(x.transpose(0, 1).contiguous())

        logits = model.lm_head(x).transpose(0, 1).contiguous()

    # Write reference activations
    with reference_path.open("w", encoding="utf-8") as f:
        write_matrix(f, "embedding", embedding_out)
        for idx, out in enumerate(block_outputs):
            write_matrix(f, f"block{idx}", out)
        write_matrix(f, "logits", logits)

    print("[OK] tiny_model.bin 和 reference_outputs.txt 已生成 (seed=42)")


if __name__ == "__main__":
    main()