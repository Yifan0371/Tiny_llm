import torch
from tiny_transformer import TinyTransformer
import struct


# -------------------------------------------------------------
# 工具函数：把 tensor 以 float32 二进制写入文件
# -------------------------------------------------------------
def write_tensor(f, t):
    t = t.contiguous().view(-1).float()
    f.write(struct.pack(f'{t.numel()}f', *t))


# -------------------------------------------------------------
# 导出权重
# -------------------------------------------------------------
def export_weights(model, path="tiny_model.bin"):
    with open(path, "wb") as f:

        # -----------------------------------------------------
        # 1) embedding
        # -----------------------------------------------------
        write_tensor(f, model.embedding.weight)

        # -----------------------------------------------------
        # 2) N 个 TransformerBlock
        # -----------------------------------------------------
        for blk in model.blocks:
            attn = blk.attn
            ffn  = blk.ffn
            ln1  = blk.ln1
            ln2  = blk.ln2

            # q_proj
            write_tensor(f, attn.q_proj.weight)
            write_tensor(f, attn.q_proj.bias)

            # k_proj
            write_tensor(f, attn.k_proj.weight)
            write_tensor(f, attn.k_proj.bias)

            # v_proj
            write_tensor(f, attn.v_proj.weight)
            write_tensor(f, attn.v_proj.bias)

            # o_proj
            write_tensor(f, attn.o_proj.weight)
            write_tensor(f, attn.o_proj.bias)

            # ffn fc1
            write_tensor(f, ffn.fc1.weight)
            write_tensor(f, ffn.fc1.bias)

            # ffn fc2
            write_tensor(f, ffn.fc2.weight)
            write_tensor(f, ffn.fc2.bias)

            # layernorm1 γ / β
            write_tensor(f, ln1.weight)
            write_tensor(f, ln1.bias)

            # layernorm2 γ / β
            write_tensor(f, ln2.weight)
            write_tensor(f, ln2.bias)

        # -----------------------------------------------------
        # 3) lm_head
        # -----------------------------------------------------
        write_tensor(f, model.lm_head.weight)
        write_tensor(f, model.lm_head.bias)

    print(f"[OK] 权重已导出到 {path}")


# -------------------------------------------------------------
# 运行
# -------------------------------------------------------------
if __name__ == "__main__":
    model = TinyTransformer(
        vocab_size=1000,
        hidden_dim=128,
        num_heads=8,
        num_layers=2,
        max_seq_len=16
    )

    export_weights(model)
