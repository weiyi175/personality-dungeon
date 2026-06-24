"""Direction A mini-trial — does UNFREEZING the encoder break the ~0.40 ceiling?

Same encoder (paraphrase-multilingual-MiniLM-L12-v2), three regimes:
  frozen     : encoder frozen, train only the 9D head  (≈ the proven ~0.40 path)
  ft_top<k>  : fine-tune top-k transformer layers + head
  ft_full    : fine-tune everything
Train on 138 Opus-labeled real wills, eval on the 35 frozen Opus test wills
(per-trait Pearson r, mean r). If ft beats frozen meaningfully -> geometry can be
reshaped -> Direction A worth full investment; else A likely hopeless.

GPU (RTX 4070). Read-only except the result print.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from api.schemas import PERSONALITY_BASIS  # noqa: E402

SB = "reports/experiments/sbert_quality"
TRAIN = ROOT / SB / "opus_train_labels.json"
TEST = ROOT / SB / "claude_reference_labels.json"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def load(p):
    d = json.loads(Path(p).read_text())
    return list(d), np.array(list(d.values()), dtype=np.float32)


def mean_pool(out, mask):
    m = mask.unsqueeze(-1).float()
    return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Linear(128, 9))

    def forward(self, ids, mask):
        return self.head(mean_pool(self.enc(input_ids=ids, attention_mask=mask), mask))


def pear(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-12 else float("nan")


def encode_batch(tok, texts):
    t = tok(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
    return t["input_ids"].to(DEV), t["attention_mask"].to(DEV)


def test_r(net, tok, te_texts, Yte):
    net.eval()
    with torch.no_grad():
        ids, mask = encode_batch(tok, te_texts)
        P = net(ids, mask).cpu().numpy()
    per = {t: pear(Yte[:, j], P[:, j]) for j, t in enumerate(PERSONALITY_BASIS)}
    return float(np.nanmean(list(per.values()))), per


def run(mode: str, tr_texts, Ytr, va_texts, Yva, te_texts, Yte):
    tok = AutoTokenizer.from_pretrained(MODEL)
    net = Net().to(DEV)
    n_layers = len(net.enc.encoder.layer)
    if mode == "frozen":
        for p in net.enc.parameters():
            p.requires_grad_(False)
        enc_lr, epochs, patience = 0.0, 300, 25
    else:
        k = int(mode.split("top")[1]) if "top" in mode else n_layers
        for p in net.enc.parameters():
            p.requires_grad_(False)
        for layer in net.enc.encoder.layer[n_layers - k:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        enc_lr, epochs, patience = 2e-5, 60, 12

    params = [{"params": net.head.parameters(), "lr": 1e-3}]
    enc_p = [p for p in net.enc.parameters() if p.requires_grad]
    if enc_p:
        params.append({"params": enc_p, "lr": enc_lr})
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    lossf = nn.MSELoss()
    Ytr_t = torch.tensor(Ytr, device=DEV)
    ids_tr, mask_tr = encode_batch(tok, tr_texts)

    best_va, best_state, bad = -9, None, 0
    bs = 16
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(tr_texts))
        for i in range(0, len(tr_texts), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pred = net(ids_tr[b], mask_tr[b])
            loss = lossf(pred, Ytr_t[b]); loss.backward(); opt.step()
        va_r, _ = test_r(net, tok, va_texts, Yva)
        if va_r > best_va:
            best_va, best_state, bad = va_r, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    te, per = test_r(net, tok, te_texts, Yte)
    return te, best_va, per


def main():
    tr_texts, Ytr_all = load(TRAIN)
    te_texts, Yte = load(TEST)
    # internal val split of the 138 for early stopping
    idx = np.random.RandomState(0).permutation(len(tr_texts))
    nva = 28
    va_i, tr_i = idx[:nva], idx[nva:]
    tr_t = [tr_texts[i] for i in tr_i]; Ytr = Ytr_all[tr_i]
    va_t = [tr_texts[i] for i in va_i]; Yva = Ytr_all[va_i]
    print(f"device={DEV} train={len(tr_t)} val={len(va_t)} test={len(te_texts)}\n")

    results = {}
    for mode in ("frozen", "ft_top2", "ft_top4", "ft_full"):
        te, va, per = run(mode, tr_t, Ytr, va_t, Yva, te_texts, Yte)
        results[mode] = {"test_mean_r": te, "val_mean_r": va, "per_trait": per}
        print(f"{mode:<10} test_mean_r={te:.3f}  (val={va:.3f})")

    base = results["frozen"]["test_mean_r"]
    best = max(("ft_top2", "ft_top4", "ft_full"), key=lambda m: results[m]["test_mean_r"])
    print(f"\nfrozen baseline = {base:.3f} | best finetune ({best}) = {results[best]['test_mean_r']:.3f} "
          f"| delta = {results[best]['test_mean_r'] - base:+.3f}")
    print("best per-trait r:", {k: round(v, 2) for k, v in results[best]["per_trait"].items()})
    (ROOT / SB / "finetune_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
