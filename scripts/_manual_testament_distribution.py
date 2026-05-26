from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from api.personality_sbert_inference import infer_personality_vector_sbert
from api.schemas import PERSONALITY_BASIS

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# 手工語料：真實遺言語感，80 條，1-5 與 6-10 各 40
CORPUS: list[str] = [
    # 1-5
    "活下去",
    "別怕",
    "我先走",
    "別哭",
    "保重",
    "快逃",
    "撐住",
    "對不起",
    "我愛你",
    "謝謝你",
    "別等我",
    "快關門",
    "相信你",
    "救大家",
    "照顧媽",
    "別回頭",
    "帶他走",
    "替我活",
    "別冒險",
    "記得回家",
    "把燈關了",
    "把門鎖好",
    "去找阿明",
    "替我道歉",
    "守住出口",
    "關掉電源",
    "去地下室",
    "聽指揮",
    "找醫生",
    "留在原地",
    "先保命",
    "照顧自己",
    "一定要活",
    "替我看海",
    "幫我養狗",
    "別忘了笑",
    "記得吃飯",
    "保護同伴",
    "就照計畫",
    "現在撤退",
    # 6-10
    "不要替我報仇",
    "照顧好媽媽",
    "快帶孩子走",
    "先救受傷的人",
    "不要回頭找我",
    "我真的不後悔",
    "別讓他們知道",
    "幫我跟她道歉",
    "先去北側出口",
    "到天亮再行動",
    "先把電源關掉",
    "快通知所有人",
    "不要相信廣播",
    "先不要開大門",
    "把文件全刪掉",
    "守住最後出口",
    "替我照顧弟弟",
    "替我照顧妹妹",
    "記得回家吃飯",
    "你們一定要活",
    "把地圖帶上走",
    "先往橋那邊撤",
    "去碼頭集合吧",
    "不要再硬撐了",
    "先把武器放下",
    "別再逞強了",
    "記得留下記號",
    "先保護好爸爸",
    "先保護好媽媽",
    "先保護好孩子",
    "不要怪任何人",
    "替我把真相說出",
    "你比我更勇敢",
    "我一直相信你",
    "先別主動攻擊",
    "先躲進地下室",
    "從北門離開吧",
    "快去找警察來",
    "把窗全部封住",
    "今天先活下來",
]


def bucket(length: int) -> str:
    if 1 <= length <= 5:
        return "1-5"
    if 6 <= length <= 10:
        return "6-10"
    return "11-20"


pred_rows: list[dict[str, object]] = []
for i, text in enumerate(CORPUS, start=1):
    vec, meta = infer_personality_vector_sbert(text)
    row: dict[str, object] = {
        "id": i,
        "text": text,
        "length": len(text),
        "bucket": bucket(len(text)),
        "model": meta.get("model", "sbert-mlp-v7"),
    }
    row.update(vec)
    pred_rows.append(row)

# 1) 語料檔
corpus_path = OUT / "testament_manual_corpus_80.tsv"
with corpus_path.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "text", "length", "bucket"])
    for r in pred_rows:
        w.writerow([r["id"], r["text"], r["length"], r["bucket"]])

# 2) 預測檔
pred_path = OUT / "testament_manual_predictions.csv"
with pred_path.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "text", "length", "bucket", *PERSONALITY_BASIS])
    for r in pred_rows:
        w.writerow([r["id"], r["text"], r["length"], r["bucket"], *[r[t] for t in PERSONALITY_BASIS]])

# 3) 分佈報告（無真值）

def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {
        "n": len(rows),
        "per_trait": {},
    }
    for t in PERSONALITY_BASIS:
        vals = [float(r[t]) for r in rows]
        svals = sorted(vals)
        q = lambda p: svals[int((len(svals) - 1) * p)]
        near_clip = sum(1 for v in vals if abs(v) >= 0.95)
        out_of_range = sum(1 for v in vals if abs(v) > 1.0)
        out["per_trait"][t] = {
            "mean": mean(vals),
            "std": pstdev(vals),
            "min": min(vals),
            "p10": q(0.10),
            "p50": q(0.50),
            "p90": q(0.90),
            "max": max(vals),
            "near_clip_ratio": near_clip / len(vals),
            "out_of_range_ratio": out_of_range / len(vals),
        }
    return out

overall = summarize(pred_rows)
buckets: dict[str, list[dict[str, object]]] = {"1-5": [], "6-10": [], "11-20": []}
for r in pred_rows:
    buckets[str(r["bucket"])].append(r)

by_bucket = {k: summarize(v) for k, v in buckets.items() if v}

# top extremes for interpretability
extremes: dict[str, dict[str, object]] = {}
for t in PERSONALITY_BASIS:
    hi = max(pred_rows, key=lambda r: float(r[t]))
    lo = min(pred_rows, key=lambda r: float(r[t]))
    extremes[t] = {
        "max": {"id": hi["id"], "text": hi["text"], "value": float(hi[t])},
        "min": {"id": lo["id"], "text": lo["text"], "value": float(lo[t])},
    }

report = {
    "dataset": {
        "name": "testament_manual_corpus_80",
        "description": "無真值壓測；手工遺言語料",
        "count": len(pred_rows),
        "bucket_counts": {k: len(v) for k, v in buckets.items() if v},
        "length_limit": "<=20",
        "model": "sbert-mlp-v7",
    },
    "overall_distribution": overall,
    "bucket_distribution": by_bucket,
    "trait_extremes": extremes,
}

report_path = OUT / "testament_manual_distribution_report.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print("CREATED", corpus_path)
print("CREATED", pred_path)
print("CREATED", report_path)
print("COUNT", len(pred_rows))
print("BUCKETS", {k: len(v) for k, v in buckets.items() if v})
