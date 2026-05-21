#!/usr/bin/env python3
"""補充短句版本，調整長度分佈"""

import csv
from pathlib import Path

SHORT_VARIANTS = [
    # 每行格式: (original_long_text, short_variant, trait, endpoint)
    ("我做事憑直覺，先衝再說", "直覺行動", "impulsiveness", "high"),
    ("冒險是我最愛的感覺", "冒險成癮", "impulsiveness", "high"),
    ("反正試過才知道，何必害怕", "試試看", "impulsiveness", "high"),
    ("我謹慎評估後再行動", "謹慎為上", "impulsiveness", "low"),
    ("計劃周密是我的作風", "計劃周密", "impulsiveness", "low"),
    ("我需要充分時間來思考後果", "三思而後行", "impulsiveness", "low"),
    
    ("我喜歡主導對話和決策", "我要掌控", "assertiveness", "high"),
    ("掌控局面是我的風格", "領導者氣質", "assertiveness", "high"),
    ("退縮不是我的個性", "無所畏懼", "assertiveness", "high"),
    ("我偏好聆聽，不喜歡出風頭", "沉默是金", "assertiveness", "low"),
    ("我的聲音不重要，聽他人更重要", "傾聽他人", "assertiveness", "low"),
    ("我在團體中傾向沉默", "內向安靜", "assertiveness", "low"),
    
    ("未來一定更好，我非常樂觀", "永遠樂觀", "optimism", "high"),
    ("再大的困難也會過去", "困難過去", "optimism", "high"),
    ("好事將會發生在我身上", "好運降臨", "optimism", "high"),
    ("事情很少如預期發展", "事與願違", "optimism", "low"),
    ("我對未來沒什麼信心", "前路黑暗", "optimism", "low"),
    ("我總是看到問題而非機會", "只見困難", "optimism", "low"),
    
    ("我能長期堅持一件困難的事", "堅持到底", "endurance", "high"),
    ("困難不會擊倒我，我會堅持到底", "越戰越勇", "endurance", "high"),
    ("放棄不在我的選項中", "決不放棄", "endurance", "high"),
    ("做事缺乏耐心，遇到困難就放棄", "容易放棄", "endurance", "low"),
    ("困難會讓我想退縮", "害怕困難", "endurance", "low"),
    ("耐心不是我的強項", "沒耐心", "endurance", "low"),
    
    ("我不信任陌生人的好意", "不信任人", "suspicion", "high"),
    ("人心險惡，我時刻警覺", "人心叵測", "suspicion", "high"),
    ("每個人都有隱藏的動機", "有所隱瞞", "suspicion", "high"),
    ("人性本善，我願意相信大多數人", "性本善", "suspicion", "low"),
    ("我傾向於相信他人的善意", "相信善意", "suspicion", "low"),
    ("我很容易信任他人", "容易相信", "suspicion", "low"),
    
    ("我喜歡隨機應變，不按計劃行事", "隨性即可", "randomness", "high"),
    ("結構化的日程令我窒息", "拒絕計劃", "randomness", "high"),
    ("臨時改變對我來說很正常", "常改計劃", "randomness", "high"),
    ("穩定規律的生活讓我感到安心", "規律安心", "randomness", "low"),
    ("我需要結構化的日程和計劃", "需要計劃", "randomness", "low"),
    ("預測性讓我感到安全", "可預測安全", "randomness", "low"),
    
    ("穩定規律的生活讓我感到安心", "穩定舒適", "stability_seeking", "high"),
    ("我尋求生活的連貫性和一致性", "保持一致", "stability_seeking", "high"),
    ("變化令我不安，我渴望穩定", "害怕變化", "stability_seeking", "high"),
    ("我喜歡隨機應變，不按計劃行事", "求變求新", "stability_seeking", "low"),
    ("變化是生活的香料，我擁抱它", "擁抱變化", "stability_seeking", "low"),
    ("穩定的生活對我來說太無聊", "反感無聊", "stability_seeking", "low"),
    
    ("探索未知事物是我最大樂趣", "探索樂趣", "curiosity", "high"),
    ("我充滿好奇心，什麼都想嘗試", "充滿好奇", "curiosity", "high"),
    ("未知領域吸引著我", "未知吸引", "curiosity", "high"),
    ("熟悉的環境才讓我有安全感", "熟悉安全", "curiosity", "low"),
    ("我對新事物沒什麼興趣", "對新事物冷淡", "curiosity", "low"),
    ("我滿足於現有的知識", "現狀滿足", "curiosity", "low"),
    
    ("我願意接受風險以換取回報", "接受風險", "risk_aversion", "low"),
    ("冒險", "冒險心強", "risk_aversion", "low"),
    ("做事不計風險是我的作風", "不計風險", "risk_aversion", "low"),
    ("風險令我卻步，我選擇安全", "風險卻步", "risk_aversion", "high"),
    ("保守", "保守穩妥", "risk_aversion", "high"),
    ("我總是尋求最安全的選項", "最安全選項", "risk_aversion", "high"),
]

def append_short_variants(input_csv: Path, output_csv: Path):
    """讀取原始 CSV，加入短句變體"""
    rows = []
    
    # 讀取原始
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 加入短句
    for text, short_text, trait, endpoint in SHORT_VARIANTS:
        rows.append({
            "text": short_text,
            "length_bucket": "1-5",
            "trait": trait,
            "endpoint": endpoint,
        })
    
    # 寫入
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "length_bucket", "trait", "endpoint"])
        writer.writeheader()
        writer.writerows(rows)
    
    # 統計
    print(f"✓ 已補充短句，總計 {len(rows)} 句")
    
    buckets = {}
    for row in rows:
        b = row["length_bucket"]
        buckets[b] = buckets.get(b, 0) + 1
    
    print("\n調整後長度分佈:")
    for b in sorted(buckets.keys()):
        pct = 100 * buckets[b] / len(rows)
        print(f"  {b}: {buckets[b]} ({pct:.1f}%)")

if __name__ == "__main__":
    input_csv = Path("/home/user/personality-dungeon/data/personality_seed_texts_v2.csv")
    output_csv = Path("/home/user/personality-dungeon/data/personality_seed_texts_v2.csv")
    append_short_variants(input_csv, output_csv)
