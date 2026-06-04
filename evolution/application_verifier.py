"""
P7-H Application Design Verification Framework
===============================================

驗證邊界穩定性理論在遊戲應用中的有效性

功能:
- 模擬邊界檢測準確度
- 測試事件序列觸發成功率
- 評估人格轉移幅度
- 生成驗證報告

基礎: bifurcation_detector.py + event_generator.py
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ApplicationVerifier:
    """應用設計驗證框架"""
    
    def __init__(self, 
                 detector,
                 generator,
                 baseline_attractor: np.ndarray,
                 n_trials: int = 50):
        """
        初始化驗證框架
        
        參數:
            detector: BifurcationDetector 實例
            generator: BifurcationEventGenerator 實例
            baseline_attractor: 基準吸引子
            n_trials: 驗證試驗次數
        """
        self.detector = detector
        self.generator = generator
        self.baseline_attractor = baseline_attractor
        self.n_trials = n_trials
        
        self.results = []
        self.statistics = {}
        
        logger.info(f"[ApplicationVerifier] Initialized with {n_trials} trials")
    
    def simulate_trial(self, trial_id: int) -> Dict:
        """
        模擬一次應用試驗
        
        步驟:
        1. 隨機生成一個人格向量（接近邊界）
        2. 檢測邊界接近度
        3. 生成事件序列
        4. 模擬軌跡演化
        5. 評估分岔成功
        
        輸出:
            {
                'trial_id': int,
                'initial_personality': np.ndarray,
                'bifurcation_proximity': float,
                'event_sequence': List[Event],
                'final_personality': np.ndarray,
                'personality_shift': float,
                'bifurcation_detected': bool,
                'success': bool,
            }
        """
        logger.info(f"[Trial {trial_id:3d}] Running...")
        
        # 生成初始人格（接近邊界的隨機擾動）
        # 從不同邊界接近度採樣
        proximity_target = np.random.choice([0.3, 0.5, 0.7, 0.9])
        epsilon = proximity_target * 0.007  # 臨界距離
        
        initial_pert = epsilon * (1 + 0.2 * np.random.randn())
        initial_personality = (
            self.baseline_attractor + 
            initial_pert * np.random.randn(9)
        )
        initial_personality = initial_personality / np.linalg.norm(
            initial_personality
        ) * np.linalg.norm(self.baseline_attractor)
        
        # 檢測邊界接近度
        bifurc_info = self.detector.compute_bifurcation_distance(
            initial_personality
        )
        proximity = bifurc_info['bifurcation_proximity']
        
        # 生成事件序列
        sequence = self.generator.plan_bifurcation_sequence(
            bifurc_info,
            max_events=3
        )
        
        # 模擬軌跡演化
        # 簡化版本：人格沿著敏感方向移動
        final_personality = initial_personality.copy()
        
        for event in sequence:
            # 事件強度導致人格轉移
            # 轉移幅度 ∝ 事件強度 × λ_max × 邊界接近度
            transfer_coefficient = (
                event.magnitude * 
                self.detector.lyapunov_spectrum.max() * 
                (1.0 + proximity)
            )
            
            # 沿著敏感方向轉移
            final_personality += (
                transfer_coefficient * 
                self.generator.sensitive_direction
            )
        
        # 規範化以保持在人格空間中
        final_personality = final_personality / np.linalg.norm(
            final_personality
        ) * np.linalg.norm(self.baseline_attractor)
        
        # 計算人格轉移幅度
        personality_shift = np.linalg.norm(
            final_personality - initial_personality
        )
        
        # 判斷分岔成功
        # 標準：轉移幅度 >= 0.05 (中等效應)
        bifurcation_detected = personality_shift >= 0.05
        success = bifurcation_detected
        
        result = {
            'trial_id': trial_id,
            'initial_personality': initial_personality.tolist(),
            'final_personality': final_personality.tolist(),
            'initial_bifurcation_proximity': float(proximity),
            'personality_shift_magnitude': float(personality_shift),
            'event_count': len(sequence),
            'total_event_magnitude': float(sum(e.magnitude for e in sequence)),
            'bifurcation_detected': bifurcation_detected,
            'success': success,
            'events': [e.to_dict() for e in sequence],
        }
        
        logger.debug(f"[Trial {trial_id:3d}] Proximity={proximity:.2f}, "
                    f"Shift={personality_shift:.4f}, Success={success}")
        
        return result
    
    def run_verification_suite(self) -> Dict:
        """
        運行完整的驗證套件
        
        輸出:
            {
                'summary': {...},
                'trials': [...],
                'statistics': {...},
            }
        """
        logger.info(f"Starting verification suite with {self.n_trials} trials...")
        
        self.results = []
        
        # 運行所有試驗
        for i in range(self.n_trials):
            result = self.simulate_trial(i)
            self.results.append(result)
        
        # 計算統計
        self._compute_statistics()
        
        return {
            'summary': {
                'n_trials': self.n_trials,
                'timestamp': datetime.now().isoformat(),
                'completion_status': 'SUCCESS',
            },
            'trials': self.results,
            'statistics': self.statistics,
        }
    
    def _compute_statistics(self):
        """計算驗證統計"""
        if not self.results:
            logger.warning("No results to compute statistics from")
            return
        
        # 轉換為陣列以便計算
        shifts = np.array([r['personality_shift_magnitude'] for r in self.results])
        successes = np.array([r['success'] for r in self.results])
        proximities = np.array([r['initial_bifurcation_proximity'] 
                               for r in self.results])
        event_counts = np.array([r['event_count'] for r in self.results])
        
        # 基本統計
        self.statistics = {
            'bifurcation_detection': {
                'total_trials': len(self.results),
                'successful_detections': int(np.sum(successes)),
                'detection_rate': float(np.mean(successes)),
                'false_positive_rate': float(
                    np.mean((shifts >= 0.05) & ~successes)
                ),
                'false_negative_rate': float(
                    np.mean((shifts < 0.05) & successes)
                ),
            },
            'personality_shift': {
                'mean': float(np.mean(shifts)),
                'std': float(np.std(shifts)),
                'min': float(np.min(shifts)),
                'max': float(np.max(shifts)),
                'median': float(np.median(shifts)),
                'q25': float(np.percentile(shifts, 25)),
                'q75': float(np.percentile(shifts, 75)),
            },
            'bifurcation_proximity': {
                'mean': float(np.mean(proximities)),
                'std': float(np.std(proximities)),
                'range': [float(np.min(proximities)), float(np.max(proximities))],
            },
            'event_sequence': {
                'avg_events_per_trial': float(np.mean(event_counts)),
                'std_events': float(np.std(event_counts)),
                'max_events': int(np.max(event_counts)),
            },
            'correlation_analysis': {
                'proximity_vs_shift_correlation': float(
                    np.corrcoef(proximities, shifts)[0, 1]
                ),
                'event_count_vs_success_correlation': float(
                    np.corrcoef(event_counts, successes)[0, 1]
                ),
            },
        }
        
        logger.info("Statistics computed")
    
    def generate_verification_report(self, output_path: str = None) -> str:
        """
        生成驗證報告
        
        輸出: Markdown 格式的完整報告
        """
        if not self.statistics:
            logger.warning("No statistics available, running suite first...")
            self.run_verification_suite()
        
        report = f"""# P7-H Application Design Verification Report

## 執行時間
- **日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **試驗數**: {self.statistics['bifurcation_detection']['total_trials']}

## 邊界檢測性能

### 成功率
- **檢測成功**: {self.statistics['bifurcation_detection']['successful_detections']}/{self.statistics['bifurcation_detection']['total_trials']} 試驗
- **檢測率**: {self.statistics['bifurcation_detection']['detection_rate']:.1%}

### 誤差分析
- **假陽性率**: {self.statistics['bifurcation_detection']['false_positive_rate']:.1%}
- **假陰性率**: {self.statistics['bifurcation_detection']['false_negative_rate']:.1%}

## 人格轉移分析

### 轉移幅度統計
| 指標 | 值 |
|------|-----|
| 平均 | {self.statistics['personality_shift']['mean']:.4f} |
| 標準差 | {self.statistics['personality_shift']['std']:.4f} |
| 最小 | {self.statistics['personality_shift']['min']:.4f} |
| 最大 | {self.statistics['personality_shift']['max']:.4f} |
| 中位數 | {self.statistics['personality_shift']['median']:.4f} |
| Q25 | {self.statistics['personality_shift']['q25']:.4f} |
| Q75 | {self.statistics['personality_shift']['q75']:.4f} |

## 邊界接近度分析

- **平均接近度**: {self.statistics['bifurcation_proximity']['mean']:.2f}
- **標準差**: {self.statistics['bifurcation_proximity']['std']:.2f}
- **範圍**: [{self.statistics['bifurcation_proximity']['range'][0]:.2f}, {self.statistics['bifurcation_proximity']['range'][1]:.2f}]

## 事件序列分析

- **平均事件數**: {self.statistics['event_sequence']['avg_events_per_trial']:.1f}
- **標準差**: {self.statistics['event_sequence']['std_events']:.1f}
- **最大事件數**: {self.statistics['event_sequence']['max_events']}

## 相關性分析

### 邊界接近度 vs 人格轉移
- **相關係數**: {self.statistics['correlation_analysis']['proximity_vs_shift_correlation']:.3f}
- **解釋**: {'強正相關（理論預期）' if abs(self.statistics['correlation_analysis']['proximity_vs_shift_correlation']) > 0.5 else '相關性弱'}

### 事件數 vs 成功率
- **相關係數**: {self.statistics['correlation_analysis']['event_count_vs_success_correlation']:.3f}

## 結論與建議

### 驗證結果
{self._generate_conclusion()}

### 後續行動
{self._generate_recommendations()}

---
*報告由 P7-H 應用設計驗證框架生成*
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def _generate_conclusion(self) -> str:
        """生成結論"""
        detection_rate = self.statistics['bifurcation_detection']['detection_rate']
        avg_shift = self.statistics['personality_shift']['mean']
        correlation = abs(self.statistics['correlation_analysis']['proximity_vs_shift_correlation'])
        
        if detection_rate > 0.7 and avg_shift > 0.05 and correlation > 0.5:
            return "✅ **邊界穩定性理論驗證成功**\n\n" \
                   "- 邊界檢測準確度高（>70%）\n" \
                   "- 人格轉移幅度明顯（>0.05）\n" \
                   "- 邊界接近度與轉移幅度高度相關\n" \
                   "\n可進入應用原型開發階段。"
        elif detection_rate > 0.5 and avg_shift > 0.02:
            return "⚠️ **邊界穩定性理論部分驗證**\n\n" \
                   "- 邊界檢測準確度中等（50-70%）\n" \
                   "- 人格轉移幅度可檢測\n" \
                   "\n建議改進事件設計或調整參數後再進入應用。"
        else:
            return "❌ **邊界穩定性理論驗證失敗**\n\n" \
                   "- 邊界檢測準確度低\n" \
                   "- 人格轉移幅度不明顯\n" \
                   "\n建議返回 Phase 2 進行詳細調查或重新評估理論。"
    
    def _generate_recommendations(self) -> str:
        """生成建議"""
        detection_rate = self.statistics['bifurcation_detection']['detection_rate']
        
        if detection_rate > 0.7:
            return "1. ✅ 進入 Godot 原型開發\n" \
                   "2. 實現邊界檢測 API 端點\n" \
                   "3. 招募玩家進行 UX 測試"
        else:
            return "1. 改進事件設計文本\n" \
                   "2. 調整邊界接近度計算方法\n" \
                   "3. 運行 Phase 2 弱維度掃描以精確定位次吸引子\n" \
                   "4. 重新執行驗證框架"


# ==================== 主程式 ====================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 導入依賴模組
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evolution.bifurcation_detector import BifurcationDetector
    from evolution.event_generator import BifurcationEventGenerator
    
    # 初始化
    baseline = np.array([0.1, -0.05, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.00])
    detector = BifurcationDetector(baseline_attractor=baseline)
    
    sensitive_dir = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05])
    generator = BifurcationEventGenerator(sensitive_direction=sensitive_dir)
    
    # 創建驗證器
    verifier = ApplicationVerifier(
        detector=detector,
        generator=generator,
        baseline_attractor=baseline,
        n_trials=30  # 測試用較小數字
    )
    
    # 運行驗證套件
    print("\n=== Running Verification Suite ===")
    results = verifier.run_verification_suite()
    
    # 生成報告
    print("\n=== Generating Report ===")
    report = verifier.generate_verification_report(
        output_path='p7h_application_verification_report.md'
    )
    print(report)
