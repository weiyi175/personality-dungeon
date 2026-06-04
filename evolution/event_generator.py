"""
P7-H Event Generation Module
============================

基於邊界穩定性的遊戲事件設計

核心功能:
- 設計與敏感方向對齐的事件
- 根據邊界接近度調制事件強度
- 規劃多步事件序列觸發分岔

基礎: P7-H Phase 1 + Phase 3 邊界檢測結果
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class Event:
    """遊戲事件基類"""
    
    def __init__(self, 
                 event_id: str,
                 event_type: str,
                 text_content: str,
                 magnitude: float,
                 duration_rounds: int = 1,
                 target_dimension: Optional[int] = None):
        """
        初始化遊戲事件
        
        參數:
            event_id: 事件唯一識別符
            event_type: 事件類型 ('dialogue', 'action', 'consequence')
            text_content: 事件文本內容
            magnitude: 事件強度 (0-1)
            duration_rounds: 事件播放時長 (回合)
            target_dimension: 目標人格維度 (0-8)
        """
        self.event_id = event_id
        self.event_type = event_type
        self.text_content = text_content
        self.magnitude = float(np.clip(magnitude, 0, 1))
        self.duration_rounds = max(1, int(duration_rounds))
        self.target_dimension = target_dimension
        self.created_at = None
        self.metadata = {}
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'text_content': self.text_content,
            'magnitude': self.magnitude,
            'duration_rounds': self.duration_rounds,
            'target_dimension': self.target_dimension,
            'metadata': self.metadata,
        }


class BifurcationEventGenerator:
    """基於邊界穩定性的事件生成器"""
    
    # 預定義的事件模板
    EVENT_TEMPLATES = {
        'dialogue': [
            "你發現自己面臨一個{context}的選擇。",
            "一位{npc}告訴你：{advice}",
            "你突然意識到：{realization}",
            "發生了一件{event_type}的事情。",
        ],
        'action': [
            "你決定{action}",
            "你被迫{action}",
            "你主動{action}",
            "你無意中{action}",
        ],
        'consequence': [
            "因為你的決定，{consequence}發生了。",
            "你的選擇導致{consequence}",
            "出乎意料，{consequence}",
            "結果證明{consequence}",
        ],
    }
    
    def __init__(self, sensitive_direction: np.ndarray):
        """
        初始化事件生成器
        
        參數:
            sensitive_direction: 最敏感的擾動方向 (9D)
        """
        self.sensitive_direction = (
            sensitive_direction / 
            (np.linalg.norm(sensitive_direction) + 1e-12)
        )
        self.event_counter = 0
        
        logger.info(f"[BifurcationEventGenerator] Initialized")
        logger.info(f"  Sensitive direction norm: "
                   f"{np.linalg.norm(self.sensitive_direction):.3f}")
    
    def design_bifurcation_event(self,
                                  bifurcation_info: Dict,
                                  event_context: str = 'default') -> Event:
        """
        設計單個分岔觸發事件
        
        參數:
            bifurcation_info: 來自 BifurcationDetector 的信息
            event_context: 事件背景 ('dialogue', 'action', 'consequence')
            
        輸出:
            Event 對象
        """
        self.event_counter += 1
        event_id = f"bifurc_{self.event_counter:04d}"
        
        # 根據邊界接近度調制事件強度
        proximity = bifurcation_info['bifurcation_proximity']
        
        # 越接近分岔點，事件越弱就能觸發
        # 這模擬了邊界穩定性的高敏感性
        base_strength = 0.5 * (1.0 - proximity)
        
        # 加上隨機擾動以增加多樣性
        event_magnitude = base_strength + 0.1 * np.random.randn()
        event_magnitude = float(np.clip(event_magnitude, 0.1, 0.9))
        
        # 事件時長也與邊界接近度相關
        # 越接近邊界，事件越短（快速切換）
        base_duration = max(1, int(10 * proximity))
        event_duration = base_duration + np.random.randint(0, 2)
        
        # 生成事件文本
        if event_context == 'default':
            # 自動選擇事件類型
            event_types = ['dialogue', 'action', 'consequence']
            event_context = event_types[self.event_counter % 3]
        
        event_text = self._generate_event_text(
            event_context, 
            proximity,
            event_magnitude
        )
        
        # 識別目標維度
        top_dim = np.argmax(np.abs(self.sensitive_direction))
        
        event = Event(
            event_id=event_id,
            event_type=event_context,
            text_content=event_text,
            magnitude=event_magnitude,
            duration_rounds=event_duration,
            target_dimension=int(top_dim)
        )
        
        # 添加元數據
        event.metadata = {
            'bifurcation_proximity': proximity,
            'sensitive_dimension': int(top_dim),
            'generated_from_theory': True,
        }
        
        logger.debug(f"Generated event {event_id}: "
                    f"magnitude={event_magnitude:.2f}, "
                    f"duration={event_duration}, "
                    f"proximity={proximity:.2f}")
        
        return event
    
    def _generate_event_text(self, 
                             event_type: str, 
                             proximity: float,
                             magnitude: float) -> str:
        """
        生成事件文本
        
        基於邊界接近度和事件強度調整語氣
        """
        template = np.random.choice(
            self.EVENT_TEMPLATES.get(event_type, ['發生了一件事。'])
        )
        
        # 根據強度和接近度填充佔位符
        if proximity < 0.5:
            context = "溫和"
            npc = "一位朋友"
            advice = "你可能需要考慮改變想法"
            realization = "你已經走上了一條熟悉的道路"
        elif proximity < 0.8:
            context = "關鍵"
            npc = "一位神秘的人物"
            advice = "現在是時候做出重大改變了"
            realization = "你的世界觀正在改變"
        else:
            context = "生死攸關"
            npc = "一位預言家"
            advice = "你必須立即改變，否則後果不堪設想"
            realization = "你進入了未知的領域，回頭已是百年身"
        
        if magnitude < 0.3:
            action = "謹慎地前進"
            consequence = "一切緩慢變化"
            event_type_desc = "微妙"
        elif magnitude < 0.7:
            action = "果斷地改變方向"
            consequence = "事情迅速發展"
            event_type_desc = "重要"
        else:
            action = "激進地革命自己"
            consequence = "一切都大不相同了"
            event_type_desc = "劇烈"
        
        result = template.format(
            context=context, npc=npc, advice=advice,
            realization=realization, action=action,
            consequence=consequence, event_type=event_type_desc
        )
        
        return result
    
    def plan_bifurcation_sequence(self,
                                   bifurcation_info: Dict,
                                   target_success_probability: float = 0.7,
                                   max_events: int = 5) -> List[Event]:
        """
        規劃多步事件序列以觸發分岔
        
        策略:
        1. 開始時弱的事件（測試系統響應）
        2. 中間事件加強（推向邊界）
        3. 最後關鍵事件（越過分岔點）
        
        參數:
            bifurcation_info: 邊界檢測信息
            target_success_probability: 目標成功率
            max_events: 最多事件數
            
        輸出:
            Event 列表
        """
        sequence = []
        proximity = bifurcation_info['bifurcation_proximity']
        
        # 計算所需事件數
        if proximity > 0.8:
            n_events = 1  # 已在邊界，1 個事件足夠
        elif proximity > 0.5:
            n_events = min(2, max_events)  # 中等距離，2-3 個事件
        else:
            n_events = min(3, max_events)  # 遠離邊界，3-5 個事件
        
        logger.info(f"Planning bifurcation sequence: "
                   f"proximity={proximity:.2f}, n_events={n_events}")
        
        event_contexts = ['dialogue', 'action', 'consequence']
        
        for i in range(n_events):
            # 漸進式增強
            # 前期弱，後期強
            progress_ratio = (i + 1) / n_events
            strength_multiplier = 0.5 + 0.5 * progress_ratio
            
            # 修改臨時的 bifurcation_info 以增加接近度
            # (模擬累積效應)
            modified_info = bifurcation_info.copy()
            modified_proximity = min(
                1.0, 
                proximity + 0.1 * i * strength_multiplier
            )
            modified_info['bifurcation_proximity'] = modified_proximity
            
            # 選擇事件類型
            event_context = event_contexts[i % len(event_contexts)]
            
            # 生成事件
            event = self.design_bifurcation_event(
                modified_info, 
                event_context
            )
            
            # 調整強度以模擬序列效應
            event.magnitude *= strength_multiplier
            event.magnitude = float(np.clip(event.magnitude, 0.1, 0.9))
            
            sequence.append(event)
        
        logger.info(f"Planned {len(sequence)} events")
        return sequence
    
    def estimate_sequence_success(self, 
                                   sequence: List[Event],
                                   detector) -> Dict:
        """
        估計事件序列的成功概率
        
        基於:
        - 總累積強度
        - 邊界接近度
        - 事件數量
        """
        total_magnitude = sum(e.magnitude for e in sequence)
        avg_magnitude = total_magnitude / len(sequence) if sequence else 0
        
        # 獲得最新的邊界信息
        # (簡化版本，實際應該追蹤軌跡演化)
        
        # 估計成功概率
        # 公式: P(success) = 1 - exp(-total_magnitude * len(sequence))
        success_prob = 1.0 - np.exp(-total_magnitude * len(sequence) / 5.0)
        success_prob = float(np.clip(success_prob, 0, 0.95))
        
        return {
            'estimated_success_probability': success_prob,
            'total_magnitude': total_magnitude,
            'average_magnitude': avg_magnitude,
            'event_count': len(sequence),
            'recommendation': 'EXECUTE' if success_prob > 0.6 else 'REVISE',
        }


# ==================== 測試範例 ====================

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # 創建事件生成器
    sensitive_dir = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05])
    generator = BifurcationEventGenerator(sensitive_direction=sensitive_dir)
    
    # 模擬邊界信息
    bifurc_info = {
        'distance_to_attractor': 0.004,
        'bifurcation_proximity': 0.6,
        'is_critical': False,
    }
    
    # 測試 1: 單個事件設計
    print("\n=== Test 1: Single Event ===")
    event = generator.design_bifurcation_event(bifurc_info)
    print(f"Event ID: {event.event_id}")
    print(f"Type: {event.event_type}")
    print(f"Magnitude: {event.magnitude:.2f}")
    print(f"Duration: {event.duration_rounds} rounds")
    print(f"Text: {event.text_content}")
    
    # 測試 2: 事件序列規劃
    print("\n=== Test 2: Event Sequence ===")
    sequence = generator.plan_bifurcation_sequence(bifurc_info, max_events=3)
    print(f"Sequence length: {len(sequence)}")
    for i, evt in enumerate(sequence):
        print(f"  [{i+1}] {evt.event_type}: magnitude={evt.magnitude:.2f}, "
              f"duration={evt.duration_rounds}")
    
    # 測試 3: 成功概率估計
    print("\n=== Test 3: Success Estimation ===")
    success_est = generator.estimate_sequence_success(sequence, detector=None)
    print(f"Success probability: {success_est['estimated_success_probability']:.2%}")
    print(f"Recommendation: {success_est['recommendation']}")
    
    # 測試 4: 不同邊界狀態的事件
    print("\n=== Test 4: Events at Different Boundary States ===")
    for proximity in [0.2, 0.5, 0.8, 0.95]:
        bifurc_info['bifurcation_proximity'] = proximity
        evt = generator.design_bifurcation_event(bifurc_info)
        print(f"Proximity {proximity:.1f}: magnitude={evt.magnitude:.2f}, "
              f"duration={evt.duration_rounds}, text={evt.text_content[:50]}...")
