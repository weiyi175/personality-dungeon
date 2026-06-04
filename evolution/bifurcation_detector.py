"""
P7-H Bifurcation Detection Module
=================================

邊界穩定性檢測和敏感方向計算

核心功能:
- 計算軌跡到分岔點的距離
- 識別不穩定特徵向量 (敏感方向)
- 估計邊界接近度與敏感強度

基礎: P7-H Phase 1 + Phase 3 Lyapunov 分析
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BifurcationDetector:
    """邊界穩定性檢測器"""
    
    def __init__(self, 
                 baseline_attractor: np.ndarray,
                 bifurcation_critical_distance: float = 0.007,
                 lyapunov_spectrum: Optional[np.ndarray] = None):
        """
        初始化邊界檢測器
        
        參數:
            baseline_attractor: 基準吸引子 (9D 向量)
            bifurcation_critical_distance: 臨界擾動距離 (ε_c)
                估計值來自 P7-G，實驗驗證值來自 Phase 1
            lyapunov_spectrum: Lyapunov 指數 (9D 向量)
                來自 P7-H Phase 3: [1.29e-10, ..., 7.32e-11]
        """
        self.baseline_attractor = baseline_attractor
        self.bifurcation_critical_distance = bifurcation_critical_distance
        
        # P7-H Phase 3 測量的 Lyapunov 譜
        if lyapunov_spectrum is None:
            # 默認使用 Phase 3 測量值
            self.lyapunov_spectrum = np.array([
                1.29e-10, 1.29e-10, 1.29e-10,
                1.29e-10, 1.29e-10, 1.29e-10,
                7.32e-11, 7.32e-11, 7.32e-11
            ])
        else:
            self.lyapunov_spectrum = lyapunov_spectrum
        
        self.kaplan_yorke_dimension = 9.0  # 全 9D 參與
        
        logger.info(f"[BifurcationDetector] Initialized")
        logger.info(f"  Critical distance: {bifurcation_critical_distance:.3e}")
        logger.info(f"  Kaplan-Yorke dimension: {self.kaplan_yorke_dimension}")
        logger.info(f"  λ_max: {self.lyapunov_spectrum.max():.3e}")
        logger.info(f"  λ_min: {self.lyapunov_spectrum.min():.3e}")
    
    def compute_bifurcation_distance(self, personality_vector: np.ndarray) -> Dict:
        """
        計算人格向量到分岔點的距離
        
        輸入:
            personality_vector: 當前人格向量 (9D)
            
        輸出:
            {
                'distance_to_attractor': float,
                'bifurcation_proximity': float (0-1),
                'is_critical': bool,
                'proximity_level': str ('safe', 'warning', 'critical'),
                'critical_threshold': float,
            }
        """
        # 計算與基準吸引子距離
        distance = np.linalg.norm(personality_vector - self.baseline_attractor)
        
        # 規範化到 [0, 1] 相對於臨界距離
        proximity = min(1.0, distance / self.bifurcation_critical_distance)
        
        # 分類邊界狀態
        if proximity < 0.5:
            proximity_level = 'safe'
        elif proximity < 0.8:
            proximity_level = 'warning'
        else:
            proximity_level = 'critical'
        
        is_critical = proximity > 0.8
        
        return {
            'distance_to_attractor': float(distance),
            'bifurcation_proximity': float(proximity),
            'is_critical': is_critical,
            'proximity_level': proximity_level,
            'critical_threshold': 0.8,
            'safe_zone_radius': self.bifurcation_critical_distance * 0.5,
            'warning_zone_radius': self.bifurcation_critical_distance * 0.8,
        }
    
    def estimate_sensitive_direction(self, 
                                     personality_vector: np.ndarray,
                                     method: str = 'marginal_stability') -> Dict:
        """
        估計最敏感的擾動方向
        
        方法 1: 'marginal_stability'
            使用邊界穩定性的特徵 (所有 λ ≈ 0)
            返回均勻分佈在所有維度上的敏感方向
            
        方法 2: 'dominant_eigenvalue'
            返回最大 Lyapunov 指數對應的特徵向量
            (需要計算 Jacobian)
            
        輸入:
            personality_vector: 當前人格向量
            method: 'marginal_stability' 或 'dominant_eigenvalue'
            
        輸出:
            {
                'direction': np.ndarray (9D 單位向量),
                'eigenvalue': float (Lyapunov 指數估計),
                'sensitivity_strength': float (敏感強度),
                'method': str,
                'dimensions_involved': list (最活躍的維度編號),
            }
        """
        if method == 'marginal_stability':
            # 基於邊界穩定性：所有維度都有相似的 λ 值
            # 使用加權組合，權重基於 Lyapunov 指數倒數
            weights = 1.0 / (np.abs(self.lyapunov_spectrum) + 1e-12)
            weights = weights / np.sum(weights)  # 規範化
            
            # 敏感方向是權重方向
            direction = weights / np.linalg.norm(weights)
            
            # 敏感強度 = 最大 Lyapunov 指數
            sensitivity_strength = float(self.lyapunov_spectrum.max())
            
            # 最活躍的維度 (權重最大的)
            top_dims = np.argsort(weights)[::-1][:3]  # 前 3 個
            
        elif method == 'dominant_eigenvalue':
            # 簡化版本：假設主特徵向量沿著主方向
            # (完整版本需要計算 Jacobian)
            
            # 對於本系統，主特徵向量接近 v₁ (P7-F 的主向量)
            direction = np.zeros(9)
            direction[0] = 1.0  # v₁ 方向
            
            sensitivity_strength = float(self.lyapunov_spectrum.max())
            top_dims = [0]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'direction': direction,
            'eigenvalue': sensitivity_strength,
            'sensitivity_strength': sensitivity_strength,
            'method': method,
            'dimensions_involved': top_dims.tolist(),
            'lambda_spectrum': self.lyapunov_spectrum.tolist(),
        }
    
    def estimate_jacobian_eigenvalues(self) -> Dict:
        """
        返回已知的 Jacobian 特性 (來自 Phase 3)
        
        輸出:
            {
                'eigenvalues': np.ndarray (Lyapunov 指數),
                'trace': float (跡，應 ≈ 0),
                'determinant': float (行列式，應 ≈ 0),
                'marginal_stability_indicator': float (越接近 0 越邊界),
            }
        """
        trace = np.sum(self.lyapunov_spectrum)
        
        # 行列式 ≈ exp(sum of λ) 
        # 因為所有 λ ≈ 0，det ≈ 1
        det_estimate = np.exp(trace)
        
        # 邊界穩定性指標 = λ_max (應接近 0)
        marginal_indicator = self.lyapunov_spectrum.max()
        
        return {
            'eigenvalues': self.lyapunov_spectrum.tolist(),
            'trace': float(trace),
            'determinant_estimate': float(det_estimate),
            'marginal_stability_indicator': float(marginal_indicator),
            'is_boundary_stable': marginal_indicator < 1e-9,
        }
    
    def predict_bifurcation_outcome(self,
                                     personality_vector: np.ndarray,
                                     perturbation_direction: np.ndarray,
                                     perturbation_magnitude: float) -> Dict:
        """
        預測擾動後的系統行為
        
        基於邊界穩定性理論:
        - 沿著敏感方向的擾動會導致軌跡遷移
        - 擾動大小與成功機率相關
        
        輸入:
            personality_vector: 當前人格向量
            perturbation_direction: 擾動方向 (9D)
            perturbation_magnitude: 擾動幅度
            
        輸出:
            {
                'predicted_trajectory': np.ndarray (擾動後的向量),
                'bifurcation_success_probability': float (0-1),
                'expected_displacement': float,
                'recommendation': str,
            }
        """
        # 規範化擾動方向
        perturb_dir_norm = perturbation_direction / (
            np.linalg.norm(perturbation_direction) + 1e-12
        )
        
        # 應用擾動
        perturbed_vector = (
            personality_vector + perturbation_magnitude * perturb_dir_norm
        )
        
        # 檢查是否在邊界
        bifurc_info = self.compute_bifurcation_distance(personality_vector)
        proximity = bifurc_info['bifurcation_proximity']
        
        # 成功機率取決於邊界接近度
        # 越接近邊界，越容易被擾動
        if proximity > 0.8:
            success_prob = min(0.95, 0.5 + proximity * 0.5)
        elif proximity > 0.5:
            success_prob = min(0.70, 0.2 + proximity * 0.3)
        else:
            success_prob = min(0.40, 0.1 + proximity * 0.2)
        
        # 期望位移（基於 Lyapunov 指數）
        # λ 越大，擾動影響越大
        expected_displacement = (
            perturbation_magnitude * self.lyapunov_spectrum.max()
        )
        
        # 建議
        if success_prob > 0.7:
            recommendation = 'HIGH_SUCCESS_PROBABILITY'
        elif success_prob > 0.5:
            recommendation = 'MODERATE_SUCCESS'
        else:
            recommendation = 'LOW_SUCCESS_PROBABILITY'
        
        return {
            'predicted_trajectory': perturbed_vector.tolist(),
            'bifurcation_success_probability': float(success_prob),
            'expected_displacement': float(expected_displacement),
            'recommendation': recommendation,
            'current_proximity': float(proximity),
        }


def load_bifurcation_detector_from_files(
    baseline_attractor_path: str,
    lyapunov_spectrum_path: str,
    bifurcation_critical_distance: float = 0.007
) -> BifurcationDetector:
    """
    從檔案載入邊界檢測器
    
    檔案來源:
    - baseline_attractor: P7-F 結果 (基準吸引子坐標)
    - lyapunov_spectrum: P7-H Phase 3 結果 (Lyapunov 譜)
    """
    # 載入基準吸引子
    with open(baseline_attractor_path, 'r') as f:
        baseline_data = json.load(f)
    baseline_vector = np.array(baseline_data['coordinates'])
    
    # 載入 Lyapunov 譜
    with open(lyapunov_spectrum_path, 'r') as f:
        spectrum_data = json.load(f)
    
    # 取第一個採樣點的譜（所有點相同）
    first_spectrum = list(spectrum_data['spectra'].values())[0]
    lyapunov_vector = np.array(first_spectrum['lyapunov_spectrum'])
    
    return BifurcationDetector(
        baseline_attractor=baseline_vector,
        bifurcation_critical_distance=bifurcation_critical_distance,
        lyapunov_spectrum=lyapunov_vector
    )


# ==================== 測試範例 ====================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 創建檢測器
    baseline = np.array([0.1, -0.05, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.00])
    detector = BifurcationDetector(baseline_attractor=baseline)
    
    # 測試 1: 安全區
    safe_personality = baseline + 0.002 * np.random.randn(9)
    print("\n=== Test 1: Safe Zone ===")
    info = detector.compute_bifurcation_distance(safe_personality)
    print(f"Distance: {info['distance_to_attractor']:.3e}")
    print(f"Proximity: {info['bifurcation_proximity']:.2f}")
    print(f"Level: {info['proximity_level']}")
    
    # 測試 2: 警告區
    warning_personality = baseline + 0.005 * np.random.randn(9)
    print("\n=== Test 2: Warning Zone ===")
    info = detector.compute_bifurcation_distance(warning_personality)
    print(f"Distance: {info['distance_to_attractor']:.3e}")
    print(f"Proximity: {info['bifurcation_proximity']:.2f}")
    print(f"Level: {info['proximity_level']}")
    
    # 測試 3: 敏感方向
    print("\n=== Test 3: Sensitive Direction ===")
    sensitive = detector.estimate_sensitive_direction(safe_personality)
    print(f"Direction (first 3 dims): {sensitive['direction'][:3]}")
    print(f"Sensitivity strength: {sensitive['sensitivity_strength']:.3e}")
    print(f"Top dimensions: {sensitive['dimensions_involved']}")
    
    # 測試 4: 擾動預測
    print("\n=== Test 4: Bifurcation Outcome Prediction ===")
    perturb_dir = sensitive['direction']
    outcome = detector.predict_bifurcation_outcome(
        safe_personality, 
        perturb_dir, 
        perturbation_magnitude=0.005
    )
    print(f"Success probability: {outcome['bifurcation_success_probability']:.2%}")
    print(f"Recommendation: {outcome['recommendation']}")
    print(f"Expected displacement: {outcome['expected_displacement']:.3e}")
