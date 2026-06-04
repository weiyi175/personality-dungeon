#!/usr/bin/env python
"""
P7-H: Attractor Landscape Explorer (2D Cell Mapping + Lyapunov Spectral Analysis)

Purpose:
  Map the personality space landscape in 2D projection (v₁, v₂)
  Characterize attractor basins, separatrix topology, and Lyapunov spectra
  
Key Features:
  - Phase 1: 2D grid cell mapping (1,681 × 200 round trajectories)
  - Phase 2: Boundary refinement scanning
  - Phase 3: Full 9D Lyapunov spectra (QR decomposition method)
  - Phase 4: Bifurcation continuation (pseudo-arclength)
  
Author: P7-H Research Pipeline
Date: 2025-06-04
"""

import argparse
import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import scipy.linalg as la

# Setup logging
logging.basicConfig(level=logging.INFO, format='[P7-H] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GridPoint:
    """Represent a grid point in (v₁, v₂) 2D plane"""
    i: int
    j: int
    epsilon_1: float
    epsilon_2: float
    final_norm: float = None
    final_distance: float = None
    attractor_category: str = None  # 'main', 'secondary', 'far', 'chaotic'
    terminal_personality: np.ndarray = None
    lyapunov_exponents: np.ndarray = None


class LandscapeExplorer:
    """
    Landscape Explorer for P7-H: Full phase space characterization
    
    Implements:
    - Cell mapping on (v₁, v₂) 2D projection
    - Lyapunov spectra calculation (QR method)
    - Attractor basin delineation
    """
    
    def __init__(self, alpha: float = 0.2, seed: int = 42, phase: int = 1):
        """
        Initialize landscape explorer
        
        Args:
            alpha: BL2 parameter
            seed: Random seed
            phase: Execution phase (1=grid, 2=boundary, 3=lyapunov, 4=bifurcation)
        """
        self.alpha = alpha
        self.seed = seed
        self.phase = phase
        self.rng = np.random.RandomState(seed)
        
        # Personality space baseline (must be before load_principal_vectors)
        self.feature_names = [
            "impulsiveness", "assertiveness", "optimism",
            "risk_aversion", "suspicion", "endurance",
            "randomness", "stability_seeking", "curiosity"
        ]
        
        # Load principal vectors from P7-F
        self.v1 = None
        self.v2 = None
        self.baseline_attractor = None
        self.load_principal_vectors()
        
        logger.info(f"Initialized: alpha={alpha}, seed={seed}, phase={phase}")
    
    def load_principal_vectors(self):
        """Load v₁ and v₂ from P7-F results"""
        pv_file = Path("reports/experiments/p7f_attractor_mapping/p7h_principal_vectors.json")
        
        if not pv_file.exists():
            raise FileNotFoundError(f"Principal vectors not found: {pv_file}")
        
        with open(pv_file, 'r') as f:
            data = json.load(f)
        
        self.v1 = np.array(data['v1']['values'])
        self.v2 = np.array(data['v2']['values'])
        
        logger.info(f"Loaded v₁ (σ={data['v1']['sigma']:.4f}, "
                   f"explained={data['v1']['explained_variance_pct']:.2f}%)")
        logger.info(f"Loaded v₂ (σ={data['v2']['sigma']:.4f}, "
                   f"explained={data['v2']['explained_variance_pct']:.2f}%)")
        
        # Load baseline attractor (P7-F α=0.2)
        attractor_file = Path("reports/experiments/p7f_attractor_mapping/p7f_attractor_coordinates.json")
        with open(attractor_file, 'r') as f:
            attractor_data = json.load(f)
        
        # Use first seed's attractor at α=0.2 as baseline
        baseline_dict = attractor_data['0.2'][0]  # seed 42
        self.baseline_attractor = np.array([baseline_dict[fname] for fname in self.feature_names])
        
        logger.info(f"Baseline attractor loaded: ||P₀|| = {np.linalg.norm(self.baseline_attractor):.4f}")
    
    def generate_grid_points(self, resolution: int = 0.001, 
                            radius: float = 0.020) -> List[GridPoint]:
        """
        Generate 2D grid of perturbation points
        
        Args:
            resolution: Grid step size (default 0.001)
            radius: Scanning radius from baseline (default ±0.020)
        
        Returns:
            List of GridPoint objects
        """
        # Calculate grid parameters
        n_steps = int(2 * radius / resolution) + 1
        range_values = np.linspace(-radius, radius, n_steps)
        
        points = []
        for i, eps1 in enumerate(range_values):
            for j, eps2 in enumerate(range_values):
                point = GridPoint(
                    i=i, j=j,
                    epsilon_1=eps1,
                    epsilon_2=eps2
                )
                points.append(point)
        
        logger.info(f"Generated {len(points)} grid points ({n_steps} × {n_steps})")
        return points
    
    def compute_attractor_distance(self, point: np.ndarray) -> float:
        """Distance to baseline attractor"""
        return np.linalg.norm(point - self.baseline_attractor)
    
    def classify_attractor_basin(self, distance: float, terminal_point: np.ndarray) -> str:
        """
        Classify which attractor basin a trajectory belongs to
        
        Categories:
            'main': Converges to main attractor (distance < 0.010)
            'secondary': Converges to secondary attractor (0.10 < distance < 0.12)
            'far': Far from both (distance > 0.15)
            'undefined': Other (possibly transient or chaotic)
        """
        if distance < 0.010:
            return 'main'
        elif 0.100 < distance < 0.120:
            return 'secondary'
        elif distance > 0.150:
            return 'far'
        else:
            return 'undefined'
    
    def run_trajectory_simulation(self, initial_personality: np.ndarray, 
                                 n_rounds: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a single trajectory simulation
        
        Placeholder: In production, call RLSessionEngine or simulation backend
        
        Args:
            initial_personality: 9D personality vector
            n_rounds: Number of rounds to simulate
        
        Returns:
            (final_personality, trajectory_matrix)
        """
        # TODO: Call actual simulation backend
        # For now, return mock data showing decay towards baseline
        
        trajectory = np.zeros((n_rounds, 9))
        trajectory[0] = initial_personality.copy()
        
        # Simulate decay towards baseline attractor
        decay_rate = 0.95
        for t in range(1, n_rounds):
            trajectory[t] = (decay_rate * trajectory[t-1] + 
                           (1 - decay_rate) * self.baseline_attractor)
        
        return trajectory[-1], trajectory
    
    def compute_lyapunov_spectra(self, initial_personality: np.ndarray,
                                n_rounds: int = 200) -> np.ndarray:
        """
        Compute full 9D Lyapunov spectrum using QR decomposition method
        
        Algorithm:
          1. Initialize tangent space basis (identity matrix)
          2. For each round:
             a) Compute Jacobian DF at current state
             b) Multiply: M ← DF · M
             c) QR decompose: M = Q·R
             d) Update M ← Q (preserve orthonormal basis)
             e) Accumulate log|R_ii| for each direction
          3. Lyapunov exponents: λᵢ = (1/T) Σ log|R_ii|
        
        Returns:
            Array of 9 Lyapunov exponents in descending order
        """
        # Initialize tangent space basis
        M = np.eye(9)
        lambda_sums = np.zeros(9)
        
        current_state = initial_personality.copy()
        
        for t in range(n_rounds):
            # Compute Jacobian at current state
            # Placeholder: Use finite differences
            h = 1e-6
            df = np.zeros((9, 9))
            
            for i in range(9):
                perturbed_state = current_state.copy()
                perturbed_state[i] += h
                
                # Simulate one step
                _, next_state_perturbed = self.run_trajectory_simulation(
                    perturbed_state, n_rounds=1
                )
                _, next_state = self.run_trajectory_simulation(
                    current_state, n_rounds=1
                )
                
                df[:, i] = (next_state_perturbed - next_state) / h
            
            # M ← DF · M
            M = df @ M
            
            # QR decomposition
            Q, R = la.qr(M)
            M = Q
            
            # Accumulate Lyapunov exponents
            diag_r = np.abs(np.diag(R))
            lambda_sums += np.log(diag_r + 1e-10)  # Add small epsilon for numerical stability
        
        # Normalize by number of rounds
        lambda_exponents = lambda_sums / n_rounds
        
        # Sort in descending order
        lambda_exponents = np.sort(lambda_exponents)[::-1]
        
        return lambda_exponents
    
    def run_phase_1_grid_scan(self, output_dir: Path):
        """
        Execute Phase 1: 2D grid cell mapping
        
        Generates 1,681 trajectories on 41×41 grid
        """
        logger.info("=" * 70)
        logger.info("PHASE 1: 2D Grid Cell Mapping")
        logger.info("=" * 70)
        
        # Generate grid points
        grid_points = self.generate_grid_points(resolution=0.001, radius=0.020)
        
        # Storage for results
        results = {
            'metadata': {
                'phase': 1,
                'alpha': self.alpha,
                'seed': self.seed,
                'timestamp': datetime.now().isoformat(),
                'grid_size': len(grid_points),
                'n_rounds': 200
            },
            'grid_points': []
        }
        
        # Run trajectories
        logger.info(f"Running {len(grid_points)} grid trajectories...")
        
        for idx, gp in enumerate(grid_points):
            if (idx + 1) % 200 == 0:
                logger.info(f"  Processed {idx + 1}/{len(grid_points)} points")
            
            # Generate initial personality
            initial_personality = (self.baseline_attractor + 
                                 gp.epsilon_1 * self.v1 + 
                                 gp.epsilon_2 * self.v2)
            
            # Run simulation
            final_personality, trajectory = self.run_trajectory_simulation(
                initial_personality, n_rounds=200
            )
            
            # Compute metrics
            distance = self.compute_attractor_distance(final_personality)
            norm = np.linalg.norm(final_personality)
            category = self.classify_attractor_basin(distance, final_personality)
            
            # Store results
            gp.final_distance = distance
            gp.final_norm = norm
            gp.attractor_category = category
            gp.terminal_personality = final_personality
            
            results['grid_points'].append({
                'i': gp.i,
                'j': gp.j,
                'epsilon_1': gp.epsilon_1,
                'epsilon_2': gp.epsilon_2,
                'final_norm': float(gp.final_norm),
                'final_distance': float(gp.final_distance),
                'attractor_category': gp.attractor_category,
                'terminal_personality': gp.terminal_personality.tolist()
            })
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'p7h_grid_scan_phase1.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Phase 1 results saved to {output_dir}")
        
        # Generate summary statistics
        self._generate_phase1_summary(results, output_dir)
    
    def _generate_phase1_summary(self, results: Dict, output_dir: Path):
        """Generate summary statistics for Phase 1"""
        grid_points = results['grid_points']
        
        categories = {}
        for gp in grid_points:
            cat = gp['attractor_category']
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        summary = {
            'total_points': len(grid_points),
            'basin_distribution': categories,
            'basin_percentages': {k: 100*v/len(grid_points) for k, v in categories.items()},
            'statistics': {
                'mean_distance': np.mean([gp['final_distance'] for gp in grid_points]),
                'std_distance': np.std([gp['final_distance'] for gp in grid_points]),
                'mean_norm': np.mean([gp['final_norm'] for gp in grid_points]),
                'std_norm': np.std([gp['final_norm'] for gp in grid_points]),
            }
        }
        
        with open(output_dir / 'p7h_phase1_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Basin distribution:")
        for cat, count in categories.items():
            pct = 100 * count / len(grid_points)
            logger.info(f"  {cat:15s}: {count:5d} ({pct:5.1f}%)")
    
    def run_phase_3_lyapunov(self, output_dir: Path, sample_points: Optional[List[Tuple[int,int]]] = None):
        """
        Execute Phase 3: Lyapunov spectra calculation
        
        Args:
            output_dir: Output directory
            sample_points: List of (i, j) coordinates to compute spectra for
                          If None, select key points automatically
        """
        logger.info("=" * 70)
        logger.info("PHASE 3: Lyapunov Spectral Analysis")
        logger.info("=" * 70)
        
        if sample_points is None:
            # Select key points: center, corners, boundary
            sample_points = [
                (20, 20),  # center (main attractor)
                (10, 10), (30, 30),  # diagonals
                (20, 0), (20, 40),  # axes
                (0, 20), (40, 20)
            ]
        
        spectra_results = {
            'metadata': {
                'phase': 3,
                'alpha': self.alpha,
                'seed': self.seed,
                'timestamp': datetime.now().isoformat(),
                'method': 'QR decomposition',
                'n_rounds': 200
            },
            'spectra': {}
        }
        
        logger.info(f"Computing Lyapunov spectra for {len(sample_points)} points...")
        
        for i, j in sample_points:
            # Generate initial personality
            eps1 = -0.020 + (i / 40) * 0.040
            eps2 = -0.020 + (j / 40) * 0.040
            
            initial_personality = (self.baseline_attractor + 
                                 eps1 * self.v1 + 
                                 eps2 * self.v2)
            
            # Compute spectrum
            spectrum = self.compute_lyapunov_spectra(initial_personality, n_rounds=200)
            
            spectra_results['spectra'][f"{i}_{j}"] = {
                'grid_position': {'i': i, 'j': j},
                'epsilon_1': float(eps1),
                'epsilon_2': float(eps2),
                'lyapunov_spectrum': spectrum.tolist(),
                'lambda_max': float(spectrum[0]),
                'lambda_min': float(spectrum[-1]),
                'trace': float(np.sum(spectrum)),
                'kaplan_yorke_dimension': self._compute_kaplan_yorke_dim(spectrum)
            }
            
            logger.info(f"  Point ({i},{j}): λ₁={spectrum[0]:.6f}, λ₉={spectrum[-1]:.6f}")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'p7h_lyapunov_spectra.json', 'w') as f:
            json.dump(spectra_results, f, indent=2)
        
        logger.info(f"✓ Lyapunov spectra saved to {output_dir}")
    
    def _compute_kaplan_yorke_dim(self, spectrum: np.ndarray) -> float:
        """
        Compute Kaplan-Yorke dimension
        
        d_KY = j + (Σ λᵢ / |λ_{j+1}|)
        where j is largest index with Σ λᵢ ≥ 0
        """
        cumsum = np.cumsum(spectrum)
        
        # Find j where sum becomes negative
        j = len(spectrum) - 1
        for idx, cs in enumerate(cumsum):
            if cs < 0:
                j = idx - 1
                break
        
        if j < 0:
            return len(spectrum)
        
        if j >= len(spectrum) - 1:
            return float(len(spectrum))
        
        d_ky = j + cumsum[j] / np.abs(spectrum[j + 1])
        return float(d_ky)


def main():
    parser = argparse.ArgumentParser(description='P7-H: Landscape Explorer')
    parser.add_argument('--alpha', type=float, default=0.2, help='BL2 parameter α')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3, 4],
                       help='Execution phase')
    parser.add_argument('--out', type=str, default='reports/experiments/p7h_landscape_explorer',
                       help='Output directory')
    parser.add_argument('--resolution', type=float, default=0.001,
                       help='Grid resolution (phase 1)')
    parser.add_argument('--radius', type=float, default=0.020,
                       help='Scanning radius (phase 1)')
    
    args = parser.parse_args()
    
    # Initialize explorer
    explorer = LandscapeExplorer(alpha=args.alpha, seed=args.seed, phase=args.phase)
    
    # Create output directory
    output_dir = Path(args.out)
    
    # Execute requested phase
    if args.phase == 1:
        explorer.run_phase_1_grid_scan(output_dir)
    elif args.phase == 3:
        explorer.run_phase_3_lyapunov(output_dir)
    else:
        logger.warning(f"Phase {args.phase} not yet implemented")
    
    logger.info("✓ P7-H execution complete")


if __name__ == '__main__':
    main()
