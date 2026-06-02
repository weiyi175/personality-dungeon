# Experiments Scripts

This folder contains deterministic experiment runners used by Phase 5.

## Files

- run_experiment.py: single-seed deterministic run
- run_matrix.py: batch runner for matrix.csv
- matrix.csv: example parameter matrix

## Example

```bash
cd /home/user/personality-dungeon
./venv/bin/python scripts/experiments/run_experiment.py --seed 42 --runs 10 --out reports/experiments/run_42.json
./venv/bin/python scripts/experiments/run_matrix.py --matrix scripts/experiments/matrix.csv --out reports/experiments/
```
