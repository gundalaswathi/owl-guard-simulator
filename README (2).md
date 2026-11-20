# OWL-Guard simulator

Python simulator for the paper "OWL-Guard: Orbit-Aware Delay-Consistency Detection of Stealthy Delayed-Replay Attacks on Satellite Uplink Telecommand Channels".

## Files
- `owl_sim.py`: system model, burst and ramp delay attacks, OWL-Guard (gated two-state Kalman filter with CUSUM, calibrated on clean passes), and the static-orbit and fixed-window baselines.
- `owl_experiments.py`: additional experiments (learned RF/MLP baselines, stricter false-positive targets, heavy-tailed and AR(1) noise, ablation, sensitivity, whole-pass delay).

## Requirements
Python 3.8+ and NumPy. The learned baselines (`owl_experiments.py learned`) also need scikit-learn.

## Run
```
python owl_sim.py table3      # calibration constants and thresholds
python owl_sim.py burst       # TPR at 1% FPR versus burst delay
python owl_sim.py ramp        # ramp attack detection
python owl_sim.py eph         # robustness to pass-constant ephemeris error
python owl_sim.py all

python owl_experiments.py learned    # RF/MLP baselines (several minutes)
python owl_experiments.py stricter   # TPR at FPR = 1e-2, 1e-3, 1e-4
python owl_experiments.py noise      # Student-t and AR(1) noise
python owl_experiments.py ablation   # component ablation
python owl_experiments.py sens       # kappa, gate limit and sigma0 sensitivity
python owl_experiments.py whole      # whole-pass delay
python owl_experiments.py all
```
Options: `--seed N` (random seed, default 1), `--npass N` (owl_sim.py only).

All results are Monte Carlo estimates: values differ slightly between seeds (for example, the packet-level TPR at 50 microseconds varies by about 0.01 between runs).

## Author
Gundala Swathi, Vellore Institute of Technology, Vellore, India (gundalaswathi@vit.ac.in)

## License
MIT
