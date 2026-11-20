# OWL-Guard simulator

Python simulator for the paper "OWL-Guard: Orbit-Aware Delay-Consistency Detection of Stealthy Delayed-Replay Attacks on Satellite Uplink Telecommand Channels".

It implements the system model, burst and ramp delay attacks, OWL-Guard (gated two-state Kalman filter with CUSUM, calibrated on clean passes), and the static-orbit and fixed-window baselines.

## Requirements
Python 3.8+ and NumPy.

## Run
```
python "owl_sim (1).py" table3   # calibration constants and thresholds
python "owl_sim (1).py" burst    # TPR at 1% FPR versus burst delay
python "owl_sim (1).py" ramp     # ramp attack detection
python "owl_sim (1).py" eph      # robustness to ephemeris error
python "owl_sim (1).py" all
```
Options: `--seed N` (random seed, default 1), `--npass N`.

## Author
Gundala Swathi, Vellore Institute of Technology, Vellore, India (gundalaswathi@vit.ac.in)

## License
MIT
