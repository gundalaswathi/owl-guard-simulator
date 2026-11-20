"""
owl_experiments.py - additional experiments for the OWL-Guard paper.
Uses the simulator in owl_sim.py (place both files in the same folder).

Usage:
    python owl_experiments.py learned    # RF/MLP baselines (Algorithm 3), burst TPR at 1% FPR
    python owl_experiments.py stricter   # TPR at FPR = 1e-2, 1e-3, 1e-4
    python owl_experiments.py noise      # Student-t and AR(1) noise
    python owl_experiments.py ablation   # component ablation
    python owl_experiments.py sens       # kappa, gate limit and sigma0 sensitivity
    python owl_experiments.py whole      # whole-pass delay (first-packet vs offset-prior initialisation)
    python owl_experiments.py all
Options: --seed N
"""
import sys
import math
import numpy as np
import owl_sim as o

WIN = 8
BURSTS = [20, 50, 100, 200, 500, 1000]


# ------------------------------------------------------------------ helpers
def run_all_scores(det, passes):
    return [det.score(p)[o.WARM:] for p in passes]


def packet_tpr(det, passes, labs, thr=None):
    thr = det.eta if thr is None else thr
    tp = fn = 0
    for p, lab in zip(passes, labs):
        s = det.score(p)[o.WARM:]
        l = lab[o.WARM:]
        flag = s > thr
        tp += int(np.sum(flag & l))
        fn += int(np.sum(~flag & l))
    return tp / (tp + fn)


def packet_fpr(det, passes, thr=None):
    thr = det.eta if thr is None else thr
    fp = n = 0
    for p in passes:
        s = det.score(p)[o.WARM:]
        fp += int(np.sum(s > thr))
        n += len(s)
    return fp / n


# ------------------------------------------------------------------ learned detectors (Algorithm 3)
def windows(x, w=WIN):
    n = len(x)
    if n < w:
        return np.empty((0, w))
    idx = np.arange(w)[None, :] + np.arange(n - w + 1)[:, None]
    return x[idx]


def features(p, kind):
    base = p.tau_obs if kind == "raw" else (p.tau_obs - p.tau_exp)
    return windows(base)          # row j covers packets j .. j+WIN-1 ; label of last packet


def build_training(rng, n_burst=300, n_ramp=200, n_clean=100):
    X = {"raw": [], "res": []}
    y = []
    passes = []
    for _ in range(n_burst):
        p = o.Pass(rng)
        d = 10 ** rng.uniform(math.log10(20), math.log10(20000))
        lab = o.add_burst(p, d, rng)
        passes.append((p, lab))
    for _ in range(n_ramp):
        p = o.Pass(rng)
        rho = 10 ** rng.uniform(math.log10(0.1), math.log10(10))
        lab = o.add_ramp(p, rho, rng)
        passes.append((p, lab))
    for _ in range(n_clean):
        p = o.Pass(rng)
        passes.append((p, np.zeros(p.n, bool)))
    for p, lab in passes:
        for kind in ("raw", "res"):
            X[kind].append(features(p, kind))
        y.append(lab[WIN - 1:])
    y = np.concatenate(y).astype(int)
    for k in X:
        X[k] = np.vstack(X[k])
    return X, y


class Learned:
    def __init__(self, model, kind, name):
        self.model, self.kind, self.name = model, kind, name

    def score(self, p):
        s = np.zeros(p.n)
        F = features(p, self.kind)
        if len(F):
            s[WIN - 1:] = self.model.predict_proba(F)[:, 1]
        return s

    def calibrate(self, clean, alpha=0.01):
        sc = np.concatenate([self.score(p)[o.WARM:] for p in clean])
        self.eta = np.quantile(sc, 1 - alpha)
        return self


def run_learned(seed=1):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    rng = np.random.default_rng(seed)
    print("Training RF and MLP on simulated passes (Algorithm 3) ...")
    X, y = build_training(rng)
    print(f"  training samples: {len(y)}  positive fraction: {y.mean():.3f}")
    dets = []
    for kind in ("raw", "res"):
        print(f'  fitting RF-{kind} ...', flush=True)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X[kind], y)
        print(f'  fitting MLP-{kind} ...', flush=True)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=seed, max_iter=200).fit(X[kind], y)
        dets.append(Learned(rf, kind, f"RF-{kind}"))
        dets.append(Learned(mlp, kind, f"MLP-{kind}"))
    clean = o.make_passes(400, rng)
    owl = o.OWL().calibrate(clean)
    stat = o.StaticOrbit().calibrate(clean)
    for d in dets:
        d.calibrate(clean)
    all_d = [("OWL-Guard", owl), ("Static orbit", stat)] + [(d.name, d) for d in dets]
    print("Packet-level TPR at 1% FPR")
    print("method        " + "".join(f"{d:>8d}us" for d in BURSTS))
    for name, det in all_d:
        row = []
        for d in BURSTS:
            ps = o.make_passes(200, rng)
            labs = [o.add_burst(p, d, rng) for p in ps]
            row.append(packet_tpr(det, ps, labs))
        print(f"{name:<14}" + "".join(f"{x:>10.3f}" for x in row))
    print("paper RF-raw  " + "".join(f"{x:>10.3f}" for x in [0.002, 0.003, 0.014, 0.077, 0.255, 0.375]))
    print("paper MLP-raw " + "".join(f"{x:>10.3f}" for x in [0.012, 0.020, 0.076, 0.146, 0.259, 0.374]))
    print("paper RF-res  " + "".join(f"{x:>10.3f}" for x in [0.041, 0.069, 0.320, 0.923, 1.000, 1.000]))
    print("paper MLP-res " + "".join(f"{x:>10.3f}" for x in [0.040, 0.080, 0.345, 0.929, 1.000, 1.000]))


# ------------------------------------------------------------------ stricter operating points
def run_stricter(seed=1):
    rng = np.random.default_rng(seed)
    clean = o.make_passes(3000, rng)
    owl = {a: o.OWL().calibrate(clean, alpha=a) for a in (1e-2, 1e-3, 1e-4)}
    stat = {a: o.StaticOrbit().calibrate(clean, alpha=a) for a in (1e-2, 1e-3, 1e-4)}
    test_clean = o.make_passes(3000, rng)
    attacked = {d: [] for d in (50, 100, 200)}
    for d in attacked:
        ps = o.make_passes(400, rng)
        labs = [o.add_burst(p, d, rng) for p in ps]
        attacked[d] = (ps, labs)
    print("TPR at stricter FPR targets (3000 clean calibration passes, 400 attacked passes per delay)")
    print("target  method        50us    100us   200us   realised FPR")
    for a in (1e-2, 1e-3, 1e-4):
        for name, det in (("OWL-Guard", owl[a]), ("Static orbit", stat[a])):
            row = [packet_tpr(det, *attacked[d]) for d in (50, 100, 200)]
            fpr = packet_fpr(det, test_clean)
            print(f"{a:<7g} {name:<12} " + " ".join(f"{x:7.4f}" for x in row) + f"   {fpr:.2e}")
    print("paper: OWL 0.855/0.983/1.000 (1e-2), 0.775/0.957/0.999 (1e-3), 0.665/0.912/0.994 (1e-4)")


# ------------------------------------------------------------------ noise robustness
def run_noise(seed=1, n=400):
    rng = np.random.default_rng(seed)
    clean = o.make_passes(400, rng)
    owl = o.OWL().calibrate(clean)
    stat = o.StaticOrbit().calibrate(clean)
    print("FPR with thresholds calibrated on Gaussian noise, and TPR at d = 100 us")
    cases = [("Gaussian", {}), ("Student t, nu=5", dict(noise="student", nu=5.0)),
             ("Student t, nu=3", dict(noise="student", nu=3.0))]
    for phi in (0.5, 0.8, 0.95):
        cases.append((f"AR(1), phi={phi}", dict(noise="ar1", phi=phi)))
    for name, kw in cases:
        cl = o.make_passes(n, rng, **kw)
        at = o.make_passes(n, rng, **kw)
        labs = [o.add_burst(p, 100, rng) for p in at]
        # recalibrated on the same noise
        cal = o.make_passes(400, rng, **kw)
        owl_r = o.OWL().calibrate(cal)
        print(f"  {name:<18} OWL FPR {packet_fpr(owl, cl) * 100:5.1f}%  static FPR "
              f"{packet_fpr(stat, cl) * 100:5.1f}%  OWL TPR@100us {packet_tpr(owl, at, labs):.3f}"
              f"  | recalibrated: TPR@100us {packet_tpr(owl_r, at, labs):.3f}")
    print("paper: t5 2.3%, t3 3.1% (static 1.4/1.2%); AR(1) FPR 8.1%, 20.3%, 25.6%; "
          "recalibrated TPR@100us 0.95, 0.83, 0.68")


# ------------------------------------------------------------------ ablation
class Variant:
    def __init__(self, mode, **kw):
        self.mode, self.kw = mode, kw
        self.c1 = self.c2 = self.eta = None

    def stats(self, p):
        if self.mode == "no_kf":
            z = (p.tau_obs - p.tau_exp) / p.sig
            g = np.zeros(p.n)
            gk = 0.0
            for k in range(p.n):
                gk = max(0.0, gk + z[k] - o.KAPPA)
                g[k] = gk
            return z, g
        if self.mode == "no_gate":
            return o.owl_stats(p, use_gate=False)
        if self.mode == "const_R":
            return o.owl_stats(p, const_R=(o.SIGMA0 * 1.0) ** 2)
        return o.owl_stats(p, **self.kw)

    def calibrate(self, clean, alpha=0.01):
        Z, G = [], []
        for p in clean:
            z, g = self.stats(p)
            Z.append(np.abs(z[o.WARM:]))
            G.append(g[o.WARM:])
        Z, G = np.concatenate(Z), np.concatenate(G)
        self.c1 = np.quantile(Z, 0.997)
        self.c2 = np.quantile(G, 0.997)
        if self.mode == "no_cusum":
            s = Z / self.c1
        elif self.mode == "no_inst":
            s = G / self.c2
        else:
            s = np.maximum(Z / self.c1, G / self.c2)
        self.eta = np.quantile(s, 1 - alpha)
        return self

    def score(self, p):
        z, g = self.stats(p)
        if self.mode == "no_cusum":
            return np.abs(z) / self.c1
        if self.mode == "no_inst":
            return g / self.c2
        return np.maximum(np.abs(z) / self.c1, g / self.c2)


def run_ablation(seed=1):
    rng = np.random.default_rng(seed)
    clean = o.make_passes(400, rng)
    variants = [("Full OWL-Guard", "full"), ("Without CUSUM (innovation only)", "no_cusum"),
                ("Without instantaneous test", "no_inst"), ("Without gate", "no_gate"),
                ("Constant noise R", "const_R"), ("Without Kalman tracker", "no_kf")]
    print("Ablation: packet-level TPR at 1% FPR")
    for name, mode in variants:
        det = Variant(mode).calibrate(clean)
        row = []
        for d in (50, 100):
            ps = o.make_passes(200, rng)
            labs = [o.add_burst(p, d, rng) for p in ps]
            row.append(packet_tpr(det, ps, labs))
        print(f"  {name:<34} {row[0]:.3f}  {row[1]:.3f}")
    print("paper: full 0.860/0.983; innovation only 0.105/0.511; no instantaneous 0.883/0.971; "
          "no gate 0.872/0.977; const R 0.806/0.963; no Kalman 0.036/0.188")


# ------------------------------------------------------------------ sensitivity
def run_sens(seed=1):
    rng = np.random.default_rng(seed)
    print("Sensitivity (TPR at 50 / 100 us)")
    clean = o.make_passes(400, rng)
    for kappa in (0.25, 0.5, 1.0):
        o.KAPPA = kappa
        det = o.OWL(kappa=kappa).calibrate(clean)
        r = []
        for d in (50, 100):
            ps = o.make_passes(200, rng)
            labs = [o.add_burst(p, d, rng) for p in ps]
            r.append(packet_tpr(det, ps, labs))
        print(f"  kappa={kappa:<5} {r[0]:.3f} {r[1]:.3f}")
    o.KAPPA = 0.5
    for gate in (3.0, 4.0, 6.0):
        det = o.OWL(gate=gate).calibrate(clean)
        r = []
        for d in (50, 100):
            ps = o.make_passes(200, rng)
            labs = [o.add_burst(p, d, rng) for p in ps]
            r.append(packet_tpr(det, ps, labs))
        print(f"  gate={gate:<5} {r[0]:.3f} {r[1]:.3f}")
    for s0 in (10.0, 25.0, 50.0, 100.0, 200.0):
        cl = o.make_passes(400, rng, sigma0=s0)
        det = o.OWL().calibrate(cl)
        r = []
        for d in (50, 100, 200):
            ps = o.make_passes(200, rng, sigma0=s0)
            labs = [o.add_burst(p, d, rng) for p in ps]
            r.append(packet_tpr(det, ps, labs))
        print(f"  sigma0={s0:<6} " + " ".join(f"{x:.3f}" for x in r) + "   (50, 100, 200 us)")
    print("paper: kappa 0.25/0.5/1.0 -> 0.81/0.86/0.74 at 50 us; sigma0=100 us -> 0.118/0.458/0.875")


# ------------------------------------------------------------------ whole-pass delay
def run_whole(seed=1):
    rng = np.random.default_rng(seed)
    clean = o.make_passes(300, rng)
    ds = [20, 50, 100, 200, 500, 1000]
    print("Delay applied to every packet of a pass: packet-level TPR")
    for init in ("first", "prior"):
        det = o.OWL(init=init).calibrate(clean)
        row = []
        for d in ds:
            ps = o.make_passes(300, rng)
            labs = [o.add_whole_pass(p, d) for p in ps]
            row.append(packet_tpr(det, ps, labs))
        print(f"  init={init:<6}" + "".join(f"{x:>8.3f}" for x in row) + "   (d = 20, 50, 100, 200, 500, 1000 us)")
    print("paper: first-packet init ~0.015 at all delays; prior init 1.000 from 500 us (0.081 at 200 us)")


if __name__ == "__main__":
    args = sys.argv[1:]
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 1
    cmd = args[0] if args else "all"
    table = {"learned": run_learned, "stricter": run_stricter, "noise": run_noise,
             "ablation": run_ablation, "sens": run_sens, "whole": run_whole}
    if cmd == "all":
        for f in table.values():
            f(seed)
            print()
    else:
        table[cmd](seed)
