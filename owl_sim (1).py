"""
owl_sim.py  -  Reference simulator for OWL-Guard (delayed-replay detection on a
LEO uplink).  Written from the paper's Table 2, Eqs. (1)-(19) and Algorithms 1-4.

Units: time in seconds, delays/noise in microseconds (us), range in km.

Usage:
    python owl_sim.py table3          # calibration constants and thresholds
    python owl_sim.py burst           # TPR at 1% FPR vs burst delay (Table 4)
    python owl_sim.py ramp            # ramp detection (Table 7, OWL-Guard only)
    python owl_sim.py eph             # FPR/TPR vs pass-constant ephemeris error
    python owl_sim.py all
Optional:  --seed N   --npass N (scale the number of attacked passes)
"""
import sys
import math
import numpy as np

C_KM_S = 299792.458           # speed of light, km/s
RE = 6371.0                   # km
H = 550.0                     # km
RS = RE + H
V_ORB = math.sqrt(398600.4418 / RS)     # km/s (7.59 km/s)
OMEGA = V_ORB / RS                       # rad/s
DT = 1.0                                 # s, packet interval
T_WIN = 300                              # s, pass window
EMIN = 10.0                              # deg
WARM = 10                                # packets excluded (filter warm-up)

# ---- Table 2 parameters -------------------------------------------------
SIGMA0 = 25.0                  # us
OFFSET_SD = 50.0               # us, initial clock offset
DRIFT_SD = 0.05                # us/s, initial drift
QB = 0.25                      # us^2
QF = 4e-6                      # (us/s)^2
SIG_EPH_KM = 2.0               # km, pass-constant range error (1-sigma)
JITTER = 10.0                  # us, relay jitter in a burst
L_BURST = 30
N_RAMP = 120
GATE = 4.0
KAPPA = 0.5
PF0 = DRIFT_SD ** 2


def sigma_e(e_deg, sigma0=SIGMA0):
    """Eq. (5): elevation dependent noise (us)."""
    return sigma0 * (1.0 + 1.5 * np.exp(-(np.asarray(e_deg) - 10.0) / 12.0))


def slant_range_km(e_deg):
    """Eq. (1): slant range from elevation."""
    e = np.radians(e_deg)
    return np.sqrt(RS ** 2 - (RE * np.cos(e)) ** 2) - RE * np.sin(e)


def pass_geometry(emax_deg):
    """Elevation (deg) at 1 Hz over a 300 s window centred on closest approach.
    Spherical geometry: cos(psi) = cos(psi0) cos(omega t)."""
    em = math.radians(emax_deg)
    psi0 = math.acos(RE / RS * math.cos(em)) - em
    t = np.arange(-T_WIN // 2, T_WIN // 2 + 1, DT)
    cpsi = math.cos(psi0) * np.cos(OMEGA * t)
    psi = np.arccos(np.clip(cpsi, -1, 1))
    e = np.degrees(np.arctan2(cpsi - RE / RS, np.sin(psi)))
    keep = e >= EMIN
    return e[keep]


class Pass:
    """One simulated pass: elevation, expected delay and (clean) observed delay."""
    __slots__ = ("e", "tau_exp", "tau_obs", "sig", "n")

    def __init__(self, rng, sigma0=SIGMA0, sig_eph_km=SIG_EPH_KM, emax=None,
                 noise="gauss", phi=0.0, nu=5.0, along_track_km=None):
        if emax is None:
            emax = rng.uniform(30.0, 90.0)
        e = pass_geometry(emax)
        n = len(e)
        self.e = e
        self.n = n
        self.tau_exp = slant_range_km(e) / C_KM_S * 1e6          # us
        sig = sigma_e(e, sigma0)
        self.sig = sig
        # clock: offset b, drift f, random walk (Eq. 3)
        b = np.empty(n)
        f = rng.normal(0.0, DRIFT_SD)
        b0 = rng.normal(0.0, OFFSET_SD)
        wb = rng.normal(0.0, math.sqrt(QB), n)
        wf = rng.normal(0.0, math.sqrt(QF), n)
        bk = b0
        for k in range(n):
            b[k] = bk
            bk = bk + f * DT + wb[k]
            f = f + wf[k]
        # pass-constant ephemeris error (Eq. 6)
        eps = rng.normal(0.0, sig_eph_km) / C_KM_S * 1e6
        # measurement noise
        if noise == "gauss":
            nk = rng.normal(0.0, 1.0, n) * sig
        elif noise == "student":           # same variance as Gaussian
            nk = rng.standard_t(nu, n) * math.sqrt((nu - 2.0) / nu) * sig
        elif noise == "ar1":               # same marginal sd
            v = rng.normal(0.0, 1.0, n)
            u = np.empty(n)
            u[0] = v[0]
            s = math.sqrt(1.0 - phi ** 2)
            for k in range(1, n):
                u[k] = phi * u[k - 1] + s * v[k]
            nk = u * sig
        else:
            raise ValueError(noise)
        extra = 0.0
        if along_track_km is not None:
            # time-varying delay error ~ range-rate * along-track timing offset
            dt_off = along_track_km / V_ORB
            rr = np.gradient(slant_range_km(e), DT) * 0 + _range_rate(emax, n, e)
            extra = rr * dt_off / C_KM_S * 1e6
        self.tau_obs = self.tau_exp + eps + b + nk + extra


def _range_rate(emax, n, e_kept):
    """Range rate (km/s) over the pass, for the along-track error model."""
    em = math.radians(emax)
    psi0 = math.acos(RE / RS * math.cos(em)) - em
    t = np.arange(-T_WIN // 2, T_WIN // 2 + 1, DT)
    cpsi = math.cos(psi0) * np.cos(OMEGA * t)
    rng_km = np.sqrt(RE ** 2 + RS ** 2 - 2 * RE * RS * cpsi)
    rr = np.gradient(rng_km, DT)
    psi = np.arccos(np.clip(cpsi, -1, 1))
    ee = np.degrees(np.arctan2(cpsi - RE / RS, np.sin(psi)))
    return rr[ee >= EMIN]


# ---- attacks ---------------------------------------------------------------
def add_burst(p, d, rng):
    """Eq. (7). Returns attack indicator (bool array)."""
    lab = np.zeros(p.n, dtype=bool)
    start = int(rng.integers(WARM, max(WARM + 1, p.n - L_BURST)))
    idx = np.arange(start, min(start + L_BURST, p.n))
    a = np.maximum(0.0, d + rng.normal(0.0, JITTER, len(idx)))
    p.tau_obs[idx] += a
    lab[idx] = True
    return lab


def add_ramp(p, rho, rng):
    """Eq. (8)."""
    lab = np.zeros(p.n, dtype=bool)
    start = int(rng.integers(WARM, max(WARM + 1, p.n - N_RAMP)))
    idx = np.arange(start, min(start + N_RAMP, p.n))
    k = np.arange(1, len(idx) + 1)
    p.tau_obs[idx] += rho * k * DT
    lab[idx] = True
    return lab


def add_whole_pass(p, d):
    p.tau_obs += d
    return np.ones(p.n, dtype=bool)


# ---- detectors -------------------------------------------------------------
def owl_stats(p, qb=QB, qf=QF, gate=GATE, kappa=KAPPA, init="first",
              use_gate=True, const_R=None):
    """Algorithm 1, stages 2-8.  Returns |z|, z and CUSUM g for every packet
    (first packet: z=0, g=0, 'accepted without test')."""
    n = p.n
    r = p.tau_obs - p.tau_exp
    R = (p.sig ** 2) if const_R is None else np.full(n, const_R)
    z = np.zeros(n)
    g = np.zeros(n)
    if init == "first":
        b, f = r[0], 0.0
        p00, p01, p11 = R[0], 0.0, PF0
        start = 1
    else:                               # offset prior N(0, 50.4 us), test packet 0
        b, f = 0.0, 0.0
        p00, p01, p11 = OFFSET_SD ** 2 + (SIG_EPH_KM / C_KM_S * 1e6) ** 2, 0.0, PF0
        start = 0
    gk = 0.0
    for k in range(start, n):
        # prediction (Eq. 12)
        bm = b + f * DT
        fm = f
        a00 = p00 + 2 * DT * p01 + DT * DT * p11 + qb
        a01 = p01 + DT * p11
        a11 = p11 + qf
        # innovation (Eq. 13)
        nu = r[k] - bm
        S = a00 + R[k]
        zk = nu / math.sqrt(S)
        # gated update (Eqs. 14-15)
        if (not use_gate) or abs(zk) < gate:
            k0 = a00 / S
            k1 = a01 / S
            b = bm + k0 * nu
            f = fm + k1 * nu
            p00 = (1 - k0) * a00
            p01 = (1 - k0) * a01
            p11 = a11 - k1 * a01
        else:
            b, f, p00, p01, p11 = bm, fm, a00, a01, a11
        # CUSUM (Eq. 16)
        gk = max(0.0, gk + zk - kappa)
        z[k] = zk
        g[k] = gk
    return z, g


class OWL:
    def __init__(self, **kw):
        self.kw = kw
        self.c1 = self.c2 = self.eta = None

    def stats(self, p):
        return owl_stats(p, **self.kw)

    def calibrate(self, clean, alpha=0.01):
        """Algorithm 2 / Eq. (19)."""
        Z, G = [], []
        for p in clean:
            z, g = self.stats(p)
            Z.append(np.abs(z[WARM:]))
            G.append(g[WARM:])
        Z = np.concatenate(Z)
        G = np.concatenate(G)
        self.c1 = np.quantile(Z, 0.997)
        self.c2 = np.quantile(G, 0.997)
        s = np.maximum(Z / self.c1, G / self.c2)
        self.eta = np.quantile(s, 1.0 - alpha)
        return self

    def score(self, p):
        z, g = self.stats(p)
        return np.maximum(np.abs(z) / self.c1, g / self.c2)


class StaticOrbit:
    name = "Static orbit"

    def score(self, p):
        return np.abs(p.tau_obs - p.tau_exp)

    def calibrate(self, clean, alpha=0.01):
        s = np.concatenate([self.score(p)[WARM:] for p in clean])
        self.eta = np.quantile(s, 1 - alpha)
        return self


class FixedWindow:
    name = "Fixed window"

    def calibrate(self, clean, alpha=0.01):
        lo = min(p.tau_exp.min() for p in clean)
        hi = max(p.tau_exp.max() for p in clean)
        self.mid = 0.5 * (lo + hi)
        s = np.concatenate([self.score(p)[WARM:] for p in clean])
        self.eta = np.quantile(s, 1 - alpha)
        return self

    def score(self, p):
        return np.abs(p.tau_obs - self.mid)


# ---- experiments -------------------------------------------------------------
def make_passes(n, rng, **kw):
    return [Pass(rng, **kw) for _ in range(n)]


def tpr_fpr(det, passes, labels):
    tp = fn = fp = tn = 0
    for p, lab in zip(passes, labels):
        s = det.score(p)[WARM:]
        flag = s > det.eta
        l = lab[WARM:]
        tp += int(np.sum(flag & l))
        fn += int(np.sum(~flag & l))
        fp += int(np.sum(flag & ~l))
        tn += int(np.sum(~flag & ~l))
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return tpr, fpr


def calibrate_all(rng, n_clean=400, **owl_kw):
    clean = make_passes(n_clean, rng)
    dets = {"OWL-Guard": OWL(**owl_kw).calibrate(clean),
            "Static orbit": StaticOrbit().calibrate(clean),
            "Fixed window": FixedWindow().calibrate(clean)}
    return clean, dets


def run_table3(seed=1):
    rng = np.random.default_rng(seed)
    clean, dets = calibrate_all(rng)
    o = dets["OWL-Guard"]
    print("Calibration on 400 clean passes (paper: c1=2.95, c2=5.13, eta=0.925,"
          " static=153 us, fixed=1525 us)")
    print(f"  OWL-Guard : c1={o.c1:.2f}  c2={o.c2:.2f}  eta={o.eta:.3f}")
    print(f"  Static    : threshold = {dets['Static orbit'].eta:.0f} us")
    print(f"  Fixed win : threshold = {dets['Fixed window'].eta:.0f} us")
    dmin = min(p.tau_exp.min() for p in clean)
    dmax = max(p.tau_exp.max() for p in clean)
    print(f"  true delay range over simulated windows: {dmin:.0f} - {dmax:.0f} us"
          " (paper: 1835 - 4898)")


BURSTS = [20, 50, 100, 200, 500, 1000, 2000]


def run_burst(seed=1, scale=1.0):
    rng = np.random.default_rng(seed)
    clean, dets = calibrate_all(rng)
    n_att = int(200 * scale)
    print("Packet-level TPR at 1% FPR (clean calibration FPR is 1% by construction)")
    print("method        " + "".join(f"{d:>8d}us" for d in BURSTS))
    res = {k: [] for k in dets}
    for d in BURSTS:
        ps = make_passes(n_att, rng)
        labs = [add_burst(p, d, rng) for p in ps]
        for k, det in dets.items():
            res[k].append(tpr_fpr(det, ps, labs)[0])
    for k, v in res.items():
        print(f"{k:<14}" + "".join(f"{x:>10.3f}" for x in v))
    print("paper OWL     " + "".join(f"{x:>10.3f}" for x in
                                    [0.194, 0.870, 0.985, 1.0, 1.0, 1.0, 1.0]))
    print("paper static  " + "".join(f"{x:>10.3f}" for x in
                                    [0.025, 0.037, 0.194, 0.786, 1.0, 1.0, 1.0]))
    print("paper fixed   " + "".join(f"{x:>10.3f}" for x in
                                    [0.002, 0.005, 0.000, 0.000, 0.009, 0.067, 0.523]))


def run_ramp(seed=1, scale=1.0):
    rng = np.random.default_rng(seed)
    clean, dets = calibrate_all(rng)
    owl = dets["OWL-Guard"]
    n_att = int(150 * scale)
    print("Ramp attack, OWL-Guard: detection prob. (alarm inside ramp) and "
          "median delay injected before first alarm")
    for rho in [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
        det_cnt, inj = 0, []
        for _ in range(n_att):
            p = Pass(rng)
            lab = add_ramp(p, rho, rng)
            s = owl.score(p)
            idx = np.where(lab)[0]
            hit = np.where(s[idx] > owl.eta)[0]
            if len(hit):
                det_cnt += 1
                inj.append(rho * (hit[0] + 1))
        med = np.median(inj) if inj else float("nan")
        print(f"  rho={rho:<5} P(det)={det_cnt / n_att:.2f}  median injected={med:.1f} us")
    print("paper (Table 7): rho=0.5 -> 0.97, 1 -> 1.00 (24 us), 2 -> 1.00, 5 -> 1.00")


def run_eph(seed=1, scale=1.0):
    rng = np.random.default_rng(seed)
    clean, dets = calibrate_all(rng)
    n = int(150 * scale)
    print("Pass-constant ephemeris error: FPR / TPR(d=500us), thresholds fixed")
    for km in [0.5, 2, 5, 10, 20, 40]:
        cl = make_passes(n, rng, sig_eph_km=km)
        at = make_passes(n, rng, sig_eph_km=km)
        labs_c = [np.zeros(p.n, bool) for p in cl]
        labs_a = [add_burst(p, 500, rng) for p in at]
        row = f"  {km:>4} km  "
        for k, det in dets.items():
            fpr = tpr_fpr(det, cl, labs_c)[1]
            tpr = tpr_fpr(det, at, labs_a)[0]
            row += f"{k}: FPR {fpr * 100:5.1f}% TPR {tpr:.2f} | "
        print(row)
    print("paper: OWL FPR 0.8-1.2% everywhere; static 2.0% @10 km, 10.2% @20, 34.1% @40")


if __name__ == "__main__":
    args = sys.argv[1:]
    seed = 1
    scale = 1.0
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])
    if "--npass" in args:
        scale = float(args[args.index("--npass") + 1]) / 200.0
    cmd = args[0] if args else "all"
    if cmd in ("table3", "all"):
        run_table3(seed)
    if cmd in ("burst", "all"):
        run_burst(seed, scale)
    if cmd in ("ramp", "all"):
        run_ramp(seed, scale)
    if cmd in ("eph", "all"):
        run_eph(seed, scale)
