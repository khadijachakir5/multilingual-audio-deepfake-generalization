\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

import numpy as np
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

N_PERMUTATIONS = 999
N_BOOTSTRAP = 2_000


def _cohen(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) +
                   (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 1e-10 else np.nan


def _hellinger(p, q):
\
\
\
\
\
\

    comb = np.concatenate([p, q])
    lo, hi = np.nanpercentile(comb, 1), np.nanpercentile(comb, 99)
    if hi - lo < 1e-10:
        return 0.
    nb = max(30, int(np.floor(np.sqrt(min(len(p), len(q))))))
    bins = np.linspace(lo, hi, nb + 1)
    hp, _ = np.histogram(p, bins=bins)
    hq, _ = np.histogram(q, bins=bins)

    hp = hp / (hp.sum() + 1e-10)
    hq = hq / (hq.sum() + 1e-10)
    return float(np.linalg.norm(np.sqrt(hp) - np.sqrt(hq)) / np.sqrt(2))


def _srh_fake(v, la, ge):
\
\
\
\
\
\
\
\
\
\

    fin = np.isfinite(v)
    v, la, ge = v[fin], la[fin], ge[fin]
    if len(v) < 10:
        return {"eta2_lang": np.nan, "eta2_gen": np.nan, "ratio": np.nan}
    rk = rankdata(v).astype(float)
    SS = np.var(rk) * len(rk)
    if SS < 1e-10:
        return {"eta2_lang": 0., "eta2_gen": 0., "ratio": np.nan}

    def ss(g):
        gm = np.mean(rk)
        s = 0.
        for x in np.unique(g):
            mm = rk[g == x]
            s += len(mm) * (np.mean(mm) - gm) ** 2
        return s

    el = float(ss(la) / SS)
    eg = float(ss(ge) / SS)
    return {"eta2_lang": el, "eta2_gen": eg,
            "ratio": float(eg / el) if el > 1e-10 else np.inf}


def _srh_classwise(v, la, lb):
\
\

    res = {}
    for cv, cn in [(0, "real"), (1, "fake")]:
        m = lb == cv
        vv, ll = v[m], la[m]
        fin = np.isfinite(vv)
        vv, ll = vv[fin], ll[fin]
        if len(vv) < 10:
            res[f"eta2_{cn}"] = np.nan
            continue
        rk = rankdata(vv).astype(float)
        SS = np.var(rk) * len(rk)
        if SS < 1e-10:
            res[f"eta2_{cn}"] = 0.
            continue
        gm = np.mean(rk)
        s = 0.
        for x in np.unique(ll):
            mm = rk[ll == x]
            s += len(mm) * (np.mean(mm) - gm) ** 2
        res[f"eta2_{cn}"] = float(s / SS)
    ef = res.get("eta2_fake", np.nan)
    er = res.get("eta2_real", np.nan)
    rfr = float(ef / er) if (er and er > 1e-10) else np.nan
    if np.isnan(ef) or np.isnan(er):
        prof = "N/A"
    elif abs(rfr - 1.0) < 0.25:
        prof = "Symmetric"
    elif ef < 0.01:
        prof = "Quasi-invariant"
    elif rfr > 1.5:
        prof = "Asymmetric"
    else:
        prof = "Moderate"
    res["ratio_fr"] = rfr
    res["profile"] = prof
    return res


def _bootstrap_eta2_ratio(v_fake, la_fake, ge_fake, rng):
\
\

    fin = np.isfinite(v_fake)
    v, l, g = v_fake[fin], la_fake[fin], ge_fake[fin]
    n = len(v)
    if n < 10:
        return np.nan, np.nan

    def pair(v_, l_, g_):
        rk = rankdata(v_).astype(float)
        SS = np.var(rk) * len(rk)
        if SS < 1e-10:
            return 0., 0.
        gm = np.mean(rk)

        def ss2(grp):
            s2 = 0.
            for x in np.unique(grp):
                mm = rk[grp == x]
                s2 += len(mm) * (np.mean(mm) - gm) ** 2
            return s2

        return float(ss2(l_) / SS), float(ss2(g_) / SS)

    ratios = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        el, eg = pair(v[idx], l[idx], g[idx])
        if el > 1e-10:
            ratios.append(eg / el)
    if len(ratios) < 10:
        return np.nan, np.nan
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def bh_correct_interactions(p_values):
\
\
\
\
\
\
\
\

    p_filled = [p if (p is not None and not np.isnan(p)) else 1.0
                for p in p_values]
    _, p_corrected, _, _ = multipletests(p_filled, alpha=0.05, method="fdr_bh")
    return p_corrected.tolist()


def _interaction_test(v_fake, la_fake, ge_fake):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

    fin = np.isfinite(v_fake)
    v, la, ge = v_fake[fin], la_fake[fin], ge_fake[fin]
    if len(v) < 20:
        return {"eta2_interaction": np.nan, "p_interaction": np.nan}
    rk = rankdata(v).astype(float)
    SS = np.var(rk) * len(rk)
    if SS < 1e-10:
        return {"eta2_interaction": 0., "p_interaction": 1.}

    def ssf(g, r):
        gm = np.mean(r)
        s = 0.
        for x in np.unique(g):
            mm = r[g == x]
            s += len(mm) * (np.mean(mm) - gm) ** 2
        return s

    def ssi(la_, ge_, r_):
        gm = np.mean(r_)
        s = 0.
        for l in np.unique(la_):
            for g in np.unique(ge_):
                c = r_[(la_ == l) & (ge_ == g)]
                if len(c) > 0:
                    s += len(c) * (np.mean(c) - gm) ** 2
        return max(s - ssf(la_, r_) - ssf(ge_, r_), 0.0)

    obs = ssi(la, ge, rk)
    eta2 = float(obs / SS) if SS > 0 else np.nan
    rng_p = np.random.default_rng(0)
    cnt = 0
    for _ in range(N_PERMUTATIONS):
        p = rng_p.permutation(len(rk))
        if ssi(la[p], ge[p], rk) >= obs:
            cnt += 1
    return {"eta2_interaction": eta2,
            "p_interaction": float((cnt + 1) / (N_PERMUTATIONS + 1))}
