"""GTA VI trophy rarity forecaster: holder-count hazard model with player growth.

Forecasts, for every trophy, the percent of Exophase-tracked players who will
hold it at the case's target, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

The idea. A trophy's earn rate is a ratio, and the ratio moves for two
reasons that must not be confused:

  * holders grow -- players who have not earned the trophy yet earn it; and
  * the denominator grows -- Exophase tracks more players over time, and the
    new ones have mostly just started, so they dilute every rate they have
    not yet earned.

So the program works in counts, not rates. For each read it turns the rate
into a holder count, C = rate * N / 100, where N is the players tracked.
C only ever rises (nobody un-earns a trophy), while N's jumps are the
refreshes. It then:

  1. fits how N grows (a power law in hours since release, which is how a
     launch's tracked base grows before it saturates), and projects N forward;
  2. estimates, per trophy, a hazard lambda: the rate at which non-holders
     earn it, from how C grew against the average number of non-holders over
     the LAST 24 hours of the history (the recent growth regime is the one
     the future continues; an average over launch night, when everything
     grew fastest, would overstate the growth still to come), shrunk toward
     the tier mean so rare trophies with few holders do not get a wild
     estimate from a handful of counts, and decaying over the forecast
     horizon as the trophy saturates toward its ceiling;
  3. steps forward hour by hour from the last read to the target, growing N
     by the fitted law and C by lambda * (N - C) * dt, so the forecast rate
     100 * C / N carries both the progression of existing players and the
     dilution of new ones;
  4. centres the median on that rate and spreads the quantiles on the logit
     scale, wider the longer the horizon and the noisier that trophy's
     history, so the band is an honest statement of how far a week-one
     forecast can be off.

On a flat history (the practice cases, launch night before the first
refresh) the hazard is zero and N is flat, so the forecast is persistence:
there is no signal to do otherwise, and the practice exists to check that
the program reads a case and writes a valid forecast, not to pay. The final
case carries two and a half days of reads with real refreshes in them, which
is what this model is built to read.

Runs offline, standard library only, deterministic (no randomness, cases in
name order, every fit a closed-form least squares). Outside the grader's
container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and
/output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; nothing to import
# from beside it, so there is no sys.path line to write.
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Logit spread per square-root day of horizon (base uncertainty of a
# week-one forecast), plus a floor so even a short horizon keeps a band.
# The coefficient is deliberately below the house baseline's 0.5: this model
# reads the growth out of the history instead of assuming the rate stands
# still, so its residual uncertainty is smaller than persistence's, and a
# band wider than the error it actually makes pays at every quantile.
SPREAD_PER_SQRT_DAY = 0.25
SPREAD_FLOOR = 0.15


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt: datetime) -> float:
    return max(0.0, (dt - RELEASE).total_seconds() / 3600.0)


def fit_power_law(xs, ys):
    """Least-squares fit of log(y) = a + b*log(x+1); returns (a, b).

    xs are hours since release, ys the players tracked. With fewer than two
    distinct x values there is no slope to fit, so b is 0 (flat).
    """
    pts = [(math.log(x + 1.0), math.log(max(1.0, y))) for x, y in zip(xs, ys)]
    n = len(pts)
    if n < 2:
        return (math.log(max(1.0, ys[0])) if ys else 0.0, 0.0)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    b = sxy / sxx
    return (my - b * mx, b)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    # --- the time series: hours since release, N, and holder counts C ------
    times = []
    ns = []
    counts = []  # counts[k][i]: holders of trophy i at read k
    for read in history:
        t = hours_since_release(parse_time(read["at"]))
        n = float(read["players_tracked"])
        times.append(t)
        ns.append(n)
        row = []
        for i in range(len(trophies)):
            r = read["rates"][i]
            row.append(0.0 if r is None else float(r) * n / 100.0)
        counts.append(row)

    last_t = times[-1]
    last_n = ns[-1]
    last_c = counts[-1][:]
    n_trophies = len(trophies)

    # --- 1. how the tracked base N grows, and where it lands ---------------
    a_n, b_n = fit_power_law(times, ns)
    b_n = min(max(b_n, 0.0), 2.0)  # never shrink, never explode

    def n_at(t: float) -> float:
        v = math.exp(a_n + b_n * math.log(t + 1.0))
        return max(last_n, v)

    target_t = last_t + case["horizon_minutes"] / 60.0
    n_target = n_at(target_t)
    # Cap the growth: the tracked base does not tenfold in one horizon.
    n_target = min(n_target, last_n * 10.0)
    n_target = max(n_target, last_n)

    # --- 2. per-trophy hazard lambda, shrunk toward the tier mean ----------
    # lambda_i = (holders gained) / (average non-holders * hours), measured
    # over the LAST 24 hours of the history, not the whole of it: a trophy's
    # earn rate is fastest while players are first reaching it and slows as
    # it saturates, so the recent growth regime is the one the future
    # continues, and an average over launch night (when everything was
    # growing fastest) would overstate the growth still to come. A trophy
    # whose holders barely moved gets a lambda near zero; one with few
    # counts gets pulled toward its tier's mean so a single count does not
    # set the forecast.
    win_start = last_t - 24.0
    w0 = 0
    while w0 < len(times) - 1 and times[w0] < win_start:
        w0 += 1
    span_h = max(1.0, last_t - times[w0])
    lam = []
    tier_sum = {}
    tier_cnt = {}
    for i, tr in enumerate(trophies):
        gained = max(0.0, counts[-1][i] - counts[w0][i])
        nonholders = 0.0
        for k in range(w0, len(times)):
            nonholders += max(0.0, ns[k] - counts[k][i])
        avg_non = nonholders / max(1, len(times) - w0)
        li = gained / (avg_non * span_h) if avg_non > 0 and span_h > 0 else 0.0
        lam.append(li)
        key = tr.get("tier", "bronze")
        tier_sum[key] = tier_sum.get(key, 0.0) + li
        tier_cnt[key] = tier_cnt.get(key, 0) + 1
    tier_mean = {k: tier_sum[k] / tier_cnt[k] for k in tier_sum}
    global_mean = sum(lam) / max(1, len(lam))
    # Shrinkage weight: the more holders a trophy gained, the more its own
    # lambda is trusted; a trophy that gained nothing leans on its tier.
    lam_s = []
    for i, tr in enumerate(trophies):
        gained = max(0.0, counts[-1][i] - counts[w0][i])
        w = gained / (gained + 5.0)
        prior = tier_mean.get(tr.get("tier", "bronze"), global_mean)
        lam_s.append(w * lam[i] + (1.0 - w) * prior)
    # A hazard cannot be negative, and one hour cannot convert the whole
    # remaining base: cap it so a step never overshoots.
    lam_s = [min(max(l, 0.0), 0.5) for l in lam_s]

    # --- 3. step forward hour by hour to the target ------------------------
    # N follows the fitted law (re-scaled so it lands on n_target); each
    # trophy's holders grow by lambda * (non-holders) * dt. The hazard
    # DECAYS as the forecast runs on: a trophy's earn rate is fastest while
    # players are first reaching it and slows as it saturates toward its
    # ceiling (a constant hazard would run the rate up toward 100%, which no
    # trophy reaches). The decay timescale is ~3 days, so over a 4.5-day
    # horizon the hazard falls to about a fifth of its opening value.
    n_now = last_n
    c_now = last_c[:]
    dt = 1.0
    steps = int(math.ceil(case["horizon_minutes"] / 60.0))
    # The per-step N multiplier that lands exactly on n_target after `steps`.
    n_mult = (n_target / last_n) ** (1.0 / steps) if steps > 0 else 1.0
    hazard_tau_h = 72.0
    for s in range(steps):
        n_now = n_now * n_mult
        decay = math.exp(-(s * dt) / hazard_tau_h)
        for i in range(n_trophies):
            room = max(0.0, n_now - c_now[i])
            c_now[i] = min(n_now, c_now[i] + lam_s[i] * decay * room * dt)

    # --- 4. quantiles: median from the counts, spread from horizon + noise -
    # How noisy each trophy's logit rate was around its own history (a
    # trophy that jumped around gets a wider band).
    resid = []
    for i in range(n_trophies):
        vals = []
        for k in range(len(times)):
            if ns[k] > 0:
                vals.append(logit(counts[k][i] * 100.0 / ns[k]))
        if len(vals) >= 2:
            m = sum(vals) / len(vals)
            resid.append(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))
        else:
            resid.append(0.0)
    base_spread = SPREAD_PER_SQRT_DAY * math.sqrt(horizon_days) + SPREAD_FLOOR

    rows = []
    for i, tr in enumerate(trophies):
        rate = 100.0 * c_now[i] / n_now if n_now > 0 else 0.0
        lg = logit(rate)
        s = math.sqrt(base_spread ** 2 + (0.5 * resid[i]) ** 2)
        qs = [round4(clamp_pct(inv_logit(lg + z * s))) for z in Z]
        # Keep the five quantiles non-decreasing after rounding.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
