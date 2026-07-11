"""Self-contained demo: seed a synthetic warehouse and run the full model spine.

Real ingest needs cached download files for ten sources; this module instead seeds
the warehouse tables directly with a small synthetic-but-plausible dataset, then
runs the downstream modeling stages in order — geo -> features -> both members ->
stack -> conformal -> copula -> publish — so anyone can watch the pipeline produce
its JSON artifacts end to end with one command (``midterms26 demo``).

The synthetic generative model is deliberately simple and *known*: a race margin is
``national[cycle] + district_pvi + incumbency_bump + noise``. That lets the demo
double as a smoke test — the conformal intervals should cover, and the stack should
not do worse than either member — without needing real election data.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

from midterms26.conformal import apply as conformal_apply
from midterms26.context import RunContext
from midterms26.features import assemble as features_assemble
from midterms26.geo import reaggregate
from midterms26.logging import get_logger
from midterms26.models import bayes, stack, tabpfn_member
from midterms26.publish import emit
from midterms26.simulate import copula
from midterms26.warehouse import connect, init_schema

log = get_logger("demo")

HISTORICAL_CYCLES = (2018, 2022)
LIVE_CYCLE = 2026
STATES = ("CA", "TX", "NY", "FL", "OH", "PA", "MI", "GA")
DISTRICTS_PER_STATE = 5  # -> 40 House races per cycle
DEFAULT_CUTOFF = date(2026, 10, 15)


def _national(cycle: int) -> float:
    """A fixed 'true' national environment (D%-R%) per cycle."""
    return {2018: 8.0, 2022: -2.0, 2026: 3.0}.get(cycle, 0.0)


def _seed(conn: object, seed: int) -> None:
    rng = random.Random(seed)

    # Stable per-seat partisanship so a district drifts consistently across cycles.
    pvi = {
        (st, f"{d:02d}"): rng.uniform(-25, 25)
        for st in STATES
        for d in range(1, DISTRICTS_PER_STATE + 1)
    }
    incumbent = {seat: rng.choice(["D", "R", None]) for seat in pvi}

    races, results, polls, ratings, finance, pres = [], [], [], [], [], []
    for cycle in (*HISTORICAL_CYCLES, LIVE_CYCLE):
        nat = _national(cycle)
        eday = date(cycle, 11, 3)
        for (st, dist), seat_pvi in pvi.items():
            rid = f"{cycle}-HOUSE-{st}-{dist}"
            inc = incumbent[(st, dist)]
            bump = 4.0 if inc == "D" else -4.0 if inc == "R" else 0.0
            margin = nat + seat_pvi + bump + rng.gauss(0, 4.0)
            is_live = cycle == LIVE_CYCLE
            races.append((rid, cycle, "HOUSE", st, dist, False, False, 0, inc, inc is None, eday))
            if not is_live:
                results.append((rid, cycle, "HOUSE", st, dist, margin, "SYNTH"))
            # A few as-of-dated polls before the election.
            for k in range(rng.randint(0, 4)):
                as_of = eday - timedelta(days=30 + 15 * k)
                polls.append((f"{rid}-p{k}", rid, as_of, margin + rng.gauss(0, 3.0)))
            ratings.append((rid, "SYNTH", eday - timedelta(days=20), max(-3, min(3, margin / 8))))
            finance.append((rid, f"{rid}-cand", eday - timedelta(days=40), rng.uniform(0.1, 0.6)))

    # Presidential-by-CD rows (PVI signal) for the two most recent pres years.
    for pres_year, natp in ((2020, 4.5), (2024, -1.5)):
        for (st, dist), seat_pvi in pvi.items():
            pres.append(
                (st, dist, "enacted", pres_year, natp + seat_pvi, date(pres_year + 1, 1, 6))
            )

    conn.executemany(  # type: ignore[attr-defined]
        "INSERT INTO races (race_id, cycle, office, state, district, is_special, "
        "is_uncontested, plan_generation, incumbent_party, is_open_seat, election_date) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        races,
    )
    conn.executemany(  # type: ignore[attr-defined]
        "INSERT INTO results_history (race_id, cycle, office, state, district, "
        "two_party_margin, source) VALUES (?,?,?,?,?,?,?)",
        results,
    )
    conn.executemany(  # type: ignore[attr-defined]
        "INSERT INTO polls (poll_id, race_id, as_of, margin) VALUES (?,?,?,?)", polls
    )
    conn.executemany(  # type: ignore[attr-defined]
        "INSERT INTO ratings (race_id, source, as_of, rating_numeric) VALUES (?,?,?,?)", ratings
    )
    conn.executemany(  # type: ignore[attr-defined]
        "INSERT INTO fec_finance (race_id, candidate_id, as_of, small_dollar_share, "
        "report_quarter) VALUES (?,?,?,?,'2026Q2')",
        finance,
    )
    conn.executemany(  # type: ignore[attr-defined]
        "INSERT INTO pres_results_by_district (state, district, plan_label, pres_year, "
        "two_party_margin, as_of) VALUES (?,?,?,?,?,?)",
        pres,
    )
    # National environment indicators (as-of-dated).
    natl = []
    for cycle in (*HISTORICAL_CYCLES, LIVE_CYCLE):
        for months_before in (2, 1):
            as_of = date(cycle, 11, 3) - timedelta(days=30 * months_before)
            natl.append(
                ("GENERIC_BALLOT", as_of, _national(cycle) + random.Random(cycle).gauss(0, 1))
            )
            natl.append(("APPROVAL", as_of, 44.0 + _national(cycle) / 2))
    conn.executemany(  # type: ignore[attr-defined]
        "INSERT INTO national_indicators (series_id, as_of, value) VALUES (?,?,?)", natl
    )


def run_demo(db_path: Path, *, cutoff: date = DEFAULT_CUTOFF, seed: int = 0) -> Path:
    """Seed a fresh warehouse and run the full modeling spine; return the site dir."""
    if db_path.exists():
        db_path.unlink()
    with connect(db_path) as conn:
        init_schema(conn)
        _seed(conn, seed)

    ctx = RunContext(db_path=db_path, cutoff_date=cutoff, do_loco=True)
    for name, module in (
        ("geo", reaggregate),
        ("features", features_assemble),
        ("bayes", bayes),
        ("tabpfn", tabpfn_member),
        ("stack", stack),
        ("conformal", conformal_apply),
        ("copula", copula),
        ("publish", emit),
    ):
        result = module.run(ctx)
        log.info("demo.step", stage=name, rows=result.rows, detail=result.detail)
    return db_path.parent / "site"
