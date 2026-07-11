"""DuckDB analytical warehouse: schema + connection helpers.

Single-file analytical DB at ``data/warehouse.duckdb``. All tables are created
idempotently so backfill / nightly runs can call :func:`init_schema` freely.

Design decisions encoded here (see project plan §0):
  * Canonical race id ``{cycle}-{office}-{state}-{district}``.
  * Every time-varying table carries an ``as_of`` DATE — the leakage guard in
    ``features/`` refuses to read rows with ``as_of > cutoff``.
  * ``predictions`` are keyed by ``plan_generation`` so a mid-cycle redraw
    produces a *new* immutable row rather than overwriting history.
  * Mondrian strata fields live on ``predictions`` so coverage can be audited
    per group in backtests.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path("data/warehouse.duckdb")

# Ordered so foreign-key-like references (races first) read naturally. DuckDB
# does not enforce FKs across all versions, so these are documentation-grade.
SCHEMA_STATEMENTS: tuple[str, ...] = (
    # -- Spine: one row per race-cycle -------------------------------------
    """
    CREATE TABLE IF NOT EXISTS races (
        race_id            TEXT PRIMARY KEY,   -- {cycle}-{office}-{state}-{district}
        cycle              INTEGER NOT NULL,
        office             TEXT    NOT NULL,   -- HOUSE | SENATE | GOV
        state              TEXT    NOT NULL,   -- USPS
        district           TEXT    NOT NULL,   -- e.g. '01', 'AL', 'SEN'
        is_special         BOOLEAN NOT NULL DEFAULT FALSE,
        is_uncontested     BOOLEAN NOT NULL DEFAULT FALSE,
        plan_generation    INTEGER NOT NULL DEFAULT 0,   -- which enacted plan
        incumbent_party    TEXT,               -- D | R | I | NULL (open)
        is_open_seat       BOOLEAN NOT NULL DEFAULT FALSE,
        election_date      DATE
    )
    """,
    # -- Historical returns (target source) --------------------------------
    """
    CREATE TABLE IF NOT EXISTS results_history (
        race_id            TEXT NOT NULL,
        cycle              INTEGER NOT NULL,
        office             TEXT NOT NULL,
        state              TEXT NOT NULL,
        district           TEXT NOT NULL,
        dem_votes          BIGINT,
        rep_votes          BIGINT,
        total_votes        BIGINT,
        two_party_margin   DOUBLE,             -- D% - R% (the target)
        source             TEXT,               -- e.g. 'MIT-EDSL'
        PRIMARY KEY (race_id)
    )
    """,
    # -- Polls (race-level, as-of-dated) -----------------------------------
    """
    CREATE TABLE IF NOT EXISTS polls (
        poll_id            TEXT PRIMARY KEY,
        race_id            TEXT NOT NULL,
        pollster_id        TEXT,
        field_start        DATE,
        field_end          DATE,
        as_of              DATE NOT NULL,       -- release date; leakage key
        mode               TEXT,                -- online | phone | mixed | ivr
        population         TEXT,                -- LV | RV | A
        sample_n           INTEGER,
        dem_share          DOUBLE,
        rep_share          DOUBLE,
        margin             DOUBLE               -- D - R
    )
    """,
    # -- Pollster registry + hierarchical skill/house-effect outputs -------
    """
    CREATE TABLE IF NOT EXISTS pollsters (
        pollster_id        TEXT PRIMARY KEY,
        name               TEXT NOT NULL,
        methodology        TEXT,
        as_of              DATE NOT NULL,       -- rolling scheme: skill as-of
        skill_shrunk       DOUBLE,              -- hierarchical Bayes estimate
        house_effect       DOUBLE
    )
    """,
    # -- FEC individual-donor finance (quarterly snapshots) ----------------
    """
    CREATE TABLE IF NOT EXISTS fec_finance (
        race_id            TEXT NOT NULL,
        candidate_id       TEXT NOT NULL,
        party              TEXT,
        report_quarter     TEXT,                -- e.g. '2026Q2'
        as_of              DATE NOT NULL,       -- reported-through date
        individual_donations DOUBLE,            -- individual-donor total ONLY
        small_dollar_share DOUBLE,
        PRIMARY KEY (race_id, candidate_id, report_quarter)
    )
    """,
    # -- FEC Schedule E independent expenditures (Phase 1b) ----------------
    # "Insider revealed preference": where parties/PACs actually spend late
    # money. One row per itemized transaction; as_of = expenditure date.
    """
    CREATE TABLE IF NOT EXISTS fec_ie (
        ie_id              TEXT PRIMARY KEY,    -- FEC sub_id (transaction key)
        race_id            TEXT NOT NULL,
        candidate_id       TEXT NOT NULL,       -- the targeted candidate
        committee_id       TEXT,                -- the spending committee
        support_oppose     TEXT NOT NULL,       -- S | O
        amount             DOUBLE NOT NULL,
        as_of              DATE NOT NULL        -- expenditure date; leakage key
    )
    """,
    # -- Candidate ideology from DIME CFscores (Phase 1b) ------------------
    # as_of = Dec 31 of the score's receipts cycle, so a same-cycle score can
    # never pass a pre-election cutoff (DIME scores use full-cycle receipts).
    """
    CREATE TABLE IF NOT EXISTS candidate_ideology (
        candidate_id       TEXT NOT NULL,       -- FEC id (crosswalked)
        cycle              INTEGER NOT NULL,    -- receipts window of the score
        as_of              DATE NOT NULL,
        cfscore            DOUBLE NOT NULL,
        cfscore_dyn        DOUBLE,              -- dynamic (per-cycle) variant
        n_donors           INTEGER,
        source             TEXT,                -- 'DIME-v4' | 'derived-fec'
        PRIMARY KEY (candidate_id, cycle)
    )
    """,
    # -- Expert ratings snapshots (as-of-dated) ----------------------------
    """
    CREATE TABLE IF NOT EXISTS ratings (
        race_id            TEXT NOT NULL,
        source             TEXT NOT NULL,       -- Cook | InsideElections | Sabato
        as_of              DATE NOT NULL,
        rating             TEXT,                -- SafeD..TossUp..SafeR
        rating_numeric     DOUBLE,              -- signed scale, + = D
        PRIMARY KEY (race_id, source, as_of)
    )
    """,
    # -- Special elections & swing index -----------------------------------
    """
    CREATE TABLE IF NOT EXISTS specials (
        special_id         TEXT PRIMARY KEY,
        race_id            TEXT,
        cycle              INTEGER NOT NULL,
        state              TEXT,
        district           TEXT,
        election_date      DATE NOT NULL,
        seat_pvi           DOUBLE,
        result_margin      DOUBLE,              -- D - R
        turnout            BIGINT,
        overperformance    DOUBLE               -- result vs seat partisanship
    )
    """,
    # -- Redistricting-native geo outputs (per district per plan gen) ------
    """
    CREATE TABLE IF NOT EXISTS districts_geo (
        race_id                       TEXT NOT NULL,
        state                         TEXT NOT NULL,
        district                      TEXT NOT NULL,
        plan_generation               INTEGER NOT NULL DEFAULT 0,
        plan_enacted_date             DATE,
        is_new_seat                   BOOLEAN NOT NULL DEFAULT FALSE,
        pvi_reaggregated              DOUBLE,   -- recomputed on new lines
        pvi_trend_2016_2024           DOUBLE,
        incumbent_constituency_overlap DOUBLE,  -- 0..1, scales incumbency
        reaggregation_error           DOUBLE,   -- vs official statewide (<0.1%)
        PRIMARY KEY (race_id, plan_generation)
    )
    """,
    # -- ACS 5-yr demographics, tidy (Phase 1b) -----------------------------
    # Stored at source geography (block group / tract / CD); the Phase 2 geo
    # pipeline aggregates onto enacted 2026 lines. as_of = public release date.
    """
    CREATE TABLE IF NOT EXISTS acs_demographics (
        geoid              TEXT NOT NULL,       -- census GEOID at geo_level
        geo_level          TEXT NOT NULL,       -- BG | TRACT | CD
        vintage            TEXT NOT NULL,       -- e.g. 'ACS5-2023'
        as_of              DATE NOT NULL,       -- release date; leakage key
        variable           TEXT NOT NULL,       -- e.g. 'pct_ba_plus'
        value              DOUBLE NOT NULL,
        PRIMARY KEY (geoid, vintage, variable)
    )
    """,
    # -- Presidential results by congressional district (Phase 1b) ---------
    # Daily Kos Elections crosswalks; PVI cross-check for redrawn lines.
    # as_of = certification proxy (Jan 6 following the presidential year).
    """
    CREATE TABLE IF NOT EXISTS pres_results_by_district (
        state              TEXT NOT NULL,
        district           TEXT NOT NULL,       -- canonical 2-digit token
        plan_label         TEXT NOT NULL,       -- which district lines
        pres_year          INTEGER NOT NULL,
        dem_share          DOUBLE,              -- two-party D share, pct
        rep_share          DOUBLE,
        two_party_margin   DOUBLE,              -- D% - R%
        as_of              DATE NOT NULL,       -- leakage key
        source             TEXT,                -- e.g. 'DailyKos'
        PRIMARY KEY (state, district, plan_label, pres_year)
    )
    """,
    # -- Assembled feature matrix (as-of-dated, leakage-guarded) -----------
    """
    CREATE TABLE IF NOT EXISTS feature_matrix (
        race_id            TEXT NOT NULL,
        cutoff_date        DATE NOT NULL,       -- as-of the whole feature row
        plan_generation    INTEGER NOT NULL DEFAULT 0,
        features           JSON NOT NULL,       -- name -> value; schema-checked
        target_margin      DOUBLE,              -- NULL for future/live races
        is_imputed_uncontested BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY (race_id, cutoff_date, plan_generation)
    )
    """,
    # -- Per-member predictive quantile grids (Phase 4) --------------------
    # One row per (race, member, fold). ``fold`` is 'live' for the in-sample
    # refit that predicts the target races, or a held-out cycle (e.g. '2018')
    # for the LOCO out-of-fold predictions the stack is fit on. ``quantiles`` is
    # a JSON map level -> margin over ``QUANTILE_LEVELS``.
    """
    CREATE TABLE IF NOT EXISTS member_predictions (
        race_id            TEXT NOT NULL,
        cutoff_date        DATE NOT NULL,
        plan_generation    INTEGER NOT NULL DEFAULT 0,
        model_member       TEXT NOT NULL,       -- BAYES | TABPFN
        fold               TEXT NOT NULL,       -- 'live' | held-out cycle
        median_margin      DOUBLE,
        quantiles          JSON NOT NULL,       -- level -> value
        PRIMARY KEY (race_id, cutoff_date, plan_generation, model_member, fold)
    )
    """,
    # -- Bayesian latent-factor loadings for the copula (Phase 4) ----------
    # The hierarchical member's shared factors give an analytic race-correlation
    # matrix (replacing the old SHAP-correlation hack). Scale is absorbed into
    # the loadings, so Cov(i,j != i) = dot(loadings_i, loadings_j) and
    # Var(i) = dot(loadings_i, loadings_i) + idiosyncratic_sd**2.
    """
    CREATE TABLE IF NOT EXISTS latent_factors (
        race_id            TEXT NOT NULL,
        cutoff_date        DATE NOT NULL,
        plan_generation    INTEGER NOT NULL DEFAULT 0,
        loadings           JSON NOT NULL,       -- component -> scaled loading
        idiosyncratic_sd   DOUBLE NOT NULL,
        PRIMARY KEY (race_id, cutoff_date, plan_generation)
    )
    """,
    # -- Predictions (immutable per plan_generation + as_of) ---------------
    """
    CREATE TABLE IF NOT EXISTS predictions (
        race_id            TEXT NOT NULL,
        as_of              DATE NOT NULL,
        plan_generation    INTEGER NOT NULL DEFAULT 0,
        model_version      TEXT NOT NULL,
        median_margin      DOUBLE,
        -- conformal intervals at alpha in {0.5, 0.2, 0.1}
        lo_50 DOUBLE, hi_50 DOUBLE,
        lo_80 DOUBLE, hi_80 DOUBLE,
        lo_90 DOUBLE, hi_90 DOUBLE,
        win_prob_dem       DOUBLE,              -- from conformalized CDF
        mondrian_group      TEXT,               -- e.g. 'incR|polled3+|redrawn|HOUSE'
        abstain            BOOLEAN NOT NULL DEFAULT FALSE,
        abstain_reason     TEXT,                -- width>tau | bin<n_min | NULL
        PRIMARY KEY (race_id, as_of, plan_generation, model_version)
    )
    """,
    # -- Joint seat-distribution forecast from the copula simulator --------
    # One row per (as_of, plan_generation, office): the marginals are the per-race
    # conformalized CDFs, the correlation comes from the Bayesian latent factors.
    """
    CREATE TABLE IF NOT EXISTS seat_forecast (
        as_of              DATE NOT NULL,
        plan_generation    INTEGER NOT NULL DEFAULT 0,
        office             TEXT NOT NULL,
        n_races            INTEGER NOT NULL,
        n_draws            INTEGER NOT NULL,
        majority_threshold INTEGER NOT NULL,
        expected_dem_seats DOUBLE,
        p_dem_majority     DOUBLE,
        seats_p10          DOUBLE,
        seats_p50          DOUBLE,
        seats_p90          DOUBLE,
        histogram          JSON,                -- seat count -> probability
        PRIMARY KEY (as_of, plan_generation, office)
    )
    """,
    # -- National / economic indicator time series (as-of-dated) -----------
    # NOTE: not in the plan's enumerated table list, but the national-environment
    # feature family (FRED macro, presidential approval, generic ballot) needs a
    # home. One tidy row per (series, observation date).
    """
    CREATE TABLE IF NOT EXISTS national_indicators (
        series_id          TEXT NOT NULL,       -- e.g. 'FRED:UNRATE', 'APPROVAL'
        as_of              DATE NOT NULL,       -- observation date; leakage key
        value              DOUBLE,
        source             TEXT,
        PRIMARY KEY (series_id, as_of)
    )
    """,
    # -- Calibration audit trail (append-only) -----------------------------
    """
    CREATE TABLE IF NOT EXISTS calibration_log (
        run_id             TEXT NOT NULL,
        logged_at          TIMESTAMP NOT NULL,
        race_id            TEXT NOT NULL,
        as_of              DATE NOT NULL,
        model_version      TEXT NOT NULL,
        alpha              DOUBLE NOT NULL,
        interval_lo        DOUBLE,
        interval_hi        DOUBLE,
        mondrian_group     TEXT,
        realized_margin    DOUBLE,              -- NULL until outcome known
        covered            BOOLEAN,             -- NULL until scored
        PRIMARY KEY (run_id, race_id, alpha)
    )
    """,
)

# Tables that carry an ``as_of`` column — the leakage guard consults this list.
AS_OF_TABLES: frozenset[str] = frozenset(
    {
        "polls",
        "pollsters",
        "fec_finance",
        "fec_ie",
        "candidate_ideology",
        "ratings",
        "acs_demographics",
        "pres_results_by_district",
        "national_indicators",
    }
)

ALL_TABLES: tuple[str, ...] = (
    "races",
    "results_history",
    "polls",
    "pollsters",
    "fec_finance",
    "fec_ie",
    "candidate_ideology",
    "ratings",
    "specials",
    "acs_demographics",
    "pres_results_by_district",
    "districts_geo",
    "feature_matrix",
    "member_predictions",
    "latent_factors",
    "predictions",
    "seat_forecast",
    "national_indicators",
    "calibration_log",
)


@contextmanager
def connect(
    db_path: Path | str = DEFAULT_DB_PATH, *, read_only: bool = False
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a DuckDB connection, creating the parent directory if needed."""
    path = Path(db_path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path), read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all warehouse tables idempotently."""
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)


def existing_tables(conn: duckdb.DuckDBPyConnection) -> set[str]:
    """Return the set of table names currently present in the DB."""
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return {r[0] for r in rows}
