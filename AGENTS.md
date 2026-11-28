# AGENTS.md

factortool fetches composite numbers from FactorDB or mersenne.ca, factors them
with external tools (YAFU, optionally CADO-NFS), and submits the results. The
project follows the [norms
standards](https://github.com/aexoden/norms/blob/main/STANDARDS.md), which the
`norms-verify` hook checks; read them for conventions not covered here.

## Commands

Python and dependencies are managed by uv (Python version pinned in
`.python-version`). These match the prek hooks (`prek.toml`) and CI:

```sh
prek install                               # once per clone: the pre-commit and commit-msg hooks
uv sync --locked                           # install the locked dependencies, as CI does
uv lock --check                            # uv.lock is in sync with pyproject.toml
uv run ruff format --check
uv run ruff check
uv run mypy .
uv run basedpyright
uv run pytest --cov=src/factortool --cov-report=term-missing
uv run pytest tests/test_engine.py -k <name>  # one file, or tests matching a name
prek run --all-files                       # every pre-commit hook, incl. reuse, markdownlint, norms
```

Both type checkers must pass, and the code is written to satisfy each: no
`Any`, every class attribute annotated, `@override` on overrides, and unused
call results discarded with `_ =`. Lint suppressions use Ruff's rule-name form,
e.g. `# ruff: ignore[too-many-arguments] (reason)`. Docstrings use Google style
and also document exceptions that callees raise through a function.

CI runs the tests on Linux, Windows and macOS: signal handling, file locking
and killing external processes are platform-specific (see `interrupt.py`,
`lock.py`, and the Windows job-object code in `tools.py`).

## Conventions

- Commits follow Conventional Commits (`feat`, `fix`, `perf`, `refactor`,
  `style`, `test`, `docs`, `build`, `ci`, `revert`, `chore`), enforced by the
  `commit-msg` hooks. Subjects are lowercase imperative, ideally within 50
  characters and never over 72, e.g.
  `fix: retry FactorDB's transient 405 responses`. Wrap the body at 72.
  Commits written with an AI agent carry a `Co-Authored-By` trailer naming it.
- Notable changes get an entry under `[Unreleased]` in `CHANGELOG.md` in the
  same commit (Keep a Changelog). Write the section as if it were written at
  release time, for its readers rather than as a commit log: changes that
  together make one difference are one entry, and a change reverted before the
  release isn't mentioned, so remove its entry. An entry for a breaking change
  (to the options, configuration, exit statuses or state files) starts with
  `**Breaking:**` and says briefly how to upgrade, and its commit marks the
  type with `!`, as in `feat!: drop the <name> setting`.
- Source files start with SPDX headers giving the MIT license and the
  copyright holder with the year the file was created, in the form existing
  files use but with the current year for a new file. Files that can't carry
  them, such as Markdown, are annotated in `REUSE.toml`, whose later
  annotations give a year to files created after the first. The `reuse` hook
  checks that headers are present, not that the year is right.
- User-facing behavior (options, exit statuses, state handling) is documented
  in `README.md`; keep it in step with behavior changes. `TODO.md` holds
  planned work.
- Settings are a closed set: an unknown key is an error, and removed settings
  are rejected with a message saying what to do instead
  (`Config.reject_removed_settings`). `config.dist.json` holds the defaults.

## Architecture

factortool runs **one batch per invocation**; its exit status
(`exit_status.py`) tells a wrapping shell loop whether it is safe to run again.

**Run flow** (`cli/main.py`): parse args → `load_config` with CLI overrides →
validate args and tools → `lock_instance` (filelock on state and work dirs) →
`run_session` → `run_batch`. A batch: `BatchController` picks a size (adapted
toward `--target_duration`, persisted per backend/mode/threads/digit
range/skip count) → `acquire_numbers` (retained work, then a fetch) →
`FactorEngine.run` → cleanup steps through `Cleanup.run`, each attempted even
if another fails (results file, stats, tool state). `backend.close()` runs
last, as it flushes pending submissions.

**Engine** (`engine.py`): two modes. `yafu` runs YAFU's `factor()` on each
number. `standard` works breadth-first across the whole batch: trial factoring
→ rho → P-1 (concurrent across numbers) → ECM level by level → a final method
per remaining number (SIQS, YAFU NFS or CADO-NFS), grouped by the method chosen
up front. Interrupts, time limits and assignment expiry are checked between
steps.

**Number** (`number.py`): holds `prime_factors` and `composite_factors` for one
input and runs each method over its remaining composites. It reports progress
through the `ResultSink` protocol (`checkpoint` for partial progress, `submit`
when complete), which backends implement; this keeps `number` from importing
the backends. `FinalMethodNeeded` lets ECM jump straight to a final method to
gather statistics, and a failed final method falls back to the other eligible
ones.

**Statistics** (`stats.py`): `FactoringStats` records time and success per
method, keyed by digit count and thread count, and from them decides the ECM
cutoffs (`get_ecm_cutoffs`) and the fastest final method (`get_final_method`;
a method with no data is run first to collect a sample). The `analyzer` CLI
reports on the same data.

**External tools** (`tools.py`): `run_tool` runs YAFU and CADO-NFS in a
temporary directory under `work_path` (`util.get_work_dir`, which copies and
rewrites `yafu.ini`). It tracks running processes so `abandon_tools` can kill
their whole trees (a Windows job object, or a POSIX process group), enforces
deadlines (`tool_deadline`), and warns about long runs. `tool_failures` is a
module-level singleton counting consecutive failures per method across runs;
three in a row raise `ToolError` (exit 4). `tests/conftest.py` resets it for
every test.

**Backends** (`backends/`): the `Backend` protocol plus `BaseBackend`, which
implements fetching, resuming retained work, and a background submission
worker thread. Concrete backends (`factordb.py` via JSON-RPC, `mersenne_ca.py`
via the assignment service) only describe their service, largely through
class attributes such as `assigns_work`, `accepts_partial_results`,
`submit_batch_size` and `transient_status_codes`. `factory.create_backend`
picks one. All HTTP goes through `http.HttpClient` (rate limits persisted
across runs, retry with backoff, `PermanentHttpError` for refusals → exit 5).

**Persistent state** (under `state_path`; paths are `Config` properties):
pending submissions as an append-only journal with compaction
(`submissions.py`), mersenne.ca assignments (`assignments.py`), statistics,
batch-size state, and tool failure streaks. Files are rewritten atomically via
`util.safe_write`. A result is recorded as soon as it's ready and removed only
once the backend accepts or rejects it, so nothing is lost if a run dies.

**Interrupts** (`interrupt.py`): `InterruptState` escalates with each signal.
Level 1 stops fetching, level 2 stops starting factorizations, level 3 abandons
the running tool (raising `Interrupted` inside `abortable`), and one more exits
at once. SIGTERM and SIGHUP act as level 3. Code checks `stop_fetching` and
`stop_factoring`, and waits with `interrupts.wait` rather than sleeping.

## Testing Notes

- `tests/helpers.py` provides `make_config` (a real validated `Config`),
  `make_number`, and `fake_time`. Tests replace a module's `time` binding with
  `fake_time(...)` instead of patching the `time` module, which would affect
  every thread, pytest's included.
- Tests may access private members (allowed by the Ruff and basedpyright
  settings for `tests/`).
