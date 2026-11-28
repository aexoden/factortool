# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A mersenne.ca backend, which works composites assigned by the mersenne.ca
  Aliquot composite service. Choose it with the new `backend` setting (or
  `--backend`), set `gimps_login`, and pace its requests with
  `mersenne_ca_request_interval`. It requires `--max_digits`, submits only
  complete factorizations, and keeps its assignments in the state directory so
  that a later run can resume them, without repeating the work already done on
  their remaining composites in `standard` mode. An assignment is worked until
  it expires, an hour after it was fetched, when any tool still running on it is
  abandoned. `--no_new_work` works only the retained assignments without
  fetching more.
- YAFU's built-in NFS as a final factoring method, enabled with `use_nfs_yafu`.
- `--max_digits` limits the size of fetched composites, and `--max_threads`,
  `--backend`, `--factoring_mode` and `--log_level` override the corresponding
  settings for a single run. The analyzer also accepts `--max_threads` and
  `--log_level`.
- `--time_limit` sets when no more factorizations are started, and
  `--hard_time_limit` sets when the running ones are abandoned.
- `--version`, with the version taken from git when the package is built.
- Results are recorded in the state directory as soon as they are ready. One
  that fails to submit five times, or is still unsent when a run ends, is
  submitted at the start of the next run. Unsent FactorDB results are discarded
  after 30 days.
- FactorDB receives partial factorizations when a run ends after finding some
  but not all factors of a number, and submissions are sent in batches.
- Rate limits a service imposes are respected across runs.
- YAFU and CADO-NFS each run in a temporary directory under the new `work_path`
  setting, which is cleaned up afterwards, so their working files no longer
  collect beside YAFU or under the system temporary directory. A copy of
  `yafu.ini` (the one next to YAFU, or `yafu_ini_path`) is placed in each.
- Requests carry a User-Agent naming factortool, its version and the
  configured account, which the `user_agent` setting can replace.
- A second instance using the same state or working directory exits with
  status 3 rather than running alongside the first.
- Arguments, settings and tool paths are checked before any work is fetched.
- When the backend has no work to hand out, the fetch is retried with backoff.
- A YAFU or CADO-NFS run lasting more than half an hour is noted in the log,
  and again every half hour until it ends.
- The analyzer shows the optimal and current ECM cutoffs.

### Changed

- **Breaking:** FactorDB is now reached through its JSON-RPC API.
  `factordb_username` and `factordb_password` are replaced by
  `factordb_api_token`, found on your FactorDB account page, and
  `factordb_response_path` and `factordb_session_path` are gone. Remove the old
  settings and set the token.
- **Breaking:** Settings not given in `config.json` take the defaults in
  `config.dist.json` (renamed from `config.json.dist`), and an unrecognized
  setting is now an error. Only `backend` (new, and required: add
  `"backend": "factordb"` to keep working as before), `max_threads` and
  `yafu_path` must be set. Starting again from `config.dist.json` and copying
  over your values is the easiest way to upgrade.
- **Breaking:** `factordb_cooldown_period` is renamed
  `factordb_request_interval`, and is now the minimum time between any two
  requests, retries included, rather than the delay before a retry. Rename the
  setting in `config.json`.
- **Breaking:** State files now live in the directory set by `state_path`
  (`state/` by default), with fixed names, and `stats_path` and
  `batch_state_path` are rejected. NFS statistics are now kept separately for
  CADO-NFS and YAFU NFS, so an existing `stats.json` is reported as
  incompatible. To keep it, rename its `nfs` key to `nfs_cado`, add
  `"schema_version": 1`, and move it into the state directory; otherwise,
  delete it to start over. Remove the old settings. A batch state file from an
  earlier version is discarded and the batch size measured afresh.
- **Breaking:** CADO-NFS is used only when `use_nfs_cado` is `true`, and
  `cado_nfs_path` is needed only then. Set it to keep using CADO-NFS.
- **Breaking:** Log messages go to standard error rather than standard output,
  and the level is set by the new `log_level` setting (or `--log_level`), which
  defaults to `INFO` rather than `DEBUG`. `LOGURU_LEVEL` no longer has an
  effect.
- **Breaking:** Exit statuses are renumbered so that only 0 means it is safe to
  run again. A run that reaches its time limit exits 0 (was 3), an interrupt
  exits 130 (was 2), and invalid arguments or configuration exit 2 (was 1).
  Repeated YAFU or CADO-NFS failures exit 4 (YAFU was 5), a backend refusing a
  request exits 5 and is contacted no further, and the other statuses are
  described in `README.md`. In the analyzer, invalid input now exits 2 (was 1)
  and missing data exits 1 (was 2). A shell loop that checked for status 2
  would now stop on a configuration error but carry on after Ctrl-C, so loop
  while `factortool` succeeds instead, as `README.md` shows.
- Repeated interrupts escalate: the first stops fetching (and ends any wait for
  work at once), the second stops starting factorizations, and the third
  abandons the running tool, killing its helper processes too. Each submits the
  results in hand before exiting. SIGTERM, SIGHUP and, on Windows, Ctrl-Break
  act as the third, and one more signal after that exits immediately.
- A YAFU or CADO-NFS failure on one number is logged and the batch continues,
  and a failed final method is retried with the other eligible ones. Only three
  consecutive failures of the same method, counted across runs, end the run.
- The time limit counts from the start of factoring rather than from startup,
  and only a `factordb` run with an automatic batch size is limited to twice
  the target duration by default. Any other run, including one with
  `--batch_size`, now has no limit unless `--time_limit` is given.
- The automatic batch size is measured separately for each backend, factoring
  mode, thread count, digit range and skip count, and each keeps its own
  measurement, so a run with different settings no longer starts the usual
  measurement over.
- ECM cutoffs are reevaluated during a run as the statistics change, rather
  than fixed at its start.
- The results file lists only the numbers a run found new factors of, in
  ascending order, and is not written when there are none.
- Every logged number shows its digit count.
- Primality testing is faster.
- factortool requires Python 3.15.2, which uv installs automatically.

### Fixed

- ECM amounts are chosen better: each remaining composite of a number does the
  ECM its own size calls for, rather than all of them doing the amount the
  largest calls for (which also skewed the statistics for the smaller sizes),
  cofactors whose digit count was never seen as an input number no longer do
  far too much ECM, cofactors below the CADO-NFS minimum no longer run ECM until
  they are factored instead of switching to SIQS, ECM is skipped entirely when
  the statistics say none is worthwhile, and gaps in the ECM statistics are
  handled.
- The automatic batch size no longer stays at one or two numbers for many runs
  after its measurement starts over. Every number factored now counts equally
  toward the measured rate, whatever the size of its batch, and the batch grows
  to the target duration, at most doubling each run, so that an unusually
  quick first number can't lead to an oversized batch.
- Statistics updates from concurrent workers are no longer lost.
- Errors in the rho and P-1 workers are no longer silently ignored.
- External tools receive the full environment, less factortool's own Python
  virtual environment, rather than only `OMP_NUM_THREADS`, which is now set to
  the thread count the tool is allowed.
- FactorDB now receives the largest prime factor even when it occurs more than
  once.
- In `yafu` mode, an interrupt during the last factorization of a batch no
  longer lets the run exit 0.
- The summary no longer counts numbers that were never started, because of an
  interrupt or the time limit, as failures.
- A failure to write the results file no longer skips saving the statistics
  and submitting results: it is logged, the remaining steps still run, and the
  run exits 6.
- Numbers from the backend that are not composite are discarded instead of
  being worked.
- A `Retry-After` date in any HTTP-date format is understood, as GMT and
  regardless of locale, and permanent HTTP client errors are no longer
  retried.
- State files are replaced more reliably on Windows, and a failed write no
  longer leaves a temporary file behind.

## [0.1.0] - 2026-03-08

Initial public release of factortool.

### Added

- Multiple factoring methods: trial factoring, Pollard's rho, P-1, ECM, SIQS,
  and NFS (via CADO-NFS).
- Direct YAFU factoring mode as an alternative to the built-in breadth-first
  search.
- Automatic fetching of composite numbers from FactorDB and submission of
  results via the FactorDB API.
- FactorDB session management with configurable cooldown period.
- Adaptive batch sizing with configurable target duration.
- Statistics collection and analysis for ECM, SIQS, and NFS performance.
- Automatic determination of optimal ECM crossover threshold and SIQS/NFS
  decision based on gathered statistics.
- Analyzer tool for reviewing factorization statistics.
- Configurable time limit for batch runs.
- Graceful handling of interrupts (Ctrl-C), finishing the current factorization
  and submitting results before exiting.
- CLI options for minimum digit count, batch size, target duration, and skip
  count.
- Structured exit codes for configuration errors, interrupts, time limit
  exceeded, and unexpected failures.

[Unreleased]: https://github.com/aexoden/factortool/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aexoden/factortool/tree/v0.1.0
