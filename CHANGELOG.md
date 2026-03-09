# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/aexoden/factortool/tree/v0.1.0
