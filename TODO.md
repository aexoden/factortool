# TODO

## Items

The following is a list of potential TODO items, roughly in my intended order of
implementation. There is no guarantee I will actually get to any of this.

* Make upgrades safer with regard to the config file. We should possibly set
  defaults for the ones that can be defaulted, and then reject unset ones that cannot.
* Investigate moving away from Tap for argument parsing. If keeping Tap, consider
  replacing underscores with hyphens in option names.
* Move state files into a directory. Consider changing config to only allow the
  directory, then using standardized names.
* Provide options to allow the user to disable the time limit or to tweak how long
  it is instead of always doing twice the target time.
* Analyzer needs to show all available ECM data, not just cut off at the first gap,
  as smaller digit sizes that were only cofactors may not have data starting at
  the lowest level.
* Investigate whether the time to factor estimate needs to account for finding all
  factors rather than just the first.
* Display digit counts even for short numbers (possibly except for trivially
  short, but probably more than 5 or so starts to become too much to tell at a glance).
* Potentially switch from the list_by_type endpoint to the download endpoint on
  FactorDB. The --random parameter reduces the need for offset to minimize
  overlap (though without knowing exactly how random works, it may not be as
  good as hoped--it'd be better if it did a random sample of all numbers of the
  requested digit size, but at least one spot made it seem like it just picks a
  random starting offset). The other caveat is that it only returns from a single
  digit level, so if we didnt' get enough work from one digit level, we'd need to
  request from the next. Another caveat is how we'd design this differently for
  a long-run/dashboard interface rather than a one-shot interface.
* Fix the time limit to only occur if target time is enabled.
* Altered submission strategy (especially for batchable services). Submit if
  either a) X seconds have passed, b) the current batch is full, or c) everything's
  done. Retain the submit spacing.
* Investigate integrating the looping process directly into the tool.
* Optional alternate buffer dashboard -- more useful once built-in looping (or
  equivalent) is added. Regardless of how this is done, need to retain the
  ability to do a specific batch size and quit.
* Revisit how factors are stored. Instead of storing each factor x times,
  potentially store it once with an exponent.
* Refactor the standard factoring method away from breadth-first. It doesn't
  really add anything, and if the program is aborted (either manually or because
  of an expired time limit), the ECM work on any remaining unfactored exponents
  is effectively lost as it will simply be repeated on another run. If done, it
  should be optional. I find breadth-first more asthetically pleasing, even if
  it's potentially more wasteful if aborted. Concerned users can simply use yafu
  directly.
* For the standard factoring method, revisit how the ECM curves and B1 values are
  determined. At the very least, if still using precomputed values, we need to
  consider differences if YAFU is internally using AVX-ECM or not.
* Consider exactly when the time limit being exceeded should terminate the
  program. (That is, should it abort as soon as possible or should it wait for
  the current factorization to finish?)
* Make the log level configurable.
* Refactor code to eliminate as many lint exceptions as possible.
* Dynamic batch sizes take too long to ramp up, especially because a sample size
  of 1 is heavily deweighted.
* Before working on a number, consider fetching any existing factors from FactorDB.
  We'll probably want to keep track of submission by factor at that point. That
  can go along with changing storage to factor and exponent. I'm not sure this
  makes tons of sense. It would make more sense to just fetch lower composites
  (which would already include cofactors of larger partially-factored numbers).
  Though I suppose it could be useful if someone mostly wants to work on, for
  example, 150 digit numbers.
* Keep allowing direct YAFU usage as many people will find that more useful, but
  because YAFU does have some bugs that can impede progress, I'd like to retain
  the "standard" self-managed factoring logic. Adding support for YAFU's ggnfs-
  based NFS would probably be good, and potentially its hybrid CADO/msieve. Both,
  at least on some of my machines, will occasionally fail for reasons I don't
  understand. (Only observed on non-AVX512 machines to this date.) (This is an
  old comment and may have changed in the meantime.)
* Improve error handling and fall back if YAFU fails for whatever reason.
* Improve the analyzer output. Output the current threshold to move away from ECM.
* Ensure that the analyzer always prints whatever data it has, rather than printing
  nothing. (This is an old note, and may already be partially or fully fixed.)
* It's best to avoid making SIQS/NFS decisions based on a single sample. The user
  may have been using their computer for something else and skewed the data.
  Going along with this, it would be good to add a special tune mode to collect
  the desired data in one action the user can control the timing of. As an
  additional consideration, it may be better to use theoretical expected ECM
  probabilities rather than calculated ones, which could be skewed by whatever
  weird collection composites FactorDB had (and how they were constructed). Some
  kind of measurement with error bars and only doing as many tests as needed to
  resolve which is fastest.
* Determine if there is a more sensible (statistically sound) method to determine
  the ECM cutoffs. Also ensure it's possible to choose "do no ECM at all". It may
  also be desirable to extend this to the other pre-ECM factoring methods. This
  may go along with dropping some of the ECM levels, since many of the early ones
  take about the same time and may not add a lot of value doing them all.
* Reconsider how to best handle fetching larger batches, especially with FactorDB
  returning a lot of 502 errors right now (which is more likely with larger batches).
* Add additional tests.
* Investigate making threads overridable on the command line.
* Add support for calculating and verifying Aliquot sequences.
* Support mersenne.ca's pretest mode (`composites_to_pretest` when fetching, and
  a `pretest_ratio` field when reporting), which asks for ECM-only pre-factoring
  without SIQS or NFS.
* Consider whether the mersenne.ca one-hour assignment window should influence
  batch sizing. Unfinished assignments are now carried across runs, so nothing is
  lost, but a batch taking much longer than an hour will have released its later
  composites back to the pool before they are reported.
* YAFU reads stdin as a batch file whenever stdin isn't a terminal (cron, systemd,
  `< /dev/null`, a heredoc), ignoring the expression passed on the command line.
  It then prints no factors, and `_factor_generic` removes the composite without
  adding any factors, so the number is treated as factored. Affects every YAFU
  call. Possible fixes: pass the expression to YAFU on stdin, and/or treat an
  empty factor list as a failure.
* YAFU calls other than NFS run with an environment containing only
  `OMP_NUM_THREADS=1` (no `PATH`, `HOME`, etc.). YAFU NFS now gets the full
  environment plus that variable. Consider doing the same everywhere.
* The final factoring stages are now grouped per method (SIQS, YAFU NFS,
  CADO-NFS), and each number only goes through its own method's stage. The old
  final NFS stage picked up everything still unfactored, so a number SIQS left
  unfinished would be retried with NFS; that no longer happens. Possibly part of
  the YAFU error handling/fallback item above.
* `YAFU_NFS_MIN_DIGITS` (85) is based on one machine using the GGNFS sievers,
  where poly selection stalled at 80 digits and below but worked at 82+. YAFU's
  hybrid CADO/msieve mode (`cadoMsieve`) hasn't been tested at all, and may have
  a different floor (or different failure modes). Consider making the floor
  configurable if it turns out to vary.
