# TODO

The following is a list of bug fixes and potential improvements, roughly in the
intended order of implementation. There is no guarantee that an item on this list
will ever actually be implemented. Some of the earlier ones may be obsoleted by
later ones.

## Code/Architectural Improvements

- Consider setting the default factoring mode to yafu. In most cases, it makes
  more sense. Should do some performance comparisons between standard and yafu
  (and maybe a depth-first standard--though that has multithreading issues to
  work out) and see if there are regimes where one is clearly better than the
  other. I expect yafu to be better optimized in any case.
  Do this comparison before investing further in standard mode: yafu mode
  ignores the progress saved with retained assignments, so as the default it
  would forgo most of that benefit unless the `-work` mapping (see the ECM curve
  item) is done.
- Consider moving the state files (assignments, pending submissions,
  statistics, batch and tool state) into a single SQLite database. Transactions
  would replace the journal's compaction and the atomic file rewrites, and
  storing individual runs rather than running totals would allow the statistics
  to report their variance.
  Design the schema together with the next few items, as each changes what is
  stored: how factors are stored (once with an exponent?), what the statistics
  must record for robust final method decisions and a revised ECM cutoff
  structure (individual runs, with timestamps if the model is to follow the
  current distribution of numbers), and what a saved ECM level means if the
  curve schedule changes. A saved `ecm_level` is only meaningful against
  `ECM_CURVES`, and the statistics are keyed by level, so the schema should
  carry a schedule version or similar. Also consider keeping progress for
  FactorDB work, which is currently lost when a run ends unfinished.
- Revisit how factors are stored. Instead of storing each factor N times,
  potentially store it once with an exponent. This is an old item and I don't
  recall the motivation. Investigation of the impacts (both positive and
  negative) would be needed before spending any time on it.
  Decide this as part of the SQLite schema design rather than after it, to avoid
  migrating twice.
- For the standard factoring method, reconsider the ECM curves and the B1 values.
  This may want to adjust for whether YAFU is using AVX-ECM or not. The lowest
  levels of ECM are probably not adding tons of value on their own, but use data
  to guide this decision.
  Changing `ECM_CURVES` changes the meaning of the ECM levels saved with
  retained assignments and of the statistics recorded per level, and removing
  levels makes saved progress fail validation (`le=ECM_MAX_LEVEL`), discarding
  the whole assignments file, so this needs a migration (best done with the
  SQLite move). If the levels became true t-levels, yafu mode could pass a
  composite's saved ECM level to YAFU's `-work` option instead of ignoring it.
- We should probably avoid making final factoring method decisions on single
  samples. Transient CPU usage could skew the data. It may be desirable to add a
  dedicated tuning mode that the user can run at a dedicated time to avoid random
  spikes. (This is a lot harder to do with ECM, however.) In any case, more
  statistically robust method of determining the error bars and the need for
  more data to resolve the winner is desirable.
- Going along with this, revise the entire structure of the ECM cutoffs. It
  should be possible to completely skip TF, rho, P-1 or ECM if it turns out
  there's no measurable benefit. The analyzer should be adjusted to treat these
  more similarly. (May go along with adjusting the ECM curve schedule.) It's an
  open question whether to target theoretical numbers for truly random numbers
  or a more weighted distribution. If we use actual data, it would need to be
  more dynamic, otherwise we could be using bad data from a different
  distribution. Could try to see if the numbers pulled from the services have
  any sort of consistent distribution, but somehow I doubt this to be consistent.
  If going with theoretical numbers, I have no idea how to calculate the odds of
  the various stages finding factors (e.g. P-1's probability assuming TF and rho
  have been done). Also may need to consider how multithreading affects any of
  these numbers. (You don't want to compare a rho run at 1 thread with a SIQS run
  at 16 threads without taking into account that 16 rho runs can take place at the
  same time--obviously this only applies if you have enough numbers to saturate
  the threads).
- Investigate whether the time to factor estimates should account for finding all
  factors rather than just the first. The original logic was that all we care
  about is finding the first factor. The cofactor then essentially goes into the
  queue as a new number to factor with its own expected duration, copying over
  any existing work that was done on the original number in its own history.
- Determine how multithreaded YAFU's implementation of SIQS is (may vary by
  digit count), and consider threading it ourselves like with rho and P-1.
  Should probably check how well YAFU NFS and CADO-NFS use threads as well.
- Why is `max_siqs_digits` even an option? Is there a historical reason for it?
  At the same time, investigate if there are any other superfluous options.
  Unrecognized settings are now an error, so any setting that is removed should
  be rejected with a message explaining what to do instead, as the removed state
  file settings are.
- Consider whether the mersenne.ca one-hour assignment window should influence
  batch sizing. As long as the target duration is well below the window (e.g.
  the default of ten minutes), it's fine, but perhaps a warning if the user
  tries to use too large a duration such that their assignments risk expiring.
  The goal is to never take more work than can be finished within the window, as
  nothing else stops an assignment from being started too late to complete. Any
  result for an expired assignment, partial or complete, is discarded.
- Investigate moving away from Tap (Typed Argument Parser) for argument parsing.
  The obvious alternatives are click and typer, both of which are far more
  active projects than Tap, but this list isn't exhaustive. If opting to keep
  Tap, at least standardize on using hyphens rather than underscores in option
  names. Whatever the parser, invalid arguments should probably keep exiting
  with status 2 (`ExitStatus.INVALID_INPUT`), as click and argparse both do by
  default. A parser with range types (such as click's `IntRange` and
  `FloatRange`) could replace the hand-written checks in
  `validate_argument_ranges`.
- Investigate switching from the FactorDB list_by_type endpoint to the download
  endpoint. The offset option would no longer be supported, but that was largely
  intended to minimize the risk of overlapping work, and the `random` option
  provides much of that benefit. (Though it's not quite as good, as download is
  restricted to a single digit count under random mode, and two people working
  on the same digit count will eventually converge.) The primary advantage is a
  much larger limit per request (50000 versus 1000). Since random mode only
  fetches a single digit count, additional logic would be necessary to pull in
  additional digit counts when necessary. A more complex example of how random
  is not quite as good is two people working at 70+ digits. If each has a
  different offset, they're unlikely to overlap, even if their wavefront expands
  well beyond 70 digits. With random mode, they'd both work 70 digits until empty
  and then move to 71, providing more overlap opportunities on small bins.
- Refactor to eliminate as many linting exceptions as possible.
- Documentation: document `--version` and the configuration settings in the
  README, and reconcile the README's Ctrl-C paragraph with the escalation list
  below it.
- Test the differences between YAFU's NFS options. Determine if it's practical
  to enable both, and characterize their individual performance and limitations.
  `YAFU_NFS_MIN_DIGITS` (85) is based on one machine using the GGNFS sievers,
  where poly selection stalled at 80 digits and below but worked at 82+. YAFU's
  hybrid CADO/msieve mode (`cadoMsieve`) hasn't been tested at all, and may have
  a different floor (or different failure modes). Consider making the floor
  configurable if it turns out to vary.
- Increased test coverage in any remaining gaps.

## Major Features (or deferred minor features)

- Remove the need for the external looping script, and allow the tool to be used
  in continuous mode. Whether this is implemented as a simple outer loop or via
  a different mechanism that refills its queue when low remains an open question.
  The current one-shot mode should remain an option in any case. The choice here
  may go along with the dashboard item. Revisit the submission policy at this
  point. There's no real need to cap submission attempts. It's not like quitting
  and starting the next run is going to make contacting the service more
  successful. In continuous mode, we'd just want contact attempts to continue
  anyway. The standard backoff capped at an hour is probably sufficient.
  Once a backend refuses a submission request, nothing further is submitted or
  fetched for the rest of the run, and the results are left for "the next run",
  which will need a retry-later policy of its own. Do this after the SQLite
  move: a long-running process writing progress after every step is where the
  per-step rewrites of the assignment state would start to cost, and the
  submission policy rework touches the journal that SQLite would replace.
- Add an alternate buffer dashboard for more convenient viewing of the work. This
  makes the most sense with a continuous mode. It would want to read the state
  while a run writes it, which SQLite allows more easily than JSON files that
  are rewritten atomically.
- As an alternative to the existing breadth-first standard mode, allow for a
  depth-first mode. This isn't hugely important, as it can effectively be done
  simply by using the yafu mode (which is probably faster anyway, though I have
  no real data to back that up). Breadth-first is potentially the most wasteful
  in the event of an aborted run, as any work that was done on unfinished
  numbers is effectively lost. (mersenne.ca assignments now keep the progress
  on each composite for the next run, so this applies only to FactorDB work,
  and would apply to neither if FactorDB progress were kept too.)
- Add the ability to calculate and verify Aliquot sequences (or other sequences).
  This would make the most sense in a new utility (similar to analyzer), but it
  can substantially reuse the factoring infrastructure.
- Support mersenne.ca's pretest mode, which asks for ECM-only pre-factoring
  without SIQS or NFS. At the time of writing, mersenne.ca is down so I cannot
  investigate further for now. The ECM levels now tracked for each composite
  should make this easier, as they already model ECM-only work.
