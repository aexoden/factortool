# TODO

The following is a list of bug fixes and potential improvements, roughly in the
intended order of implementation. There is no guarantee that an item on this list
will ever actually be implemented. Some of the earlier ones may be obsoleted by
later ones.

## Bug Fixes

- The analyzer currently stops showing ECM data as soon as it encounters a level
  with no data. While comparatively rare, it's possible for there to be gaps as
  numbers can start at arbitrary points if they were added as cofactors. The
  same assumption is in `_get_average_time_internal`, so a digit count mostly
  reached via cofactors never leaves the initial fallback cutoff.
- Submissions are retried forever and can't be interrupted, so a service outage
  hangs the shutdown. A 429 without a `Retry-After` header waits an hour, and a
  supplied value isn't clamped. Give up after a bounded time, write the unsent
  results to a pending file and resubmit them at the start of the next run.
- Two instances sharing a configuration overwrite each other's statistics,
  batch state and assignment state. Take a lock file at startup and refuse to
  run if another instance holds it.
- Look for any instances of existing tests codifying "odd" behavior. (In other
  words, the test was written to accept whatever the current behavior was rather
  than a more objectively correct behavior and implementing the necessary fixes).

## Minor Features

- Add an option to disable the time limit and adjust the time limit directly,
  rather than always defaulting to double the target time. While there, consider
  if the time limit should abort immediately or finish the current factorization
  (or some happy medium). Killing the running tool at some hard limit is also
  the only realiable defense against a tool that hangs, which is otherwise never
  detected. It might be worth checking if YAFU and CADO-NFS regularly generate
  log output for monitoring. Could also consider looking at their CPU usage, but
  that wouldn't help with a spinlock.
- Display digit counts even for smaller numbers.
- For backends that batch submissions, consider a more intelligent submission
  strategy: Submit if X seconds have passed, the current batch is full or if
  everything's done.
- Make the log level configurable.
- Allow temporarily setting max_threads via a command-line parameter. Along with
  backend, it's probably the option people are most likely to want to change
  between runs.
- Change the README's shell loop to stop on any exit status other than 0 and 3.
  It currently only stops on 2 and 6, so a configuration error, invalid
  argument, tool failure or cleanup failure (7) retries every second.
- Verify whether `cado-nfs.py` uses its own directory under `/tmp` unless given
  `--workdir`. If so, pass the managed working directory so an aborted run
  doesn't leave it behind. The `stdin` passed to it also looks unnecessary.
- Remove stale `yafu-*` and `nfs-cado-*` working directories at startup, as a
  hard kill leaves them behind.
- Write the assignment state when work is assigned rather than only at exit, so
  a hard kill doesn't forget reserved work.
- Log when a FactorDB fetch is capped at 1000 numbers.

## Code/Architectural Improvements

- Move all state files into a directory with standardized names, changing the
  configuration to instead specify the directory.
- Determine how multithreaded YAFU's implementation of SIQS is (may vary by
  digit count), and consider threading it ourselves like with rho and P-1.
- Why is `max_siqs_digits` even an option? Is there a historical reason for it?
  At the same time, investigate if there are any other superfluous options.
- Consider whether the mersenne.ca one-hour assignment window should influence
  batch sizing. As long as the target duration is well below the window (e.g.
  the default of ten minutes), it's fine, but perhaps a warning if the user tries
  to use too large a duration such that their assignments risk expiring.
- Investigate moving away from Tap (Typed Argument Parser) for argument parsing.
  The obvious alternatives are click and typer, both of which are far more
  active projects than Tap, but this list isn't exhaustive. If opting to keep
  Tap, at least standardize on using hyphens rather than underscores in option
  names. At the same time, consider looking at the current exit status codes,
  and see if there are changes that can be made to improve the codes or to
  better follow any standards.
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
- Investigate whether the time to factor estimates should account for finding all
  factors rather than just the first. The original logic was that all we care
  about is finding the first factor. The cofactor then essentially goes into the
  queue as a new number to factor with its own expected duration, copying over
  any existing work that was done on the original number in its own history.
- Revisit how factors are stored. Instead of storing each factor N times,
  potentially store it once with an exponent. This is an old item and I don't
  recall the motivation. Investigation of the impacts (both positive and
  negative) would be needed before spending any time on it.
- For the standard factoring method, reconsider the ECM curves and the B1 values.
  This may want to adjust for whether YAFU is using AVX-ECM or not. The lowest
  levels of ECM are probably not adding tons of value on their own, but use data
  to guide this decision.
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
- Refactor to eliminate as many linting exceptions as possible.
- Dynamic batch sizes can be slow to ramp up. This may not be as bad as it was
  in the past, but it's still worth looking at once more. One cause: after a
  reset, the first measurement is weighted by the time since the last update,
  which is close to nothing for a fast batch. Seeding directly from the first
  measurement when there's no prior rate would help.
- Remove the unused `flask` dependency.
- The `*_cooldown_period` settings are only used as the initial retry delay,
  not as a cooldown between requests. Either rename them or implement the
  cooldown.
- Consider `gmpy2.is_strong_bpsw_prp` for `is_prime` above the range where the
  fixed Miller-Rabin bases are deterministic.
- Documentation: add an Unreleased section to the changelog (mersenne.ca, YAFU
  NFS and the JSON-RPC migration have all landed since 0.1.0), document
  `--version` and the configuration settings in the README, and reconcile the
  README's Ctrl-C paragraph with the escalation list below it.
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
  may go along with the dashboard item.
- Add an alternate buffer dashboard for more convenient viewing of the work. This
  makes the most sense with a continuous mode.
- As an alternative to the existing breadth-first standard mode, allow for a
  depth-first mode. This isn't hugely important, as it can effectively be done
  simply by using the yafu mode (which is probably faster anyway, though I have
  no real data to back that up). Breadth-first is potentially the most wasteful
  in the event of an aborted run, as any work that was done on unfinished
  numbers is effectively lost.
- Add the ability to calculate and verify Aliquot sequences (or other sequences).
  This would make the most sense in a new utility (similar to analyzer), but it
  can substantially reuse the factoring infrastructure.
- Support mersenne.ca's pretest mode, which asks for ECM-only pre-factoring
  without SIQS or NFS. At the time of writing, mersenne.ca is down so I cannot
  investigate further for now.
