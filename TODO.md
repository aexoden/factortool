# TODO

The following is a list of potential TODO items, roughly in my intended order of
implementation. There is no guarantee I will actually get to any of this.

* Provide options to allow the user to disable the time limit or to tweak how long
  it is instead of always doing twice the target time.
* Fix the time limit to only occur if target time is enabled.
* Revisit how factors are stored. Instead of storing each factor x times,
  potentially store it once with an exponent.
* Refactor the standard factoring method away from breadth-first. It doesn't
  really add anything, and if the program is aborted (either manually or because
  of an expired time limit), the ECM work on any remaining unfactored exponents is
  effectively lost as it will simply be repeated on another run.
* Consider exactly when the time limit being exceeded should terminate the
  program. (That is, should it abort as soon as possible or should it wait for
  the current factorization to finish?)
* Make the log level configurable.
* Refactor code to eliminate as many lint exceptions as possible.
* Dynamic batch sizes take too long to ramp up, especially because a sample size
  of 1 is heavily deweighted.
* CTRL-C during a fetch wait does not gracefully exit. You currently have to do
  a second one to exit. This might make sense (even if no factoring is occurring)
  if it's a submission that's being waited on, but if the request is a fetch, it's
  unnecessary.
* Before working on a number, consider fetching any existing factors from FactorDB.
  We'll probably want to keep track of submission by factor at that point. That can
  go along with changing storage to factor and exponent.
* Investigate integrating the looping process directly into the tool. This needs
  to change the way login is handled to potentially revalidate the cookie, though
  if submission is largely done via the API, this will be less important. I
  originally thought about managing the cookie directly and logging back in if we
  were unexpectedly logged out, but changes to FactorDB to make the cookie IDs
  permanent make this much less relevant today.
* Keep allowing direct YAFU usage as many people will find that more useful, but
  because YAFU does have some bugs that can impede progress, I'd like to retain
  the "standard" self-managed factoring logic. Adding support for YAFU's ggnfs-
  based NFS would probably be good, and potentially its hybrid CADO/msieve. Both,
  at least on some of my machines, will occasionally fail for reasons I don't
  understand. (Only observed on non-AVX512 machines to this date.)
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
  weird collection composites FactorDB had (and how they were constructed).
* Reconsider how to best handle fetching larger batches, especially with FactorDB
  returning a lot of 502 errors right now (which is more likely with larger batches).
* Add additional tests.
* Investigate making threads overridable on the command line.
* Add support for calculating and verifying Aliquot sequences.
