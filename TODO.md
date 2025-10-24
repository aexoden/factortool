# TODO

The following is a list of potential TODO items, roughly in my intended order of
implementation. There is no guarantee I will actually get to any of this.

* Revisit how factors are stored. Instead of storing each factor x times,
  potentially store it once with an exponent.
* Refactor code to eliminate as many lint exceptions as possible.
* Dynamic batch sizes take too long to ramp up, especially because a sample size
  of 1 is heavily deweighted.
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
* Ensure that a given ECM level is never done if we know it will take longer than
  SIQS/NFS, even if early in the data collection phase.
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
