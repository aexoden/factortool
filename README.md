# factortool

-----

`factortool` is a utility for factoring numbers, primarily for submission to
FactorDB and mersenne.ca.

## Features

* Uses multiple factoring methods, including trial factoring, rho, P-1, ECM, SIQS
  and NFS.
* Automatically measures duration and success rate to determine the optimal ECM
  crossover threshold and decision between SIQS and NFS.
* As an alternative to the built-in breadth-first factoring, can also simply
  directly use YAFU for each fetched number.
* Automatically fetches composite numbers and submits results, either from FactorDB
  or from the mersenne.ca Aliquot composite service.

## Usage

`factortool` currently leverages both [YAFU](https://github.com/bbuhrow/yafu) and
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) to do most of the factoring
work. As such, you will need a correctly configured installation of both.

The recommended way to install the program is to have [uv](https://docs.astral.sh/uv/)
installed, and to simply run the program with `uv run factortool`.

Copy the `config.dist.json` to `config.json` and edit it as appropriate. You may
then run the program. It accepts the following options:

* `--config_path`: To specify a configuration file other than config.json.
* `--min_digits`: The minimium number of digits fetched composite numbers should
  have.
* `--max_digits`: The maximum number of digits fetched composite numbers should
  have. Required by the mersenne.ca backend.
* `--batch_size`: The number of composite numbers to fetch from FactorDB. A value
  of 0 (default) attempts to use an automatic batch size to meet a target time.
* `--target-duration`: The number of seconds to target when using an automatic
  batch size. The default is 600 seconds (ten minutes).
* `--skip_count`: How many composite numbers to skip on FactorDB. Useful for working
  at an offset to avoid conflicts. Not supported by the mersenne.ca backend (which
  assigns distinct work to each user).

Note that the program itself does not loop. Such functionality could be added in
theory, but this way ensures memory leaks aren't an issue. I find it convenient
to use a shell script such as the following:

```sh
while true ; do
    uv run factortool --min_digits 55 --batch_size 60 --skip_count 277 ;
    status=$? ;
    if [ $status -eq 2 ] || [ $status -eq 6 ]; then exit $status ; fi ;
    sleep 1 ;
done
```

I typically run this as a one-liner. It's been split into multiple lines here to
keep the line length down. To stop the script, simply press Ctrl-C. `factortool`
will finish the current factorization it is working on, submit any finished results,
and then exit. The shell script is designed to stop if `factortool` exits due to
an interrupt such as Ctrl-C (exit status 2) or for a permanent HTTP error (exit
status 6).

Repeated interrupts escalate:

* The first stops `factortool` from fetching any more work, but lets the already
  fetched batch run to completion as normal (subject to the normal time limit).
* The second gives up on the rest of the batch, but completes the current
  factorization. Any partial factorizations are reported and untouched assigned
  work (on the mersenne.ca backend) is retained for the next run.
* The third abandons the current factorization.

All three submit whatever results are in hand before exiting.

If you are using direct YAFU support (by setting `factoring_mode` to `yafu` in
config.json), I recommend ensuring YAFU's NFS functionality is correctly
configured.

## Backends

The `backend` setting in config.json selects the source of composite numbers and
where factors are submitted.

* `factordb`: fetches composites from FactorDB and submits factors back there.
  Set `factordb_username` and `factordb_password` to log in; otherwise, results
  are submitted anonymously.
* `mersenne_ca`: fetches assigned composites from the [mersenne.ca Aliquot composite
  service](https://www.mersenne.ca/aliquot/?compositelist=1) and submits results
  through that service. Set `gimps_login` to your GIMPS username. `--max-digits`
  is required, and `--skip-count` is not supported (or needed to avoid conflict).

Partial factorizations are submitted if a run ends after finding one or more
factors. For `mersenne_ca`, unfinished assignments are saved in `assignment_state_path`
and resumed on the next run.

## User Agent

Requests carry a User-Agent header naming the tool, its version, the project URL
and the configured account for the selected backend. For example:

```text
factortool/0.1.0 (Username; +https://github.com/aexoden/factortool)
```

Set `user_agent` in config.json to replace the value entirely if you would rather
send something else.

## YAFU Working Directories

`factortool` runs each YAFU invocation in a separate temporary directory under
`work_path`, preventing YAFU's working files from interfering with other runs.

A `yafu.ini` is copied into each working directory and relative tool paths are
adjusted automatically. By default, factortool uses the `yafu.ini` next to the
YAFU binary; set `yafu_ini_path` to use a different file.

## Notes

I suspect using YAFU directly is faster, though I have not done an apples-to-apples
test. The `standard` mode is left in both for fun and as a historical curiosity.

## Error Codes

The program returns the following non-zero error codes:

* 1: Configuration error
* 2: Interrupted (any interrupt level)
* 3: Time limit exceeded
* 4: Unexpected CADO-NFS failure
* 5: Unexpected YAFU failure
* 6: Permanent HTTP error in the backend

## License

`factortool` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html)
license.
