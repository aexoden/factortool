# factortool

-----

`factortool` is a utility for factoring numbers, primarily for submission to
FactorDB and mersenne.ca.

## Features

* Uses multiple factoring methods, including trial factoring, rho, P-1, ECM, SIQS
  and NFS (via CADO-NFS or YAFU's built-in NFS).
* Automatically measures duration and success rate to determine the optimal ECM
  crossover threshold and the final choice of SIQS, YAFU NFS or CADO-NFS.
* As an alternative to the built-in breadth-first factoring, can also simply
  directly use YAFU for each fetched number.
* Automatically fetches composite numbers and submits results, either from FactorDB
  or from the mersenne.ca Aliquot composite service.

## Usage

`factortool` currently leverages both [YAFU](https://github.com/bbuhrow/yafu) and
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) to do most of the factoring
work. As such, you will need a correctly configured installation of YAFU. For NFS
support, you will need either YAFU's NFS configured, or CADO-NFS installed.

The recommended way to install the program is to have [uv](https://docs.astral.sh/uv/)
installed, and to simply run the program with `uv run factortool`.

Copy the `config.dist.json` to `config.json` and edit it as appropriate. Only
`backend`, `max_threads` and `yafu_path` are required. If you are using the
`mersenne_ca` backend, you also need to set `gimps_login`, and if you enable
`use_nfs_cado`, you also need to set `cado_nfs_path`. Anything unspecified
defaults to the values in `config.dist.json`.

The configuration and the options below are checked before any work is fetched.
`max_threads` must be at least 1, `yafu_path` must be an executable file, as
must `cado_nfs_path` if `use_nfs_cado` is enabled, and `yafu_ini_path` must
exist if it is set.

You may then run the program. It accepts the following options:

* `--config_path`: To specify a configuration file other than config.json.
* `--min_digits`: The minimium number of digits fetched composite numbers should
  have. Must be at least 1.
* `--max_digits`: The maximum number of digits fetched composite numbers should
  have. Required by the mersenne.ca backend.
* `--batch_size`: The number of composite numbers to fetch. A value of 0 (default)
  attempts to use an automatic batch size to meet a target time.
* `--target_duration`: The number of seconds to target when using an automatic
  batch size. Must be positive. The default is 600 seconds (ten minutes). If
  factoring takes more than twice this long, the run ends after the current
  factorization. A run with an explicit `--batch_size` has no time limit.
* `--skip_count`: How many composite numbers to skip on FactorDB. Useful for working
  at an offset to avoid conflicts. Not supported by the mersenne.ca backend (which
  assigns distinct work to each user).
* `--no_new_work`: Work only any retained assignments from a previous run, and
  don't fetch any more. Only supported by the mersenne.ca backend.

Note that the program itself does not loop. Such functionality could be added in
theory, but this way ensures memory leaks aren't an issue. I find it convenient
to use a shell script such as the following:

```sh
bash -c '
    while true ; do
        uv run factortool --min_digits 55 --batch_size 60 --skip_count 277 ;
        status=$? ;
        if [ $status -eq 2 ] || [ $status -eq 6 ] || [ $status -eq 8 ]; then
            exit $status ;
        fi ;
        sleep 1 ;
    done
'
```

I typically run this as a one-liner. It's been split into multiple lines here to
keep the line length down. The loop runs under `bash -c` so that `exit` only
ends the loop, rather than the entire shell it was typed into. This also makes
it easy to prefix the whole loop with another command, such as `taskset -c
12-15` to pin `factortool` (and the processes it spawns) to specific CPU cores.

To stop the script, simply press Ctrl-C. `factortool` will finish the current
batch it is working on, submit any finished results, and then exit. The shell
script is designed to stop if `factortool` exits due to an interrupt such as
Ctrl-C (exit status 2), for a permanent HTTP error (exit status 6) or because
another instance is already running (exit status 8).

Repeated interrupts escalate:

* The first stops `factortool` from fetching any more work, but lets the already
  fetched batch run to completion as normal (subject to the time limit, if any).
* The second gives up on the rest of the batch, but completes the current
  factorization. Any partial factorizations are reported and untouched assigned
  work (on the mersenne.ca backend) is retained for the next run.
* The third abandons the current factorization.

All three submit whatever results are in hand before exiting. An interrupt that
arrives while those results are being submitted stops the submission, and leaves
the rest for the next run.

SIGTERM, SIGHUP and, on Windows, Ctrl-Break have the same effect as the third
interrupt. Once the current factorization has been abandoned, one more signal
exits immediately, without saving anything further or submitting the remaining results.

If YAFU or CADO-NFS fail on a number (by exiting with an error or returning an
incomplete factorization), the failure is logged and the batch continues. A
final factoring method that fails is first retried with the other eligible final
methods. If the same method fails three times in a row, or a tool fails every
time it is run during a batch, the installation is assumed to be broken and
`factortool` exits with status 5 (YAFU) or 4 (CADO-NFS).

If you are using direct YAFU support (by setting `factoring_mode` to `yafu` in
config.json), I recommend ensuring YAFU's NFS functionality is correctly
configured.

## Final Factoring Methods

In `standard` mode, once an appropriate amount of ECM is done, each remaining
composite is finished with one of three methods:

* SIQS, via YAFU, for composites of up to `max_siqs_digits` digits.
* YAFU's built-in NFS, if `use_nfs_yafu` is `true`, for composites of at least
  85 digits. Configure YAFU via the `yafu.ini` configuration file, and make sure
  `nfs()` works in YAFU.
* CADO-NFS, if `use_nfs_cado` is `true`, for composites of at least 57 digits.

Among the eligible methods, `factortool` picks the one with the lowest average
time for that digit count. A method without any data is run first to collect a sample.

## Backends

The `backend` setting in config.json selects the source of composite numbers and
where factors are submitted.

* `factordb`: fetches composites from FactorDB and submits factors back there,
  using its [JSON-RPC API](https://factordb.com/api.php). Set `factordb_api_token`
  to submit as your account (and be credited for factors); otherwise, results
  are submitted anonymously. You can get your API token by signing in and
  clicking your username in the upper right to access your account page.
* `mersenne_ca`: fetches assigned composites from the [mersenne.ca Aliquot composite
  service](https://www.mersenne.ca/aliquot/?compositelist=1) and submits results
  through that service. Set `gimps_login` to your GIMPS username. `--max_digits`
  is required, and `--skip_count` is not supported (or needed to avoid conflict).

Partial factorizations are submitted if a run ends after finding one or more
factors. For `mersenne_ca`, unfinished assignments are saved in the state
directory and resumed on the next run. Assignments will be dropped if they come
within ten minutes of expiration without being started.

## Unsent Results

Every result is recorded in the state directory as soon as it is ready to
submit, and removed once the backend has accepted or rejected it. Whatever is
left when `factortool` exits is submitted at the start of the next run with the
same backend. Unsent results do not change the exit status.

A result that still has not been submitted is eventually discarded: after 24
hours for `factordb`, or when its assignment expires for `mersenne_ca`. A result
found during the current run is always attempted at least once. Rate limits are
respected across runs.

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

* 1: Configuration error, invalid arguments or invalid statistics data
* 2: Interrupted (any interrupt level) or asked to end by a signal
* 3: Time limit exceeded
* 4: Unexpected CADO-NFS failure
* 5: Unexpected YAFU failure
* 6: Permanent HTTP error in the backend
* 7: Error saving state or results or during shutdown
* 8: Another instance is using the state or working directory

## License

`factortool` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html)
license.
