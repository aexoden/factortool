# factortool

-----

`factortool` is a utility for factoring numbers, primarily for submission to FactorDB.

## Features

* Uses multiple factoring methods, including trial factoring, rho, P-1, ECM, SIQS
  and NFS.
* Automatically measures duration and success rate to determine the optimal ECM
  crossover threshold and decision between SIQS and NFS.
* As an alternative to the built-in breadth-first factoring, can also simply
  directly use YAFU for each fetched number.
* Automatically fetches composite numbers from FactorDB and submits results.

## Usage

`factortool` currently leverages both [YAFU](https://github.com/bbuhrow/yafu) and
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) to do most of the factoring
work. As such, you will need a correctly configured installation of both.

The recommended way to install the program is to have [uv](https://docs.astral.sh/uv/)
installed, and to simply run the program with `uv run factortool`.

Copy the `config.json.dist` to `config.json` and edit it as appropriate. You may
then run the program. It accepts the following options:

* `--config_path`: To specify a configuration file other than config.json.
* `--min_digits`: The minimium number of digits fetched composite numbers should
  have.
* `--batch_size`: The number of composite numbers to fetch from FactorDB. A value
  of 0 (default) attempts to use an automatic batch size to meet a target time.
* `--target-duration`: The number of seconds to target when using an automatic
  batch size. The default is 600 seconds (ten minutes).
* `--skip_count`: How many composite numbers to skip on FactorDB. Useful for working
  at an offset to avoid conflicts.

Note that the program itself does not loop. Such functionality could be added in
theory, but this way ensures memory leaks aren't an issue. I find it convenient
to use a shell script such as the following:

```sh
while true ; do
    uv run factortool --min_digits 55 --batch_size 60 --skip_count 277 ;
    if [ $? -eq 2 ] ; then exit 2 ; fi ; sleep 1 ;
done
```

I typically run this as a one-liner. It's been split into multiple lines here to
keep the line length down. To stop the script, simply press Ctrl-C. `factortool`
will finish the current factorization it is working on, submit any finished results,
and then exit. The shell script is designed to stop if `factortool` exits due to
an interrupt (such as Ctrl-C).

If you are using direct YAFU support (by setting `factoring_mode` to `yafu` in
config.json), I recommend ensuring YAFU's NFS functionality is correctly
configured.

## Notes

I suspect using YAFU directly is faster, though I have not done an apples-to-apples
test. The `standard` mode is left in both for fun and as a historical curiosity.

## Error Codes

The program returns the following non-zero error codes:

* 1: Configuration error
* 2: Interrupted
* 4: Unexpected CADO-NFS failure
* 5: Unexpected YAFU failure

## License

`factortool` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html)
license.
