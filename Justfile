set positional-arguments
alias t := test
alias f := fix
alias u := update

default:
  @just --list

fix:
  pre-commit run --all-files

update:
  poetry run poetry up --latest
  poetry update
  pre-commit autoupdate
