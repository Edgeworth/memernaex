set positional-arguments
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

check:
  poetry check
  poetry run mypy --install-types --non-interactive
  pre-commit run --all-files
