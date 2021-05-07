##
# Combinators makefile
#
# @file
# @version 0.1
SHELL := bash
RUN_FLAGS :=
PYTHON := python
PYTEST := pytest
GIT := git

all: experiments test

test:
	$(PYTEST) ./tests

experiments: ex/annealing ex/apgs_bshape

.ONESHELL:
ex/%:
	export PYTHONPATH="$$($(GIT) rev-parse --show-toplevel):$$PYTHONPATH"
	if [ -d ./experiments/$(@F) ]; then
		cd ./experiments/$(@F) && $(PYTHON) ./main.py $(RUN_FLAGS)
	else
		echo "========================================="
		echo "./experiments/$(@F) is not an experiment!"
		echo
		echo "please choose from one of:"
		echo "    " annealing apgs_bshape
	fi

profile/%:
	make PYTHON="fil-profile run" RUN_FLAGS="--iteration 500" ex/$(@F)

# end
