##
# Combinators makefile
#
# @file
# @version 0.1
SHELL := bash
EX_FLAGS :=
PYTHON := python

all: experiments test

test:
	pytest ./tests

experiments: ex/annealing ex/apgs_bshape

.ONESHELL:
ex/%:
	export PYTHONPATH="$$PWD:$$PYTHONPATH"
	if [ -d ./experiments/$(@F) ]; then
		cd ./experiments/$(@F) && $(PYTHON) ./main.py $(EX_FLAGS)
	else
		echo "========================================="
		echo "./experiments/$(@F) is not an experiment!"
		echo
		echo "please choose from one of:"
		echo "    " annealing apgs_bshape
	fi

profile/%:
	make PYTHON=fil-profile EX_FLAGS="--iteration 1000" ex/$(@F)

# end
