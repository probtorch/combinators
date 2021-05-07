##
# Combinators makefile
#
# @file
# @version 0.1
SHELL := bash

all: experiments test

test:
	pytest ./tests

experiments: ex/annealing ex/apgs_bshape

.ONESHELL:
ex/%:
	export PYTHONPATH="$$PWD:$$PYTHONPATH"
	cd ./experiments/$(@F) && python ./main.py

.ONESHELL:
profile/%:
	export PYTHONPATH="$$PWD:$$PYTHONPATH"
	if [ -d ./experiments/$(@F) ]; then
		cd ./experiments/$(@F) && fil-profile ./main.py --iteration 1000
	else
		echo "========================================="
		echo "./experiments/$(@F) is not an experiment!"
		echo
		echo "please choose from one of:"
		echo "    " annealing apgs_bshape
	fi



# end
