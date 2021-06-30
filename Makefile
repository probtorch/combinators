##
# Makefile, largely used for running experiments and kicking off fil memory profiling
#
# @file
# @version 0.1
SHELL := bash
RUN_FLAGS :=
PYTHON := python
PYTEST := pytest
GIT := git
ITERATIONS :=

all: experiments

experiments: ex/annealing ex/apgs

.ONESHELL:
ex/%:
	export PYTHONPATH="$$($(GIT) rev-parse --show-toplevel):$$PYTHONPATH"
	if [ -d ./experiments/$(@F) ]; then
		cd ./experiments/$(@F)
ifdef ITERATIONS
		$(PYTHON) ./main.py --iterations $(ITERATIONS) $(RUN_FLAGS)
else
		$(PYTHON) ./main.py $(RUN_FLAGS)
endif
	else
		echo "========================================="
		echo "./experiments/$(@F) is not an experiment!"
		echo
		echo "please choose from one of:"
		echo "    " annealing apgs
	fi

profile/%:
ifdef ITERATIONS
	make PYTHON="fil-profile run" ITERATIONS=$(ITERATIONS) ex/$(@F)
else
	make PYTHON="fil-profile run" ITERATIONS=2000 ex/$(@F)
endif

# end
