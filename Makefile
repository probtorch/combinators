##
# Combinators makefile
#
# @file
# @version 0.1

test:
	pytest ./tests
bug:
	pytest -s ./tests/test_inference_v2.py -k test_extend_propose
# end
