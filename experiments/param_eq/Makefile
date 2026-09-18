.PHONY: smoke binary container haskell paired aggregate

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
RESULTS := $(ROOT)/experiments/param_eq/results
RAW := $(RESULTS)/raw
PAIRED := $(RAW)/paired.csv
EXECUTION_MODE ?= release

smoke:
	cd $(ROOT) && uv run pytest --benchmark-disable -q -m param_eq_smoke python/tests/param_eq

binary:
	cd $(ROOT) && uv run python -m experiments.param_eq.run --variant binary --execution-mode $(EXECUTION_MODE) --output $(RAW)/binary.csv

container:
	cd $(ROOT) && uv run python -m experiments.param_eq.run --variant container --execution-mode $(EXECUTION_MODE) --output $(RAW)/container.csv

haskell:
	cd $(ROOT) && uv run python -m experiments.param_eq.run --implementation haskell --variant binary --execution-mode $(EXECUTION_MODE) --output $(RAW)/haskell.csv

paired:
	cd $(ROOT) && uv run python -m experiments.param_eq.run --variant both --execution-mode $(EXECUTION_MODE) --output $(PAIRED)

aggregate: paired
	cd $(ROOT) && uv run python -m experiments.param_eq.aggregate --binary $(PAIRED) --container $(PAIRED) --output-dir $(RESULTS)
