# Canonical local workflow for the retained param-eq replication package.
# The file dependencies document the expected order:
# 1. vendor the raw archived Haskell sources
# 2. run the current local Haskell pipeline across the retained rows
# 3. run the retained Egglog corpus baseline against those rows
# 4. build the Pagie runtime comparison
# 5. refresh the notebook
# 6. run the replication-local tests

PARAM_EQ_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(PARAM_EQ_DIR)/../../../..)
EGGLOG_PARAM_EQ_DATA_DIR ?= $(REPO_ROOT)/../param-eq-haskell
export EGGLOG_PARAM_EQ_DATA_DIR

ARTIFACT_DIR := $(PARAM_EQ_DIR)/artifacts
ORIGINAL_ROOT := $(ARTIFACT_DIR)/original
ORIGINAL_RUNTIMES := $(ORIGINAL_ROOT)/runtimes
LIVE_RESULTS := $(ARTIFACT_DIR)/live_results.csv
EGGLOG_RESULTS := $(ARTIFACT_DIR)/egglog_results.csv
EGGLOG_BINARY_RESULTS := $(ARTIFACT_DIR)/egglog_results_binary.csv
EGGLOG_CONTAINER_RESULTS := $(ARTIFACT_DIR)/egglog_results_container.csv
EGGLOG_BINARY_SAMPLE_RESULTS := $(ARTIFACT_DIR)/egglog_results_binary_sample.csv
EGGLOG_CONTAINER_SAMPLE_RESULTS := $(ARTIFACT_DIR)/egglog_results_container_sample.csv
RUNTIME_COMPARE_ROWS := $(ARTIFACT_DIR)/runtime_compare.csv
EGGLOG_RANK_MISSES := $(ARTIFACT_DIR)/egglog_rank_misses.csv
RANK_MISS_LONGER_RUNS := $(ARTIFACT_DIR)/egglog_rank_miss_longer_runs.csv
NOTEBOOK := $(PARAM_EQ_DIR)/replication.ipynb
SAMPLE_LIMIT_PER_DATASET ?= 10

HASKELL_FIXTREE := $(EGGLOG_PARAM_EQ_DATA_DIR)/src/FixTree.hs
HASKELL_MAIN := $(EGGLOG_PARAM_EQ_DATA_DIR)/src/Main.hs
HASKELL_RESULTS := $(EGGLOG_PARAM_EQ_DATA_DIR)/results
HASKELL_RUNTIME := $(EGGLOG_PARAM_EQ_DATA_DIR)/runtimes
HASKELL_TABLES := \
	$(HASKELL_RESULTS)/pagie_table_counts.csv \
	$(HASKELL_RESULTS)/kotanchek_table_counts.csv
HASKELL_RESULT_ROWS := \
	$(HASKELL_RESULTS)/pagie_results \
	$(HASKELL_RESULTS)/kotanchek_results
HASKELL_EXPRS := \
	$(wildcard $(HASKELL_RESULTS)/exprs/*_exprs_*) \
	$(wildcard $(HASKELL_RESULTS)/exprs_simpl/*_exprs_*)

.PHONY: archived-artifacts live-haskell egglog-binary egglog-container egglog-binary-sample egglog-container-sample compare-binary-container compare-binary-container-sample baseline-rerun rank-misses rank-miss-longer-runs artifacts notebook test all

$(ORIGINAL_RUNTIMES): $(PARAM_EQ_DIR)/normalize_archives.py $(PARAM_EQ_DIR)/paths.py $(HASKELL_TABLES) $(HASKELL_RESULT_ROWS) $(HASKELL_EXPRS) $(HASKELL_RUNTIME)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.normalize_archives

$(LIVE_RESULTS): $(PARAM_EQ_DIR)/run_haskell_corpus.py $(PARAM_EQ_DIR)/live_results.py $(PARAM_EQ_DIR)/paths.py $(ORIGINAL_RUNTIMES) $(HASKELL_FIXTREE) $(HASKELL_MAIN)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.run_haskell_corpus

$(EGGLOG_RESULTS): $(PARAM_EQ_DIR)/run_egglog_corpus.py $(PARAM_EQ_DIR)/egglog_results.py $(PARAM_EQ_DIR)/domain.py $(PARAM_EQ_DIR)/pipeline.py $(ORIGINAL_RUNTIMES)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.run_egglog_corpus --variant baseline --output $@

$(EGGLOG_BINARY_RESULTS): $(PARAM_EQ_DIR)/run_egglog_corpus.py $(PARAM_EQ_DIR)/egglog_results.py $(PARAM_EQ_DIR)/domain.py $(PARAM_EQ_DIR)/pipeline.py $(ORIGINAL_RUNTIMES)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.run_egglog_corpus --variant baseline --output $@

$(EGGLOG_CONTAINER_RESULTS): $(PARAM_EQ_DIR)/run_egglog_corpus.py $(PARAM_EQ_DIR)/egglog_results.py $(PARAM_EQ_DIR)/domain.py $(PARAM_EQ_DIR)/pipeline.py $(ORIGINAL_RUNTIMES)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.run_egglog_corpus --variant container --output $@

$(EGGLOG_BINARY_SAMPLE_RESULTS): $(PARAM_EQ_DIR)/run_egglog_corpus.py $(PARAM_EQ_DIR)/egglog_results.py $(PARAM_EQ_DIR)/domain.py $(PARAM_EQ_DIR)/pipeline.py $(ORIGINAL_RUNTIMES)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.run_egglog_corpus --variant baseline --limit-per-dataset $(SAMPLE_LIMIT_PER_DATASET) --output $@

$(EGGLOG_CONTAINER_SAMPLE_RESULTS): $(PARAM_EQ_DIR)/run_egglog_corpus.py $(PARAM_EQ_DIR)/egglog_results.py $(PARAM_EQ_DIR)/domain.py $(PARAM_EQ_DIR)/pipeline.py $(ORIGINAL_RUNTIMES)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.run_egglog_corpus --variant container --limit-per-dataset $(SAMPLE_LIMIT_PER_DATASET) --output $@

$(RUNTIME_COMPARE_ROWS): $(PARAM_EQ_DIR)/run_runtime_compare.py $(PARAM_EQ_DIR)/pipeline.py $(ORIGINAL_RUNTIMES) $(HASKELL_MAIN) $(HASKELL_FIXTREE)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.run_runtime_compare

$(EGGLOG_RANK_MISSES): $(PARAM_EQ_DIR)/analyze_baseline_results.py $(ORIGINAL_RUNTIMES) $(EGGLOG_RESULTS)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.analyze_baseline_results

$(RANK_MISS_LONGER_RUNS): $(PARAM_EQ_DIR)/analyze_rank_miss_longer_runs.py $(EGGLOG_RANK_MISSES)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.analyze_rank_miss_longer_runs

$(NOTEBOOK): $(PARAM_EQ_DIR)/replication.py $(ORIGINAL_RUNTIMES) $(LIVE_RESULTS) $(EGGLOG_RESULTS) $(RUNTIME_COMPARE_ROWS)
	cd $(REPO_ROOT) && uv run python python/egglog/exp/param_eq/replication.py

archived-artifacts: $(ORIGINAL_RUNTIMES)

live-haskell: $(LIVE_RESULTS)

egglog-binary: $(EGGLOG_BINARY_RESULTS)

egglog-container: $(EGGLOG_CONTAINER_RESULTS)

egglog-binary-sample: $(EGGLOG_BINARY_SAMPLE_RESULTS)

egglog-container-sample: $(EGGLOG_CONTAINER_SAMPLE_RESULTS)

compare-binary-container:
	$(MAKE) -B egglog-binary
	$(MAKE) -B egglog-container
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.summarize_corpus_comparison --old $(EGGLOG_BINARY_RESULTS) --new $(EGGLOG_CONTAINER_RESULTS) --old-variant baseline --new-variant container --old-label Binary --new-label Containers --input-kind both --sort size-delta --limit 40

compare-binary-container-sample:
	$(MAKE) -B egglog-binary-sample
	$(MAKE) -B egglog-container-sample
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.summarize_corpus_comparison --old $(EGGLOG_BINARY_SAMPLE_RESULTS) --new $(EGGLOG_CONTAINER_SAMPLE_RESULTS) --old-variant baseline --new-variant container --old-label Binary --new-label Containers --input-kind both --sort size-delta --limit 40

baseline-rerun: $(LIVE_RESULTS) $(EGGLOG_RESULTS) $(RUNTIME_COMPARE_ROWS)

rank-misses: $(EGGLOG_RANK_MISSES)

rank-miss-longer-runs: $(RANK_MISS_LONGER_RUNS)

artifacts: $(ORIGINAL_RUNTIMES) $(LIVE_RESULTS) $(EGGLOG_RESULTS) $(RUNTIME_COMPARE_ROWS)

notebook: $(NOTEBOOK)

test:
	cd $(REPO_ROOT) && uv run pytest python/egglog/exp/param_eq/test_pipeline.py python/egglog/exp/param_eq/test_corpus_artifacts.py python/egglog/exp/param_eq/test_rank_misses.py python/egglog/exp/param_eq/test_rank_miss_longer_runs.py python/egglog/exp/param_eq/test_replication_notebook.py -q

all: test
