# Canonical local workflow for the retained param-eq replication package.
# The file dependencies document the expected order:
# 1. normalize archived Haskell outputs
# 2. run the current local Haskell pipeline across the same rows
# 3. run the retained Egglog corpus baseline against those rows
# 4. refresh the golden cases and notebook
# 5. run the replication-local tests

PARAM_EQ_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(PARAM_EQ_DIR)/../../../..)
EGGLOG_PARAM_EQ_DATA_DIR ?= $(REPO_ROOT)/../param-eq-haskell
export EGGLOG_PARAM_EQ_DATA_DIR

ARTIFACT_DIR := $(PARAM_EQ_DIR)/artifacts
GOLDEN := $(PARAM_EQ_DIR)/haskell_golden.json
NOTEBOOK := $(PARAM_EQ_DIR)/replication.ipynb
ARCHIVED_HASKELL_ROWS := $(ARTIFACT_DIR)/haskell_paper_rows.csv
LIVE_HASKELL_ROWS := $(ARTIFACT_DIR)/haskell_live_rows.csv
ARCHIVED_RUNTIME_ROWS := $(ARTIFACT_DIR)/pagie_runtime_scatter.csv
RUNTIME_COMPARE_ROWS := $(ARTIFACT_DIR)/pagie_runtime_compare.csv
EGGLOG_ROWS := $(ARTIFACT_DIR)/egglog_paper_rows.csv
HASKELL_FIXTREE := $(EGGLOG_PARAM_EQ_DATA_DIR)/src/FixTree.hs
HASKELL_MAIN := $(EGGLOG_PARAM_EQ_DATA_DIR)/src/Main.hs
HASKELL_RESULTS := $(EGGLOG_PARAM_EQ_DATA_DIR)/results
HASKELL_RUNTIME := $(EGGLOG_PARAM_EQ_DATA_DIR)/runtimes
HASKELL_TABLES := \
	$(HASKELL_RESULTS)/pagie_table_counts.csv \
	$(HASKELL_RESULTS)/kotanchek_table_counts.csv
HASKELL_EXPRS := \
	$(wildcard $(HASKELL_RESULTS)/exprs/*_exprs_*) \
	$(wildcard $(HASKELL_RESULTS)/exprs_simpl/*_exprs_*)

.PHONY: golden archived-artifacts live-haskell artifacts notebook test all

$(ARCHIVED_HASKELL_ROWS) $(ARCHIVED_RUNTIME_ROWS): $(PARAM_EQ_DIR)/normalize_archives.py $(PARAM_EQ_DIR)/paths.py $(HASKELL_TABLES) $(HASKELL_EXPRS) $(HASKELL_RUNTIME)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.normalize_archives

$(RUNTIME_COMPARE_ROWS): $(PARAM_EQ_DIR)/run_runtime_compare.py $(PARAM_EQ_DIR)/run_egglog_corpus.py $(ARCHIVED_RUNTIME_ROWS) $(HASKELL_MAIN) $(HASKELL_FIXTREE)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.run_runtime_compare

$(LIVE_HASKELL_ROWS): $(PARAM_EQ_DIR)/run_haskell_corpus.py $(PARAM_EQ_DIR)/generate_haskell_golden.py $(PARAM_EQ_DIR)/pipeline.py $(PARAM_EQ_DIR)/paths.py $(ARCHIVED_HASKELL_ROWS) $(HASKELL_FIXTREE) $(HASKELL_MAIN)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.run_haskell_corpus

$(EGGLOG_ROWS): $(PARAM_EQ_DIR)/run_egglog_corpus.py $(PARAM_EQ_DIR)/pipeline.py $(ARCHIVED_HASKELL_ROWS) $(ARCHIVED_RUNTIME_ROWS)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.run_egglog_corpus

$(GOLDEN): $(PARAM_EQ_DIR)/generate_haskell_golden.py $(PARAM_EQ_DIR)/pipeline.py $(PARAM_EQ_DIR)/paths.py $(HASKELL_FIXTREE) $(HASKELL_MAIN) $(ARCHIVED_HASKELL_ROWS)
	cd $(REPO_ROOT) && uv run python -m egglog.exp.param_eq.generate_haskell_golden

$(NOTEBOOK): $(PARAM_EQ_DIR)/replication.py $(ARCHIVED_HASKELL_ROWS) $(LIVE_HASKELL_ROWS) $(EGGLOG_ROWS) $(RUNTIME_COMPARE_ROWS)
	cd $(REPO_ROOT) && uv run python python/egglog/exp/param_eq/replication.py

archived-artifacts: $(ARCHIVED_HASKELL_ROWS) $(ARCHIVED_RUNTIME_ROWS)

live-haskell: $(LIVE_HASKELL_ROWS)

golden: $(GOLDEN)

artifacts: $(ARCHIVED_HASKELL_ROWS) $(ARCHIVED_RUNTIME_ROWS) $(LIVE_HASKELL_ROWS) $(EGGLOG_ROWS) $(RUNTIME_COMPARE_ROWS)

notebook: $(NOTEBOOK)

test: $(GOLDEN) $(EGGLOG_ROWS) $(NOTEBOOK)
	cd $(REPO_ROOT) && uv run pytest python/egglog/exp/param_eq/test_pipeline.py python/egglog/exp/param_eq/test_replication_notebook.py -q

all: test
