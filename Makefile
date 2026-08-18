# fed_diversity -- `make check` is the first thing to run on a new machine.
PY ?= python
DATA_ROOT ?= ./data
DEVICE ?= cpu
RESULTS ?= ./results

.PHONY: help check test smoke data status dry-run primary alpha clean

help:
	@echo "make data            download + verify CIFAR-10 (once, before any sweep)"
	@echo "make check           full test suite + smoke path; non-zero exit on failure"
	@echo "make check DEVICE=cuda:0     same, on a GPU"
	@echo "make dry-run         list the primary grid and estimated wall-clock"
	@echo "make primary         plot fig_local_steps.png"
	@echo "make alpha           plot fig_alpha.png"

data:
	$(PY) prepare_data.py --data-root $(DATA_ROOT)

# The gate. Runs every test plus the smoke path end to end and prints one
# PASS/FAIL line. Exits non-zero if anything fails, so CI or a shell && chain
# stops here rather than proceeding to a sweep on a broken checkout.
check:
	@$(PY) check.py --device $(DEVICE) --data-root $(DATA_ROOT)

test:
	@FED_DIVERSITY_TEST_DEVICE=$(DEVICE) FED_DIVERSITY_DATA=$(DATA_ROOT) \
		$(PY) test_fed_diversity.py

smoke:
	$(PY) run_sweep.py --smoke --devices $(DEVICE) --per-gpu 1 \
		--data-root $(DATA_ROOT) --results-dir $(RESULTS)_smoke

status:
	$(PY) run_sweep.py --status --results-dir $(RESULTS)

dry-run:
	$(PY) run_sweep.py --preset local-steps --dry-run --results-dir $(RESULTS)

primary:
	$(PY) plot_primary.py --results-dir $(RESULTS)

alpha:
	$(PY) plot_alpha.py --results-dir $(RESULTS)

clean:
	rm -rf __pycache__ $(RESULTS)_smoke
