.PHONY: docs

# Render every report, then the index last: it only links reports that are already rendered.
docs:
	cd docs && export QUARTO_PYTHON="$$(uv run --project .. python -c 'import sys; print(sys.executable)')" && \
	for f in report_*.qmd index.qmd; do quarto render "$$f" || exit 1; done
