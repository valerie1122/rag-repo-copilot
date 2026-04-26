.PHONY: eval readme all clone-fastapi smoke clean-cache

# 1) Clone FastAPI as the eval target (only first time).
#    If repos/fastapi exists but is missing the source subdir (e.g. from a
#    failed earlier clone), wipe it and re-clone cleanly.
clone-fastapi:
	@if [ -d repos/fastapi/fastapi ]; then \
		echo "repos/fastapi already cloned, skipping"; \
	else \
		rm -rf repos/fastapi; \
		mkdir -p repos; \
		git clone --depth 1 https://github.com/tiangolo/fastapi.git repos/fastapi; \
	fi

# 2) Run the 3-mode ablation. Caches embeddings to eval/cache_fastapi.npz.
#    First run: ~3-5 min (embeds 395 chunks via OpenAI).  Re-runs: ~30s.
eval: clone-fastapi
	python -m scripts.evaluate \
		--repo-path repos/fastapi/fastapi \
		--queries eval/queries_fastapi.json \
		--cache eval/cache_fastapi.npz \
		--out eval/results.json

# 3) Re-render the README's results table from eval/results.json.
readme:
	python -m scripts.render_readme

# All-in-one
all: eval readme
	@echo
	@echo "Done. Open README.md to see the populated results table."

# Sanity-check the metrics math without hitting OpenAI.
smoke:
	python -m eval.smoke_test

# Wipe the embedding cache (force re-embed on next eval run).
clean-cache:
	rm -f eval/cache_fastapi.npz eval/cache_fastapi_chunks.json
