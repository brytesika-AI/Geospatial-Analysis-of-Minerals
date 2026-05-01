.PHONY: setup phase1 phase2 phase3 phase4 clean lint test

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	cp -n .env.example .env || true
	mkdir -p data/raw data/processed models
	@echo "✅  Environment ready. Edit .env with your API keys then run: make phase1"

# ──────────────────────────────────────────────
# Phase 1 — Data + Model
# ──────────────────────────────────────────────
phase1: data model

data:
	@echo "📥  Downloading USGS data..."
	python scripts/01_download_data.py

model:
	@echo "🔧  Engineering features..."
	python scripts/02_process_data.py
	@echo "🤖  Training models (LR / RF / XGBoost)..."
	python scripts/03_train_model.py

# ──────────────────────────────────────────────
# Phase 2 — Streamlit MVP
# ──────────────────────────────────────────────
phase2:
	@echo "🗺️  Launching Streamlit app..."
	streamlit run app/streamlit_app.py

# ──────────────────────────────────────────────
# Phase 3 — LLM smoke-test
# ──────────────────────────────────────────────
phase3:
	python -c "from app.llm_interpreter import GeoInterpreter; \
	           gi = GeoInterpreter(); \
	           print(gi.interpret_score(lat=33.45, lon=-112.07, score=0.82, features={}))"

# ──────────────────────────────────────────────
# Phase 4 — Cloudflare deployment
# ──────────────────────────────────────────────
phase4:
	cd cloudflare && npx wrangler deploy

# ──────────────────────────────────────────────
# Dev helpers
# ──────────────────────────────────────────────
lint:
	ruff check . && black --check .

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f data/processed/*.csv data/processed/*.pkl models/*.pkl
	@echo "🧹  Cleaned build artifacts"
