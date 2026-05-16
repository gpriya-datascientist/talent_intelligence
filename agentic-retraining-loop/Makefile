.PHONY: install setup demo test run

install:
	pip install -e ".[dev]"

setup: install
	python scripts/generate_sample_data.py
	python scripts/train_baseline_model.py

demo: setup
	python scripts/demo_run.py

test:
	pytest tests/ -v

run:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard.py

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down
