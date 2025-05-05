.PHONY: all clean setup environment simulation dataset train evaluate report

all: setup environment simulation dataset train evaluate report

setup:
	@echo "Setting up project directories..."
	mkdir -p dataset/images dataset/labels
	mkdir -p weights
	mkdir -p docs/figures

environment:
	@echo "Setting up conda environment..."
	conda env create -f environment.yml
	@echo "Installing pip requirements..."
	conda run -n istanbul-waste python -m pip install -r requirements.txt

simulation:
	@echo "Setting up AirSim and Unreal Engine..."
	conda run -n istanbul-waste python scripts/setup_airsim.py
	conda run -n istanbul-waste python scripts/setup_unreal.py
	@echo "Generating synthetic images..."
	conda run -n istanbul-waste python -m generator.sim.render --config config/sim_config.yaml

dataset:
	@echo "Curating dataset..."
	conda run -n istanbul-waste python scripts/curate_dataset.py --config config/dataset_config.yaml

train:
	@echo "Training YOLOv8 detector..."
	conda run -n istanbul-waste python -m detector.train --config config/training_config.yaml

evaluate:
	@echo "Evaluating detector performance..."
	conda run -n istanbul-waste python -m detector.evaluate --weights weights/synthetic_only.pt --data dataset --config config/training_config.yaml

report:
	@echo "Generating analysis report..."
	conda run -n istanbul-waste jupyter nbconvert --execute --to notebook --inplace docs/report.ipynb

clean:
	@echo "Cleaning project..."
	rm -rf dataset/images/* dataset/labels/*
	rm -rf weights/*
	rm -rf docs/figures/*

deep-clean: clean
	@echo "Removing conda environment..."
	conda env remove -n istanbul-waste
