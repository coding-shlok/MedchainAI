.PHONY: install install-torch test test-fast contracts-install contracts-test \
        chain-up demo demo-isic-smoke demo-isic sweep-attacks sweep-dp clean

install:
	pip install -r requirements.txt --break-system-packages

install-torch:
	pip install torch torchvision --break-system-packages

test:
	PYTHONPATH=src pytest tests/ -v

test-fast:
	PYTHONPATH=src pytest tests/ -q

contracts-install:
	npm install

contracts-test:
	npm test

# Start a local Hardhat chain in the background and deploy. Run this before any
# experiment configured with chain.backend: web3.
chain-up:
	npx hardhat node > /tmp/hardhat-node.log 2>&1 &
	sleep 3
	npm run deploy:local

# Zero-dependency end-to-end check: proves the whole protocol (FL + reputation +
# mock chain + audit) works with nothing but numpy installed.
demo:
	PYTHONPATH=src python3 experiments/run_simulation.py --config configs/experiments/baseline.yaml

# Real ResNet/CNN training through the same FL+chain pipeline. Needs
# `pip install torch torchvision pandas pillow` (install-torch covers the first
# two) and ISIC 2019 unpacked into data/isic2019/ -- see README/datasets.py.
# Run the smoke target first: it proves the pipeline on ~300 images in minutes.
demo-isic-smoke:
	PYTHONPATH=src python3 experiments/run_simulation_torch.py \
		--config configs/experiments/isic2019_smoke.yaml

demo-isic:
	PYTHONPATH=src python3 experiments/run_simulation_torch.py \
		--config configs/experiments/isic2019.yaml

sweep-attacks:
	PYTHONPATH=src python3 experiments/run_attack_sweep.py \
		--base configs/experiments/baseline.yaml --out results/sweep_attacks --rounds 30

sweep-dp:
	PYTHONPATH=src python3 experiments/run_dp_sweep.py \
		--base configs/experiments/dp_privacy_tradeoff.yaml --out results/sweep_dp

clean:
	rm -rf results/* artifacts/cas __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	touch results/.gitkeep
