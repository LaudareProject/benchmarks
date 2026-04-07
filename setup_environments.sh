#!/bin/bash
set -e

echo "🐍 Setting up project environments..."
if [[ -d ".venv" || -d ".venv-kraken" || -d ".venv-calamari" ]]; then
  rm -rf .venv*
fi

# --- Parent Environment ---
echo "📦 Setting up parent environment in .venv and installing dependencies..."
uv venv .venv
uv sync
echo "✅ Parent environment ready."
echo ""

PARENT_VENV_PYTHON=".venv/bin/python"

# --- Kraken Environment ---
echo "🐙 Setting up Kraken environment in .venv-kraken..."
uv venv .venv-kraken
uv pip install --python .venv-kraken/bin/python -r requirements-kraken.txt
echo "✅ Kraken environment ready."
echo ""

# --- Calamari Environment ---
echo "🦑 Setting up Calamari environment in .venv-calamari..."
uv venv .venv-calamari
uv pip install --python .venv-calamari/bin/python -r requirements-calamari.txt
echo "✅ Calamari environment ready."
echo "🎉 All environments created successfully."

# --- Kraken Models Download ---
echo "⬇️ Downloading Kraken models..."
mkdir -p models/
# curl -L https://zenodo.org/records/14602569/files/blla.mlmodel?download=1 -o models/blla.mlmodel
curl -L https://zenodo.org/records/15030337/files/catmus-medieval-1.6.0.mlmodel?download=1 -o models/catmus-medieval-1.6.0.mlmodel
echo "Checking hash of downloaded models..."
# echo "77a638a83c9e535620827a09e410ed36391e9e8e8126d5796a0f15b978186056  models/blla.mlmodel" | sha256sum --check
echo "227d10f15a936c2492105b7796a6152647c742f248af117856f417008eadbc9c  models/catmus-medieval-1.6.0.mlmodel" | sha256sum --check
echo "✅ Kraken models downloaded and verified."
