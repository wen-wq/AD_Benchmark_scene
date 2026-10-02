#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN=${PYTHON_BIN:-/data/tianle/miniconda3/envs/data-process-cpu/bin/python}
export PYTHONPATH="$PWD/vendor:/data/tianle/projects/cosmos-transfer2.5:/data/tianle/projects/cosmos-predict2.5/packages/cosmos-cuda${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -s beijing.osm.pbf ]]; then
  curl -fL --retry 3 'https://download.openstreetmap.fr/extracts/asia/china/beijing-latest.osm.pbf' -o beijing.osm.pbf
fi
if [[ ! -s osrm_route.json ]]; then
  curl -fL --retry 3 'https://router.project-osrm.org/route/v1/driving/116.30973,39.99078;116.30546,39.98527?overview=full&geometries=geojson&steps=true' -o osrm_route.json
fi

"$PYTHON_BIN" extract_osm.py
if [[ "${LLM_PROVIDER:-session}" == 'session' ]]; then
  "$PYTHON_BIN" llm_provider.py --provider session
else
  : "${LLM_BASE_URL:?set LLM_BASE_URL}"
  : "${LLM_MODEL:?set LLM_MODEL}"
  "$PYTHON_BIN" llm_provider.py --provider openai-compatible --base-url "$LLM_BASE_URL" --model "$LLM_MODEL"
fi
"$PYTHON_BIN" pku_pipeline.py build
"$PYTHON_BIN" analyze_collisions.py
if [[ -n "${BAIDU_AK:-}" ]]; then
  "$PYTHON_BIN" street_images.py --baidu
else
  "$PYTHON_BIN" street_images.py
fi
"$PYTHON_BIN" harvest_kartaview.py
"$PYTHON_BIN" pku_pipeline.py render --topdown-only
"$PYTHON_BIN" make_previews.py
"$PYTHON_BIN" native_render.py
"$PYTHON_BIN" native_mosaic.py
"$PYTHON_BIN" validate.py --require-native
"$PYTHON_BIN" package.py
