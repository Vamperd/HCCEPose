#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
TEXTURES_PATH="${TEXTURES_PATH:-${REPO_ROOT}/cc0textures-512}"
SOURCE_DATASET_PATH="${SOURCE_DATASET_PATH:-${REPO_ROOT}/dji-action4}"
OUTPUT_DATASET_PATH="${OUTPUT_DATASET_PATH:-${REPO_ROOT}/dji-action4-twoobj-materials}"
PY_SCRIPT="${PY_SCRIPT:-${REPO_ROOT}/scripts/render_dji_action4_material_scene.py}"
MATERIAL_START="${MATERIAL_START:-0}"
MATERIAL_STOP="${MATERIAL_STOP:-50}"

if [[ ! "${MATERIAL_START}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] MATERIAL_START must be a non-negative integer: ${MATERIAL_START}" >&2
    exit 1
fi

if [[ ! "${MATERIAL_STOP}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] MATERIAL_STOP must be a non-negative integer: ${MATERIAL_STOP}" >&2
    exit 1
fi

if (( MATERIAL_STOP <= MATERIAL_START )); then
    echo "[ERROR] MATERIAL_STOP must be greater than MATERIAL_START." >&2
    exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "[ERROR] Missing renderer script: ${PY_SCRIPT}" >&2
    exit 1
fi

echo "[INFO] GPU_ID=${GPU_ID}"
echo "[INFO] TEXTURES_PATH=${TEXTURES_PATH}"
echo "[INFO] SOURCE_DATASET_PATH=${SOURCE_DATASET_PATH}"
echo "[INFO] OUTPUT_DATASET_PATH=${OUTPUT_DATASET_PATH}"
echo "[INFO] MATERIAL_RANGE=[${MATERIAL_START}, ${MATERIAL_STOP})"

for (( material_index=MATERIAL_START; material_index<MATERIAL_STOP; material_index++ )); do
    echo "[INFO] Starting material index ${material_index}"
    export EGL_DEVICE_ID="${GPU_ID}"
    python "${PY_SCRIPT}" \
        --gpu-id "${GPU_ID}" \
        --textures-path "${TEXTURES_PATH}" \
        --source-dataset-path "${SOURCE_DATASET_PATH}" \
        --output-dataset-path "${OUTPUT_DATASET_PATH}" \
        --material-index "${material_index}" \
        --skip-done
done

echo "[INFO] Finished material range [${MATERIAL_START}, ${MATERIAL_STOP})"
