#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
TEXTURES_PATH="${TEXTURES_PATH:-${REPO_ROOT}/cc0textures-512}"
SOURCE_DATASET_PATH="${SOURCE_DATASET_PATH:-${REPO_ROOT}/dji-action4-real}"
OUTPUT_DATASET_PATH="${OUTPUT_DATASET_PATH:-${REPO_ROOT}/dji-action4-real-wrist-occlusion}"
WRIST_GLB="${WRIST_GLB:-${REPO_ROOT}/dji-action4-real-with-hand/wrist3d.glb}"
PY_SCRIPT="${PY_SCRIPT:-${REPO_ROOT}/scripts/render_dji_action4_wrist_scene.py}"
MATERIAL_START="${MATERIAL_START:-0}"
MATERIAL_STOP="${MATERIAL_STOP:-50}"
OBJECT_COUNT="${OBJECT_COUNT:-2}"
VIEWS_PER_SCENE="${VIEWS_PER_SCENE:-20}"
OCCLUSION_PROFILE="${OCCLUSION_PROFILE:-medium}"
WRIST_DECIMATE_RATIO="${WRIST_DECIMATE_RATIO:-0.25}"
RENDER_SAMPLES="${RENDER_SAMPLES:-32}"

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

if [[ ! "${OBJECT_COUNT}" =~ ^[0-9]+$ ]] || (( OBJECT_COUNT < 1 )); then
    echo "[ERROR] OBJECT_COUNT must be a positive integer: ${OBJECT_COUNT}" >&2
    exit 1
fi

if [[ ! "${VIEWS_PER_SCENE}" =~ ^[0-9]+$ ]] || (( VIEWS_PER_SCENE < 1 )); then
    echo "[ERROR] VIEWS_PER_SCENE must be a positive integer: ${VIEWS_PER_SCENE}" >&2
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
echo "[INFO] WRIST_GLB=${WRIST_GLB}"
echo "[INFO] OBJECT_COUNT=${OBJECT_COUNT}"
echo "[INFO] VIEWS_PER_SCENE=${VIEWS_PER_SCENE}"
echo "[INFO] OCCLUSION_PROFILE=${OCCLUSION_PROFILE}"
echo "[INFO] WRIST_DECIMATE_RATIO=${WRIST_DECIMATE_RATIO}"
echo "[INFO] RENDER_SAMPLES=${RENDER_SAMPLES}"
echo "[INFO] MATERIAL_RANGE=[${MATERIAL_START}, ${MATERIAL_STOP})"

for (( material_index=MATERIAL_START; material_index<MATERIAL_STOP; material_index++ )); do
    echo "[INFO] Starting wrist material index ${material_index}"
    export EGL_DEVICE_ID="${GPU_ID}"
    python "${PY_SCRIPT}" \
        --gpu-id "${GPU_ID}" \
        --textures-path "${TEXTURES_PATH}" \
        --source-dataset-path "${SOURCE_DATASET_PATH}" \
        --output-dataset-path "${OUTPUT_DATASET_PATH}" \
        --wrist-glb "${WRIST_GLB}" \
        --object-count "${OBJECT_COUNT}" \
        --views-per-scene "${VIEWS_PER_SCENE}" \
        --occlusion-profile "${OCCLUSION_PROFILE}" \
        --wrist-decimate-ratio "${WRIST_DECIMATE_RATIO}" \
        --render-samples "${RENDER_SAMPLES}" \
        --material-index "${material_index}" \
        --skip-done
done

echo "[INFO] Finished wrist material range [${MATERIAL_START}, ${MATERIAL_STOP})"
