#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SCENE_NUM="${SCENE_NUM:-1}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/dji-action4}"
TEXTURES_PATH="${TEXTURES_PATH:-${REPO_ROOT}/cc0textures-512}"
GEN_SCRIPT="${GEN_SCRIPT:-${REPO_ROOT}/s2_p1_gen_pbr_data.py}"

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "[ERROR] DATASET_PATH does not exist: ${DATASET_PATH}" >&2
    exit 1
fi

if [[ ! -d "${DATASET_PATH}/models" ]]; then
    echo "[ERROR] DATASET_PATH/models does not exist: ${DATASET_PATH}/models" >&2
    exit 1
fi

if [[ ! -f "${DATASET_PATH}/models/models_info.json" ]]; then
    echo "[ERROR] Missing models_info.json. Run: python s1_p3_obj_infos.py --dataset-path dji-action4" >&2
    exit 1
fi

if [[ ! -d "${TEXTURES_PATH}" ]]; then
    echo "[ERROR] TEXTURES_PATH does not exist: ${TEXTURES_PATH}" >&2
    exit 1
fi

echo "[INFO] GPU_ID=${GPU_ID}"
echo "[INFO] SCENE_NUM=${SCENE_NUM}"
echo "[INFO] DATASET_PATH=${DATASET_PATH}"
echo "[INFO] TEXTURES_PATH=${TEXTURES_PATH}"

bash "${REPO_ROOT}/s2_p1_gen_pbr_data.sh" \
    "${GPU_ID}" \
    "${SCENE_NUM}" \
    "${TEXTURES_PATH}" \
    "${DATASET_PATH}" \
    "${GEN_SCRIPT}"
