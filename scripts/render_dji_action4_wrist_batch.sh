#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
TEXTURES_PATH="${TEXTURES_PATH:-${REPO_ROOT}/cc0textures-512}"
SOURCE_DATASET_PATH="${SOURCE_DATASET_PATH:-${REPO_ROOT}/dji-action4-real}"
OUTPUT_DATASET_PATH="${OUTPUT_DATASET_PATH:-${REPO_ROOT}/dji-action4-real-wrist-occlusion}"
WRIST_GLB="${WRIST_GLB:-${REPO_ROOT}/dji-action4-real-with-hand/wrist3d.glb}"
JACKET_GLB="${JACKET_GLB:-${REPO_ROOT}/dji-action4-real-with-hand/jack3d.glb}"
JACKET_ENABLED="${JACKET_ENABLED:-1}"
PY_SCRIPT="${PY_SCRIPT:-${REPO_ROOT}/scripts/render_dji_action4_wrist_scene.py}"
MATERIAL_START="${MATERIAL_START:-0}"
MATERIAL_STOP="${MATERIAL_STOP:-50}"
OBJECT_COUNT="${OBJECT_COUNT:-2}"
VIEWS_PER_SCENE="${VIEWS_PER_SCENE:-20}"
OCCLUSION_PROFILE="${OCCLUSION_PROFILE:-medium}"
WRIST_SIZE_SCALE="${WRIST_SIZE_SCALE:-1.0}"
WRIST_DECIMATE_RATIO="${WRIST_DECIMATE_RATIO:-0.25}"
JACKET_SIZE_SCALE="${JACKET_SIZE_SCALE:-1.0}"
JACKET_DECIMATE_RATIO="${JACKET_DECIMATE_RATIO:-0.10}"
JACKET_TOP_Z="${JACKET_TOP_Z:-0.02}"
JACKET_FRONT_UP_AXIS="${JACKET_FRONT_UP_AXIS:-neg_y}"
ROOM_FLOOR_GAP="${ROOM_FLOOR_GAP:-0.03}"
ROOM_UV_TILE_SIZE="${ROOM_UV_TILE_SIZE:-0.50}"
TARGET_SIDE_UP_PROB="${TARGET_SIDE_UP_PROB:-0.8}"
TARGET_SIDE_UP_AXIS="${TARGET_SIDE_UP_AXIS:-y}"
RENDER_SAMPLES="${RENDER_SAMPLES:-32}"
CAMERA_MODE="${CAMERA_MODE:-orbit}"
ORBIT_RADIUS="${ORBIT_RADIUS:-0.55}"
ORBIT_DISTANCE="${ORBIT_DISTANCE:-${ORBIT_RADIUS}}"
ORBIT_PITCH_DEG="${ORBIT_PITCH_DEG:-}"
ORBIT_PITCH_MIN_DEG="${ORBIT_PITCH_MIN_DEG:-20}"
ORBIT_PITCH_MAX_DEG="${ORBIT_PITCH_MAX_DEG:-60}"
ORBIT_PITCH_SAMPLE_MODE="${ORBIT_PITCH_SAMPLE_MODE:-per_frame}"
ORBIT_HIGH_PITCH_PROB="${ORBIT_HIGH_PITCH_PROB:-0.7}"
ORBIT_HIGH_PITCH_MIN_DEG="${ORBIT_HIGH_PITCH_MIN_DEG:-50}"
ORBIT_HIGH_PITCH_MAX_DEG="${ORBIT_HIGH_PITCH_MAX_DEG:-89}"
ORBIT_LOW_PITCH_MIN_DEG="${ORBIT_LOW_PITCH_MIN_DEG:-0}"
ORBIT_LOW_PITCH_MAX_DEG="${ORBIT_LOW_PITCH_MAX_DEG:-50}"
ORBIT_ARC_DEG="${ORBIT_ARC_DEG:-360}"
ORBIT_ROLL_DEG="${ORBIT_ROLL_DEG:-0}"

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
echo "[INFO] JACKET_ENABLED=${JACKET_ENABLED}"
echo "[INFO] JACKET_GLB=${JACKET_GLB}"
echo "[INFO] OBJECT_COUNT=${OBJECT_COUNT}"
echo "[INFO] VIEWS_PER_SCENE=${VIEWS_PER_SCENE}"
echo "[INFO] OCCLUSION_PROFILE=${OCCLUSION_PROFILE}"
echo "[INFO] WRIST_SIZE_SCALE=${WRIST_SIZE_SCALE}"
echo "[INFO] WRIST_DECIMATE_RATIO=${WRIST_DECIMATE_RATIO}"
echo "[INFO] JACKET_SIZE_SCALE=${JACKET_SIZE_SCALE}"
echo "[INFO] JACKET_DECIMATE_RATIO=${JACKET_DECIMATE_RATIO}"
echo "[INFO] JACKET_TOP_Z=${JACKET_TOP_Z}"
echo "[INFO] JACKET_FRONT_UP_AXIS=${JACKET_FRONT_UP_AXIS}"
echo "[INFO] ROOM_FLOOR_GAP=${ROOM_FLOOR_GAP}"
echo "[INFO] ROOM_UV_TILE_SIZE=${ROOM_UV_TILE_SIZE}"
echo "[INFO] TARGET_SIDE_UP_PROB=${TARGET_SIDE_UP_PROB}"
echo "[INFO] TARGET_SIDE_UP_AXIS=${TARGET_SIDE_UP_AXIS}"
echo "[INFO] RENDER_SAMPLES=${RENDER_SAMPLES}"
echo "[INFO] CAMERA_MODE=${CAMERA_MODE}"
echo "[INFO] ORBIT_RADIUS=${ORBIT_RADIUS}"
echo "[INFO] ORBIT_DISTANCE=${ORBIT_DISTANCE}"
echo "[INFO] ORBIT_PITCH_DEG=${ORBIT_PITCH_DEG:-random}"
echo "[INFO] ORBIT_PITCH_MIN_DEG=${ORBIT_PITCH_MIN_DEG}"
echo "[INFO] ORBIT_PITCH_MAX_DEG=${ORBIT_PITCH_MAX_DEG}"
echo "[INFO] ORBIT_PITCH_SAMPLE_MODE=${ORBIT_PITCH_SAMPLE_MODE}"
echo "[INFO] ORBIT_HIGH_PITCH_PROB=${ORBIT_HIGH_PITCH_PROB}"
echo "[INFO] ORBIT_HIGH_PITCH_MIN_DEG=${ORBIT_HIGH_PITCH_MIN_DEG}"
echo "[INFO] ORBIT_HIGH_PITCH_MAX_DEG=${ORBIT_HIGH_PITCH_MAX_DEG}"
echo "[INFO] ORBIT_LOW_PITCH_MIN_DEG=${ORBIT_LOW_PITCH_MIN_DEG}"
echo "[INFO] ORBIT_LOW_PITCH_MAX_DEG=${ORBIT_LOW_PITCH_MAX_DEG}"
echo "[INFO] ORBIT_ARC_DEG=${ORBIT_ARC_DEG}"
echo "[INFO] ORBIT_ROLL_DEG=${ORBIT_ROLL_DEG}"
echo "[INFO] MATERIAL_RANGE=[${MATERIAL_START}, ${MATERIAL_STOP})"

orbit_pitch_args=()
if [[ -n "${ORBIT_PITCH_DEG}" ]]; then
    orbit_pitch_args+=(--orbit-pitch-deg "${ORBIT_PITCH_DEG}")
else
    orbit_pitch_args+=(--orbit-pitch-min-deg "${ORBIT_PITCH_MIN_DEG}")
    orbit_pitch_args+=(--orbit-pitch-max-deg "${ORBIT_PITCH_MAX_DEG}")
    orbit_pitch_args+=(--orbit-pitch-sample-mode "${ORBIT_PITCH_SAMPLE_MODE}")
    orbit_pitch_args+=(--orbit-high-pitch-prob "${ORBIT_HIGH_PITCH_PROB}")
    orbit_pitch_args+=(--orbit-high-pitch-min-deg "${ORBIT_HIGH_PITCH_MIN_DEG}")
    orbit_pitch_args+=(--orbit-high-pitch-max-deg "${ORBIT_HIGH_PITCH_MAX_DEG}")
    orbit_pitch_args+=(--orbit-low-pitch-min-deg "${ORBIT_LOW_PITCH_MIN_DEG}")
    orbit_pitch_args+=(--orbit-low-pitch-max-deg "${ORBIT_LOW_PITCH_MAX_DEG}")
fi

jacket_enabled_args=(--jacket-enabled)
case "${JACKET_ENABLED}" in
    0|false|False|FALSE|no|No|NO)
        jacket_enabled_args=(--no-jacket-enabled)
        ;;
esac

for (( material_index=MATERIAL_START; material_index<MATERIAL_STOP; material_index++ )); do
    echo "[INFO] Starting wrist material index ${material_index}"
    export EGL_DEVICE_ID="${GPU_ID}"
    python "${PY_SCRIPT}" \
        --gpu-id "${GPU_ID}" \
        --textures-path "${TEXTURES_PATH}" \
        --source-dataset-path "${SOURCE_DATASET_PATH}" \
        --output-dataset-path "${OUTPUT_DATASET_PATH}" \
        --wrist-glb "${WRIST_GLB}" \
        "${jacket_enabled_args[@]}" \
        --jacket-glb "${JACKET_GLB}" \
        --object-count "${OBJECT_COUNT}" \
        --views-per-scene "${VIEWS_PER_SCENE}" \
        --occlusion-profile "${OCCLUSION_PROFILE}" \
        --wrist-size-scale "${WRIST_SIZE_SCALE}" \
        --wrist-decimate-ratio "${WRIST_DECIMATE_RATIO}" \
        --jacket-size-scale "${JACKET_SIZE_SCALE}" \
        --jacket-decimate-ratio "${JACKET_DECIMATE_RATIO}" \
        --jacket-top-z "${JACKET_TOP_Z}" \
        --jacket-front-up-axis "${JACKET_FRONT_UP_AXIS}" \
        --room-floor-gap "${ROOM_FLOOR_GAP}" \
        --room-uv-tile-size "${ROOM_UV_TILE_SIZE}" \
        --target-side-up-prob "${TARGET_SIDE_UP_PROB}" \
        --target-side-up-axis "${TARGET_SIDE_UP_AXIS}" \
        --render-samples "${RENDER_SAMPLES}" \
        --camera-mode "${CAMERA_MODE}" \
        --orbit-radius "${ORBIT_RADIUS}" \
        --orbit-distance "${ORBIT_DISTANCE}" \
        "${orbit_pitch_args[@]}" \
        --orbit-arc-deg "${ORBIT_ARC_DEG}" \
        --orbit-roll-deg "${ORBIT_ROLL_DEG}" \
        --material-index "${material_index}" \
        --skip-done
done

echo "[INFO] Finished wrist material range [${MATERIAL_START}, ${MATERIAL_STOP})"
