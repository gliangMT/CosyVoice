#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
#
# Engineering from-scratch training:
# - Randomly initialize the selected CosyVoice module (llm/flow/hifigan).
# - Keep the pretrained Qwen backbone/tokenizer, speech tokenizer, and CAM++.
# - Never warm-start from pretrained llm.pt, flow.pt, or hift.pt.
# - --resume may restore a checkpoint produced by this scratch experiment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${PYTHONPATH:-}"
. ./path.sh || exit 1

WORK_DIR="$(pwd)"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${WORK_DIR}/logs/cosyvoice3_scratch_${TIMESTAMP}"
EXP_ROOT="${WORK_DIR}/exp/cosyvoice3_scratch"
TENSORBOARD_ROOT="${WORK_DIR}/tensorboard/cosyvoice3_scratch"
mkdir -p "${LOG_DIR}"

stage="${stage:-5}"
stop_stage="${stop_stage:-5}"

data_url="${data_url:-www.openslr.org/resources/60}"
data_dir="${data_dir:-/home/cosyvoice-test/data/libritts}"
pretrained_model_dir="${pretrained_model_dir:-/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B}"

# Train one module per launch because LLM and Flow use different train_conf.
# MODELS="${MODELS:-llm}"
MODELS="${MODELS:-flow}"

if [ "$(wc -w <<< "${MODELS}")" -ne 1 ]; then
  echo "Select exactly one model per scratch run: MODELS=llm or MODELS=flow." >&2
  exit 2
fi

for arg in "$@"; do
  case "${arg}" in
    --checkpoint|--checkpoint=*)
      echo "This scratch script does not accept --checkpoint." >&2
      exit 2
      ;;
  esac
done

if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
  echo "Data Download"
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    local/download_and_untar.sh "${data_dir}" "${data_url}" "${part}"
  done
fi

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    mkdir -p "data/${x}"
    python local/prepare_data.py \
      --src_dir "${data_dir}/LibriTTS/${x}" \
      --des_dir "data/${x}" \
      --instruct "You are a helpful assistant.<|endofprompt|>"
  done
fi

# These pretrained extractors are intentionally retained in engineering scratch.
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  echo "Extract CAM++ speaker embeddings"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    ../../../tools/extract_embedding.py \
      --dir "data/${x}" \
      --onnx_path "${pretrained_model_dir}/campplus.onnx"
  done
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  echo "Extract discrete speech tokens"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    ../../../tools/extract_speech_token.py \
      --dir "data/${x}" \
      --onnx_path "${pretrained_model_dir}/speech_tokenizer_v3.onnx"
  done
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
  echo "Prepare parquet data"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    mkdir -p "data/${x}/parquet"
    ../../../tools/make_parquet_list.py \
      --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir "data/${x}" \
      --des_dir "data/${x}/parquet"
  done
fi

export COSYVOICE_ONNX_PROVIDER="${COSYVOICE_ONNX_PROVIDER:-musa}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3200000}"
export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-musa}"

export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export MCCL_ALGOS="${MCCL_ALGOS:-1}"
export MCCL_BUFFSIZE="${MCCL_BUFFSIZE:-20971520}"
export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-14}"
export MCCL_CHECK_POINTERS="${MCCL_CHECK_POINTERS:-0}"
export MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-3}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"

num_gpus="$(awk -F, '{print NF}' <<< "${MUSA_VISIBLE_DEVICES}")"
job_id="${job_id:-1986}"
dist_backend="${dist_backend:-mccl}"
num_workers="${num_workers:-2}"
prefetch="${prefetch:-100}"
train_engine="${train_engine:-torch_ddp}"
rdzv_endpoint="${rdzv_endpoint:-localhost:1234}"

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
  echo "Engineering scratch training: ${MODELS}"
  echo "No pretrained CosyVoice module checkpoint will be loaded."

  cat data/{train-clean-100,train-clean-360,train-other-500}/parquet/data.list > data/train.data.list
  cat data/{dev-clean,dev-other}/parquet/data.list > data/dev.data.list

  for model in ${MODELS}; do
    case "${model}" in
      llm|flow|hifigan) ;;
      *)
        echo "Unsupported model '${model}'. Expected llm, flow, or hifigan." >&2
        exit 2
        ;;
    esac

    model_dir="${EXP_ROOT}/${model}/${train_engine}"
    tensorboard_dir="${TENSORBOARD_ROOT}/${model}/${train_engine}"
    log_file="${LOG_DIR}/${model}_${TIMESTAMP}.log"
    mkdir -p "${model_dir}" "${tensorboard_dir}"

    echo "Starting randomly initialized ${model}; log: ${log_file}"
    torchrun \
      --nnodes=1 \
      --nproc_per_node="${num_gpus}" \
      --rdzv_id="${job_id}_${model}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${rdzv_endpoint}" \
      ../../../cosyvoice/bin/train.py \
      --train_engine "${train_engine}" \
      --config conf/cosyvoice3.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path "${pretrained_model_dir}/CosyVoice-BlankEN" \
      --onnx_path "${pretrained_model_dir}" \
      --model "${model}" \
      --model_dir "${model_dir}" \
      --tensorboard_dir "${tensorboard_dir}" \
      --ddp.dist_backend "${dist_backend}" \
      --num_workers "${num_workers}" \
      --prefetch "${prefetch}" \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --resume auto \
      --save_per_step 1000 \
      --deepspeed.save_states model+optimizer \
      "$@" 2>&1 | tee "${log_file}"
  done
fi

echo "Finished. Logs: ${LOG_DIR}"
