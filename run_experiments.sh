#!/bin/bash

echo "Running surprisal experiments on VUA-METANOV dataset"

INPUT="dataset/VUA-METANOV_mod.json"
OUTPUT="results_surprisal"

models=(
  "EleutherAI/pythia-70m"
  "EleutherAI/pythia-160m"
  "EleutherAI/pythia-410m"
  "EleutherAI/pythia-1b"
  "EleutherAI/pythia-1.4b"
  "EleutherAI/pythia-2.8b"
  "EleutherAI/pythia-6.9b"
  "EleutherAI/pythia-12b"
)


for model in "${models[@]}"; do
    echo "Running model: $model"
    python src/record_surprisal_vua.py \
      --input "$INPUT" \
      --output "$OUTPUT" \
      --model "$model" \
      --dtype float32 \
      --device-map-auto \
      --pimentel-fix \
      --cloze

    echo "Finished model: $model"
    echo "---------------------------"
done

echo "All runs finished"




echo "Running checkpoints surprisal experiments on VUA-METANOV dataset"

INPUT="dataset/VUA-METANOV_mod.json"
OUTPUT="results_surprisal"

MODEL="EleutherAI/pythia-70m"

CHECKPOINTS=()

EARLY_STEPS=(0 1 2 4 8 16 32 64 128 256 512)
for s in "${EARLY_STEPS[@]}"; do
  CHECKPOINTS+=("step${s}")
done

for s in $(seq 1000 1000 143000); do
  CHECKPOINTS+=("step${s}")
done

echo "Total checkpoints: ${#CHECKPOINTS[@]}"
if [[ "${#CHECKPOINTS[@]}" -ne 154 ]]; then
  echo "ERROR: Expected 154 checkpoints, got ${#CHECKPOINTS[@]}"
  exit 1
fi

for i in "${!CHECKPOINTS[@]}"; do
  ckpt="${CHECKPOINTS[$i]}"
  idx="$(printf "%03d" "$i")"

  echo "Running checkpoint #$idx : $MODEL @ $ckpt"

  python src/record_surprisal_vua.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --model-revision "$ckpt" \
    --dtype float32 \
    --device-map-auto \
    --pimentel-fix \
    --cloze

  echo "Finished checkpoint: $ckpt"
  echo "---------------------------"
done

echo "All runs finished"
