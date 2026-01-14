#!/bin/bash


####################################
#### Run surprisal experiments on VUAMC dataset ####
#######################################
echo "Running surprisal experiments on VUAMC dataset"

INPUT="datasets/VUA-METANOV_mod.json"
OUTPUT="results"

# List of model IDs
models=(
    "openai-community/gpt2"
    "openai-community/gpt2-medium"
    "openai-community/gpt2-large"
    "openai-community/gpt2-xl"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
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


####################################
#### Run surprisal experiments on Lai2009 dataset ####
#######################################

echo "Running surprisal experiments on Lai2009 dataset"

INPUT="datasets/Lai2009-METANOV_mod.json"
OUTPUT="results"

# List of model IDs
models=(
    "openai-community/gpt2"
    "openai-community/gpt2-medium"
    "openai-community/gpt2-large"
    "openai-community/gpt2-xl"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
)


for model in "${models[@]}"; do
    echo "Running model: $model"
    
    python src/record_surprisal_synthetic.py \
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

####################################
### Run surprisal experiments on GPT-4o dataset ####
######################################


echo "Running surprisal experiments on GPT-4o dataset"

INPUT="datasets/GPT-4o-METANOV_mod.json"
OUTPUT="results"

# List of model IDs
models=(
    "openai-community/gpt2"
    "openai-community/gpt2-medium"
    "openai-community/gpt2-large"
    "openai-community/gpt2-xl"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
)


for model in "${models[@]}"; do
    echo "Running model: $model"
    
    python src/record_surprisal_synthetic.py \
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
