#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
model_name=PatchTST
root_path=./dataset/illness/
data_path=national_newsta.csv
target='ILITOTAL'

# Search Space
seq_len=(36 48 60)
e_layers=(2 4 8)
n_heads=(2 8)
d_ff=(1024 2048)
d_model=(1024 2048)
learning_rate=(0.0001)
batch_size=(32)
patch_len=(16 24)
stride=(10 12)

LOG_FILE="autotune_log.txt"
RESULT_FILE="result_long_term_forecast.txt"
> $LOG_FILE
> $RESULT_FILE

run_model() {
  local seq_len=$1
  local patch_len=$2
  local stride=$3
  local e_layers=$4
  local n_heads=$5
  local d_model=$6
  local d_ff=$7
  local learning_rate=$8
  local batch_size=$9

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id ili_36_24 \
    --model $model_name \
    --data custom \
    --features S \
    --target $target \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len 12 \
    --patch_len $patch_len \
    --stride $stride \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate $learning_rate \
    --train_epochs 10 \
    --batch_size $batch_size \
    --itr 1

  echo "$seq_len,$patch_size,$stride,$e_layers,$n_heads,$d_model,$d_ff,$learning_rate,$batch_size" >> $LOG_FILE
}

# RANDOM SEARCH
for i in {1..4}; do
  seq_len=${seq_len[$RANDOM % ${#seq_len[@]}]}
  patch_len=${patch_len[$RANDOM % ${#patch_len[@]}]}
  stride=${stride[$RANDOM % ${#stride[@]}]}
  e_layers=${e_layers[$RANDOM % ${#e_layers[@]}]}
  n_heads=${n_heads[$RANDOM % ${#n_heads[@]}]}
  d_model=${d_model[$RANDOM % ${#d_model[@]}]}
  d_ff=${d_ff[$RANDOM % ${#d_ff[@]}]}
  lr=${learning_rate[$RANDOM % ${#learning_rate[@]}]}
  batch_size=${batch_size[$RANDOM % ${#batch_size[@]}]}

  run_model $seq_len $patch_len $stride $e_layers $n_heads $d_model $d_ff $lr $batch_size
done

# READ BEST CONFIG FROM FILE AFTER RANDOM SEARCH
best_line=$(grep "SMAPE:" $RESULT_FILE | sort -t ':' -k2 -n | head -1)
best_smape=$(echo $best_line | awk -F'SMAPE: ' '{print $2}' | awk '{print $1}')
best_config_line=$(grep -B1 "SMAPE: $best_smape" $RESULT_FILE | head -1)

# Extract hyperparameters from the config line
IFS=',' read -r seq_len patch_len stride e_layers n_heads d_model d_ff lr batch_size <<< "$best_config_line"

# GREEDY HILL CLIMBING
neighborhood_search() {
  local -n arr=$1
  local val=$2
  for x in "${arr[@]}"; do
    [ "$x" != "$val" ] && echo "$x"
  done
}

for iter in {1..5}; do
  improved=false
  for param in seq_len patch_len stride e_layers n_heads d_model d_ff lr batch_size; do
    current_val=${!param}
    eval "space=(\"\${${param}s[@]}\")"
    for alt in $(neighborhood_search space "$current_val"); do
      eval "$param=$alt"
      run_model $seq_len $patch_len $stride $e_layers $n_heads $d_model $d_ff $lr $batch_size
      if [ -f $RESULT_FILE ]; then
        new_smape=$(grep "SMAPE:" $RESULT_FILE | tail -1 | awk '{print $2}')
        if (( $(echo "$new_smape < $best_smape" | bc -l) )); then
          best_smape=$new_smape
          best_config="$seq_len $patch_len $stride $e_layers $n_heads $d_model $d_ff $lr $batch_size"
          improved=true
          break
        fi
      fi
      eval "$param=$current_val"
    done
    $improved && break
  done
  ! $improved && break
done

echo "Best config: $best_config with SMAPE: $best_smape" | tee -a $LOG_FILE
