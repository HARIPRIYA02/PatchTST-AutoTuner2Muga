import random
import subprocess
import csv
from collections import defaultdict
from copy import deepcopy
import os

# Corrected log path resolution

def extract_metrics_from_log():
    smape = mae = mse = None
    script_dir = os.path.dirname(__file__)
    log_path = os.path.abspath(os.path.join(script_dir, "../../../result_long_term_forecast.txt"))

    if not os.path.exists(log_path):
        return 0.0, 0.0, 0.0

    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines[::-1]:  # reverse search to find latest values first
            if "Total_SMAPE:" in line:
                smape = float(line.strip().split(":")[-1])
            elif "MAE:" in line:
                mae = float(line.strip().split(":")[-1])
            elif "MSE:" in line:
                mse = float(line.strip().split(":")[-1])
            if smape is not None and mae is not None and mse is not None:
                break

    if smape is None or mae is None or mse is None:
        return 0.0, 0.0, 0.0

    return smape, mae, mse

search_space = {
    "seq_len": [36, 48, 60, 72, 84, 96, 108, 120, 134, 146],
    "e_layers": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "n_heads": [2, 4, 6, 8],
    "d_ff": [512, 1024, 2048],
    "d_model": [512, 1024, 2048],
    "learning_rate": [0.0001, 0.00001, 0.001],
    "batch_size": [16, 32]
}

results_file = "tuning_results.csv"
value_scores = defaultdict(lambda: defaultdict(list))

def sample_random_config(space):
    return {param: random.choice(values) for param, values in space.items()}

def get_neighbors(config, space):
    neighbors = []
    for param in config:
        values = space[param]
        idx = values.index(config[param])
        for delta in [-1, 1]:
            new_idx = idx + delta
            if 0 <= new_idx < len(values):
                new_config = deepcopy(config)
                new_config[param] = values[new_idx]
                neighbors.append(new_config)
    return neighbors

def train_patchtst(config):
    model_id = f"ili_{config['seq_len']}_12_el{config['e_layers']}_nh{config['n_heads']}_dm{config['d_model']}_df{config['d_ff']}"
    script_dir = os.path.dirname(__file__)
    run_path = os.path.abspath(os.path.join(script_dir, "../../../run.py"))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../"))

    cmd = [
        "python", "-u", run_path,
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", "./dataset/illness/",
        "--data_path", "national_newsta.csv",
        "--model_id", model_id,
        "--model", "PatchTST",
        "--data", "custom",
        "--features", "S",
        "--target", "ILITOTAL",
        "--seq_len", str(config["seq_len"]),
        "--label_len", "18",
        "--pred_len", "12",
        "--e_layers", str(config["e_layers"]),
        "--d_layers", "1",
        "--factor", "3",
        "--enc_in", "7",
        "--dec_in", "7",
        "--c_out", "7",
        "--des", "Exp",
        "--n_heads", str(config["n_heads"]),
        "--d_model", str(config["d_model"]),
        "--d_ff", str(config["d_ff"]),
        "--learning_rate", str(config["learning_rate"]),
        "--batch_size", str(config["batch_size"]),
        "--itr", "1"
    ]
    subprocess.run(cmd, cwd=project_root)
    return extract_metrics_from_log()

def prune_search_space(search_space, value_scores, threshold=0.15, min_trials=3):
    for param in list(search_space.keys()):
        all_scores = [s for v in value_scores[param].values() for s in v]
        if len(all_scores) < 10:
            continue
        global_avg = sum(all_scores) / len(all_scores)
        for val in list(search_space[param]):
            scores = value_scores[param][val]
            if len(scores) < min_trials:
                continue
            avg_score = sum(scores) / len(scores)
            if avg_score > global_avg + threshold:
                print(f"Pruned {param}={val} (avg SMAPE: {avg_score:.3f})")
                search_space[param].remove(val)

def log_result(score, config, smape, mae, mse):
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([score, config, smape, mae, mse])

def run_autotuner(n_random=30, greedy=True, prune=True):
    trials = []
    for trial in range(n_random):
        config = sample_random_config(search_space)
        if config["d_model"] == config["d_ff"]:
            continue
        smape, mae, mse = train_patchtst(config)
        score = smape
        trials.append((score, config))
        for k, v in config.items():
            value_scores[k][v].append(score)
        log_result(score, config, smape, mae, mse)
        if prune and trial > 10 and trial % 5 == 0:
            prune_search_space(search_space, value_scores)

    if greedy:
        top_configs = sorted(trials, key=lambda x: x[0])[:5]
        for base_score, base_config in top_configs:
            neighbors = get_neighbors(base_config, search_space)
            for neighbor in neighbors:
                if neighbor["d_model"] == neighbor["d_ff"]:
                    continue
                smape, mae, mse = train_patchtst(neighbor)
                score = smape
                trials.append((score, neighbor))
                log_result(score, neighbor, smape, mae, mse)

    best_trial = min(trials, key=lambda x: x[0])
    print(f"\n✅ Best SMAPE: {best_trial[0]:.4f} with config: {best_trial[1]}")

if __name__ == "__main__":
    run_autotuner()
