import random
import subprocess
import csv
from copy import deepcopy
import os
import re
# --- Filter thresholds ---
HORIZON_1_LIMIT = 11.99
HORIZON_ALL_LIMIT = 39.99
TOTAL_SMAPE_LIMIT = 30

results_file = "tuning_results.csv"

# --- Sample a random config ---
def sample_random_config(space):
    return {param: random.choice(values) for param, values in space.items()}

# --- Generate neighbors by changing one hyperparameter ---
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

def extract_latest_metrics():
    log_path = "/content/PatchTST-AutoTuner2Muga/result_long_term_forecast.txt"

    if not os.path.exists(log_path):
        return None, None, None, None

    with open(log_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Extract last block of output (between "setting" lines)
    blocks = []
    block = []
    for line in lines:
        if line.lower().startswith("setting"):
            if block:
                blocks.append(block)
                block = []
        block.append(line)
    if block:
        blocks.append(block)

    if not blocks:
        return None, None, None, None

    # Take the last full block
    last_block = blocks[-1]

    horizon_smapes = []
    total_smape = mae = mse = None

    for line in last_block:
        if line.startswith("horizon:"):
            try:
                val = float(line.split("smape:")[-1])
                horizon_smapes.append(val)
            except:
                pass
        elif "Total_SMAPE:" in line or "Overall SMAPE:" in line or "SMAPE:" in line:
            try:
                total_smape = float(line.split(":")[-1])
            except:
                pass
        


        elif "mse:" in line.lower() and "mae:" in line.lower():
             try:
                mse_match = re.search(r"mse:([\d\.]+)", line, re.IGNORECASE)
                mae_match = re.search(r"mae:([\d\.]+)", line, re.IGNORECASE)
                if mse_match:
                     mse = float(mse_match.group(1))
                if mae_match:
                     mae = float(mae_match.group(1))
             except:
               pass
        
    return horizon_smapes, total_smape, mae, mse

# --- Filter condition check ---
def is_valid(horizon_smapes, total_smape):
    if not horizon_smapes or len(horizon_smapes) < 12:
        return False
    if horizon_smapes[0] > HORIZON_1_LIMIT:
        return False
    if any(h > HORIZON_ALL_LIMIT for h in horizon_smapes):
        return False
    if total_smape is None or total_smape > TOTAL_SMAPE_LIMIT:
        return False
    return True

# --- Train model and apply filtering ---
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
        "--patch_size", str(config["patch_size"]),
        "--stride", str(config["stride"]),
        "--itr", "1"
    ]
    subprocess.run(cmd, cwd=project_root)
    print(f"\n[Running Config] Patch Size: {config['patch_size']}, Stride: {config['stride']}")

    # Extract and filter
    horizon_smapes, total_smape, mae, mse = extract_latest_metrics()
    if is_valid(horizon_smapes, total_smape):
        return total_smape, mae, mse
    return None, None, None

# --- Log results to CSV ---
def log_result(score, config, mae, mse):
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([score, config, mae, mse])

# --- Main tuning loop ---
def run_autotuner(n_random=30, greedy=True):
    trials = []

    # Random Search
    for trial in range(n_random):
        config = sample_random_config(search_space)
        if config["stride"] >= config["patch_size"]:
            continue
        smape, mae, mse = train_patchtst(config)
        if smape is not None:
            trials.append((smape, config))
            log_result(smape, config, mae, mse)

    # Greedy Search on Neighbors
    if greedy:
        top_configs = sorted(trials, key=lambda x: x[0])[:5]
        for base_score, base_config in top_configs:
            neighbors = get_neighbors(base_config, search_space)
            for neighbor in neighbors:
                if neighbor["stride"] >= neighbor["patch_size"]:
                    continue
                smape, mae, mse = train_patchtst(neighbor)
                if smape is not None:
                    trials.append((smape, neighbor))
                    log_result(smape, neighbor, mae, mse)

    if trials:
        best_trial = min(trials, key=lambda x: x[0])
        print(f"\n✅ Best SMAPE: {best_trial[0]:.4f} with config: {best_trial[1]}")
    else:
        print("\n❌ No valid configurations found after filtering.")

# --- Search space ---
search_space = {
    "seq_len": [48,146],
    "e_layers": [4],
    "n_heads": [8],
    "d_ff": [2048],
    "d_model": [1024],
    "learning_rate": [0.0001],
    "batch_size": [32],
    "patch_size": [12, 16, 18, 20, 24],
    "stride": [4, 10, 12, 14]
}

# --- Entry Point ---
if __name__ == "__main__":
    run_autotuner()
