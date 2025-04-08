#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Navigate to the directory containing your tuner script
cd scripts/long_term_forecast/ILI_script

# Run the autotuner Python script
python3 tunerpa.py
