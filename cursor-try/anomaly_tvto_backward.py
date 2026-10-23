"""Run the current tvto training with autograd anomaly detection; user code unchanged."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

import main

torch.autograd.set_detect_anomaly(True, check_nan=False)
main.main()
