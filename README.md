# UCADM-IDS
Urban Context-Aware Drift Model for Adaptive Network Intrusion Detection.
## About
This repository contains the implementation of UCADM-IDS, an adaptive intrusion detection framework that models urban context-specific concept drift. The framework combines:

- **UCADM**: Translates urban phenomena (Riyadh Season, Ramadan, infrastructure growth) into drift parameters
- **FFNN**: Lightweight neural network (128-64-32) with online learning
- **ICE-CP**: Class-perception drift detector with 0.05% false alarm rate
Evaluated on CSE-CIC-IDS2018 V2 (3.15M flows) across 5 drift scenarios, achieving 97-99% accuracy.

## Environment Setup
Check the Environment.txt file for a complete setup configuration. 

Dataset
Download the CSE-CIC-IDS2018 V2 dataset from Zenodo:
https://doi.org/10.5281/zenodo.10149295
