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

**Run Experiments in Jupyter Notebook**
Creating a New Project in Jupyter Notebook
Step 1: Launch Jupyter Notebook
Open your terminal or command prompt and navigate to your project directory:


cd D:/ConceptDrift
jupyter notebook
Your default web browser will open with the Jupyter interface.

Step 2: Create a New Notebook
In the Jupyter interface, click on the code/ folder to enter it.

Click the "New" button (top right) and select "Python 3 (ipykernel)".

A new notebook will open in a new tab.

Rename the notebook by clicking the title "Untitled" and typing: UCADM_Experiments.

Step 3: Add the Code to a Cell
Copy the entire code provided in this repository.

Paste it into the first cell of your new notebook.

Step 4: Run the Code
Click the "Run" button (▶) or press Shift + Enter to execute the cell.

The code will begin executing. You will see progress output as it runs.

For a quick test, the code uses nrows=50000 (50,000 samples). For the full dataset, change this to nrows=None.

The complete run on the full dataset (3.15 million samples) takes approximately 6–12 hours.

Step 5: Save Your Work
After the code completes:

Click File → Save and Checkpoint.

Close the notebook tab.

Shut down the Jupyter server by pressing Ctrl + C twice in the terminal.

Output Files Generated
After a successful run, the following files are automatically saved to your project folders:
📊 CSV Files (in results/ folder)
📈 Figures (in figures/ folder)
🧠 Model File (in models/ folder)
File Name	Description
final_streaming_model.keras	The trained LSTM model with all weights saved. Can be loaded for inference or further fine-tuning: model = tf.keras.models.load_model('models/final_streaming_model.keras')
