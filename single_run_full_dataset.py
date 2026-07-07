# ============================================================================
# FULL DATASET RUN – ALL 5 SCENARIOS (E1, E2, E3, E4, E5)
# ============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import os

# Configuration
DATA_PATH = "D:/ConceptDrift/data/CSE-CIC-IDS-2018-V2.csv"
NROWS = None  # FULL DATASET
INITIAL_TRAIN = 50000
BATCH_SIZE = 500
ONLINE_EPOCHS = 1

print("="*60)
print("FULL DATASET RUN – ALL 5 SCENARIOS")
print("="*60)

# Load full dataset
print("\nLoading full dataset...")
df = pd.read_csv(DATA_PATH, nrows=NROWS)

# Find label column
label_col = None
for col in df.columns:
    if df[col].dtype == 'object' and 'Benign' in df[col].unique():
        label_col = col
        break
if label_col is None:
    label_col = 'label' if 'label' in df.columns else None

print(f"Using label column: '{label_col}'")

df['label_binary'] = (df[label_col] != 'Benign').astype(int)

# Features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'label_binary']
X = df[numeric_cols].fillna(0).values
y = df['label_binary'].values

print(f"Total samples: {len(X)}")
print(f"Attack ratio: {y.mean():.3f}")

# Split
X_train, X_stream = X[:INITIAL_TRAIN], X[INITIAL_TRAIN:]
y_train, y_stream = y[:INITIAL_TRAIN], y[INITIAL_TRAIN:]

print(f"Training: {X_train.shape[0]}, Streaming: {X_stream.shape[0]}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_stream_scaled = scaler.transform(X_stream)

# ============================================================================
# FFNN Model
# ============================================================================
def create_ffnn(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Class weights
unique, counts = np.unique(y_train, return_counts=True)
total = len(y_train)
class_weights = {0: total / (2 * counts[0]), 1: total / (2 * counts[1])}
print(f"Class weights: {class_weights}")

# Train initial model
base_model = create_ffnn(X_train_scaled.shape[1])
base_model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, 
               class_weight=class_weights, verbose=1)
print("Initial training completed.\n")

# ============================================================================
# Drift Scenario Functions (ALL 5)
# ============================================================================
def apply_riyadh_season(X, y, start_idx=50000, ramp_up=3000, peak_duration=87000, ramp_down=5000, flip_prob=0.3):
    X_d = X.copy()
    y_d = y.copy()
    n = len(X)
    for i in range(n):
        if i < start_idx:
            mult = 1.0
        elif i < start_idx + ramp_up:
            mult = 1.0 + 4.0 * (i - start_idx) / ramp_up
        elif i < start_idx + ramp_up + peak_duration:
            mult = 5.0
        elif i < start_idx + ramp_up + peak_duration + ramp_down:
            mult = 5.0 - 4.0 * (i - (start_idx + ramp_up + peak_duration)) / ramp_down
        else:
            mult = 1.0
        X_d[i] = X[i] * mult
        if start_idx <= i < start_idx + ramp_up + peak_duration + ramp_down:
            if np.random.rand() < flip_prob:
                y_d[i] = 1 - y_d[i]
    return X_d, y_d

def apply_infrastructure_growth(X, y, growth_rate=0.00083):
    X_d = X.copy()
    for i in range(len(X)):
        X_d[i] = X[i] * (1.0 + growth_rate * i)
    return X_d, y

def apply_ramadan(X, y, first_start=100000, cycle_length=45000, ramadan_length=15000, flip_prob=0.7):
    X_d = X.copy()
    y_d = y.copy()
    n = len(X)
    hours = (np.arange(n) // 500) % 24
    for i in range(n):
        if i < first_start:
            continue
        cycle_pos = (i - first_start) % cycle_length
        if cycle_pos >= ramadan_length:
            continue
        hour = hours[i]
        mult = 1.0
        if 20 <= hour or hour < 3:
            mult *= 1.5
        elif 6 <= hour < 18:
            mult *= 0.7
        if hour in [4,11,15,18,22]:
            minute = (i % (60*24)) % 60
            if minute < 30:
                mult *= 2.0
        X_d[i] = X[i] * mult
        if (hour >= 20 or hour < 3) and np.random.rand() < flip_prob:
            y_d[i] = 1 - y_d[i]
    return X_d, y_d

def apply_compound(X, y, start_season=50000, end_season=140000, start_ramadan=100000, end_ramadan=115000, flip_prob=0.7):
    X_d, y_d = apply_riyadh_season(X, y, start_idx=start_season, flip_prob=0.3)
    n = len(X)
    hours = (np.arange(n) // 500) % 24
    for i in range(n):
        if start_ramadan <= i < end_ramadan:
            hour = hours[i]
            mult = 1.0
            if 20 <= hour or hour < 3:
                mult = 1.5
            elif 6 <= hour < 18:
                mult = 0.7
            if hour in [4,11,15,18,22]:
                minute = (i % (60*24)) % 60
                if minute < 30:
                    mult *= 2.0
            X_d[i] = X[i] * mult
            if (hour >= 20 or hour < 3) and np.random.rand() < flip_prob:
                y_d[i] = 1 - y_d[i]
    return X_d, y_d

# ============================================================================
# Streaming Evaluation Function
# ============================================================================
def run_streaming(X_stream, y_stream, base_model, scenario_name):
    model = clone_model(base_model)
    model.build((None, X_stream.shape[1]))
    model.set_weights(base_model.get_weights())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    n_batches = len(X_stream) // BATCH_SIZE
    accuracies = []
    all_preds, all_true = [], []
    
    for i in tqdm(range(n_batches), desc=f"Processing {scenario_name}"):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        Xb = X_stream[start:end]
        yb = y_stream[start:end]
        
        y_prob = model.predict(Xb, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        all_preds.extend(y_pred)
        all_true.extend(yb)
        acc = accuracy_score(yb, y_pred)
        accuracies.append(acc)
        
        # Online fine-tune
        model.fit(Xb, yb, epochs=ONLINE_EPOCHS, batch_size=64, 
                  class_weight=class_weights, verbose=0)
    
    final_acc = accuracy_score(all_true, all_preds)
    final_prec = precision_score(all_true, all_preds, zero_division=0)
    final_rec = recall_score(all_true, all_preds)
    final_f1 = f1_score(all_true, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_true, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp+tn) > 0 else 0.0
    
    return {
        'scenario': scenario_name,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'final_accuracy': final_acc,
        'precision': final_prec,
        'recall': final_rec,
        'f1': final_f1,
        'fpr': fpr
    }

# ============================================================================
# Run ALL 5 Experiments
# ============================================================================
results = []

scenarios_list = [
    ("E1_Baseline", lambda X,y: (X,y)),
    ("E2_RiyadhSeason", lambda X,y: apply_riyadh_season(X, y, start_idx=50000)),
    ("E3_Infrastructure", lambda X,y: apply_infrastructure_growth(X, y)),
    ("E4_Ramadan", lambda X,y: apply_ramadan(X, y, first_start=100000)),
    ("E5_Compound", lambda X,y: apply_compound(X, y))
]

for name, transform in scenarios_list:
    print("\n" + "="*60)
    print(f"Running {name}")
    print("="*60)
    X_drift, y_drift = transform(X_stream_scaled.copy(), y_stream.copy())
    X_drift = np.clip(X_drift, -10, 10)
    res = run_streaming(X_drift, y_drift, base_model, name)
    results.append(res)
    print(f"Mean Accuracy: {res['mean_accuracy']:.4f} ({res['mean_accuracy']*100:.2f}%)")

# ============================================================================
# Summary
# ============================================================================
summary_df = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL SUMMARY – ALL 5 SCENARIOS (Full Dataset)")
print("="*60)
print(summary_df.to_string())
print("="*60)

# Save
summary_df.to_csv("D:/ConceptDrift/results/full_dataset_all_5_scenarios.csv", index=False)
print("\nResults saved to D:/ConceptDrift/results/full_dataset_all_5_scenarios.csv")
