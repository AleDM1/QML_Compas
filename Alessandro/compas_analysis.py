import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.sklearn.metrics import statistical_parity_difference, equal_opportunity_difference
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load the COMPAS dataset
dataset = CompasDataset()

# Convert to pandas DataFrame
df = dataset.convert_to_dataframe()[0]

# First, let's inspect what columns we actually have
print("Columns in the dataset:")
print(df.columns.tolist())
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Store original columns for comparison
original_columns = df.columns.tolist()
print(f"\nTotal original features: {len(original_columns)}")

# Define columns to potentially remove (only if they exist)
columns_to_remove = ['id', 'name', 'first', 'last', 'compas_screening_date', 
                    'dob', 'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date']

# Remove only existing columns
existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
non_existing_columns = [col for col in columns_to_remove if col not in df.columns]

print(f"\n📋 FEATURE REMOVAL SUMMARY:")
print(f"Features actually removed: {existing_columns_to_remove}")
print(f"Features not found in dataset: {non_existing_columns}")
print(f"Total features removed: {len(existing_columns_to_remove)}")

if existing_columns_to_remove:
    df = df.drop(columns=existing_columns_to_remove)

# Show remaining columns after removal
remaining_columns = df.columns.tolist()
print(f"\nRemaining features after removal: {len(remaining_columns)}")
print(f"Remaining features: {remaining_columns}")

# Remove rows with missing values
print(f"\nRows before dropna: {len(df)}")
df = df.dropna()
print(f"Rows after dropna: {len(df)}")

# Encode categorical variables
categorical_columns = df.select_dtypes(include='object').columns
print(f"\nCategorical columns to encode: {categorical_columns.tolist()}")

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
# Note: The target variable in COMPAS dataset might be named differently
target_column = 'two_year_recid' if 'two_year_recid' in df.columns else 'Probability'
print(f"\nUsing target column: {target_column}")

X = df.drop(columns=[target_column])
y = df[target_column]

# Final summary
print(f"\n🎯 FINAL DATASET SUMMARY:")
print(f"Final features for training: {len(X.columns)}")
print(f"Feature names: {X.columns.tolist()}")
print(f"Target variable: {target_column}")
print(f"Dataset shape: X={X.shape}, y={y.shape}")



# Analisi dello sbilanciamento
class_counts = y.value_counts()
total = len(y)

print("\n=== Analisi dello sbilanciamento della variabile target ===")
for label, count in class_counts.items():
    percentage = 100 * count / total
    print(f"Classe {int(label)}: {count} casi ({percentage:.2f}%)")

# Differenza tra le due classi
imbalance_ratio = abs(class_counts[0] - class_counts[1]) / total
print(f"Imbalance Ratio (assoluto): {imbalance_ratio:.4f} → {(imbalance_ratio * 100):.2f}% di sbilanciamento")

# Visualizzazione
plt.figure(figsize=(6, 4))
plt.bar(class_counts.index.astype(str), class_counts.values, color=['steelblue', 'tomato'])
plt.title("Distribuzione della variabile target: two_year_recid")
plt.xlabel("Classe")
plt.ylabel("Numero di istanze")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Analisi congiunta tra target e attributo protetto
df_fair = df[[target_column, 'race']].copy()

# Mappa valori numerici a etichette leggibili (se appropriato)
race_labels = {0.0: 'Gruppo A', 1.0: 'Gruppo B'}
df_fair['race_label'] = df_fair['race'].map(race_labels)
df_fair[target_column] = df_fair[target_column].astype(int)

# Tabella di contingenza
contingency = pd.crosstab(df_fair['race_label'], df_fair[target_column], normalize='index') * 100

print("\n=== Distribuzione congiunta (percentuale) ===")
print(contingency)

# Visualizzazione
plt.figure(figsize=(6, 4))
sns.barplot(data=df_fair, x='race_label', y=target_column, estimator=lambda x: sum(x)/len(x)*100)
plt.title("Percentuale di recidiva (two_year_recid) per gruppo razziale")
plt.ylabel("Recidivi (%)")
plt.xlabel("Gruppo razziale")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === Salvataggio del dataset preprocessato in CSV ===


# Percorso di salvataggio
directory_path = r"D:\Visualstudio_Code_Projects\QML_Compas\datasets"
filename = "compas_from_aif360.csv"
full_path = os.path.join(directory_path, filename)

# Creazione directory se non esiste
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"[INFO] Directory creata: {directory_path}")
else:
    print(f"[INFO] Directory già esistente: {directory_path}")

# Salvataggio condizionato del file
if os.path.exists(full_path):
    print(f"[INFO] File già esistente, nessuna azione necessaria: {full_path}")
else:
    print(f"[INFO] File non trovato, procedo con il salvataggio del dataset COMPAS preprocessato.")
    df.to_csv(full_path, index=False)
    print(f"[INFO] Dataset salvato in formato CSV: {full_path}")


# Define protected attribute (race)
# Check available options for protected attributes
if 'race' in df.columns:
    protected_attr = df['race']
    print("Using 'race' as protected attribute")
elif 'Race' in df.columns:
    protected_attr = df['Race'] 
    print("Using 'Race' as protected attribute")
else:
    print("Available columns for protected attribute:")
    print([col for col in df.columns if 'race' in col.lower()])
    # Use the first race-related column found
    race_cols = [col for col in df.columns if 'race' in col.lower()]
    if race_cols:
        protected_attr = df[race_cols[0]]
        print(f"Using '{race_cols[0]}' as protected attribute")
    else:
        # Fallback - use first categorical column
        protected_attr = df[categorical_columns[0]] if len(categorical_columns) > 0 else df.iloc[:, 0]
        print(f"Fallback: using first available column as protected attribute")

print(f"Protected attribute unique values: {protected_attr.unique()}")

# Split the data
#X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
#    X, y, protected_attr, test_size=0.3, random_state=42
#)

X_train_full, X_test, y_train_full, y_test, prot_train_full, prot_test = train_test_split(
    X, y, protected_attr, test_size=0.3, random_state=42)

# Split further into training and validation (e.g., 80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
_, prot_val = train_test_split(prot_train_full, test_size=0.2, random_state=42)
print(f"\nTraining set shape: {X_train.shape}, Validation set shape: {X_val.shape}, Test set shape: {X_test.shape}")

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class AlphaReLU(nn.Module):
    def __init__(self, alpha=1.0):
        super(AlphaReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, self.alpha * x, torch.zeros_like(x))

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc2(x))
    

class ComplexNN_1(nn.Module):
    def __init__(self, input_dim):
        super(ComplexNN_1, self).__init__()
        self.hidden_dim = 16
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.alpha_relu = AlphaReLU(alpha=1.0)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim) 
        self.fc3 = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.alpha_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.alpha_relu(self.fc2(x))
        #x = self.dropout(x)
        #x = self.alpha_relu(self.fc2(x))
        #x = self.dropout(x)
        #x = self.alpha_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.alpha_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.alpha_relu(self.fc2(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc3(x))

#model = SimpleNN(X_train_scaled.shape[1])
model = ComplexNN_1(X_train_scaled.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"\nTraining model with {X_train_scaled.shape[1]} features...")

# Before training 
X_val_scaled = scaler.transform(X_val)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

best_loss = float('inf')
patience = 20  # epoche di tolleranza
counter = 0


# Train the model
model.train()
if 1 == 2:
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

        # Predict
        model.eval()
        with torch.no_grad():
            y_pred_proba = model(X_test_tensor).numpy()
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

else:
    print("Training with early stopping and validation...")

    # Validation
    for epoch in range(1000):  # numero massimo di epoche
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        if val_loss < best_loss - 1e-4:  # miglioramento minimo
            best_loss = val_loss
            counter = 0
            best_model_state = model.state_dict()  # salvataggio del miglior modello
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}, best val_loss = {best_loss:.4f}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_model_state)

    # Predict
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).numpy()
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()




# Classic metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\n=== Model Performance ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

# Fairness metrics
try:
    spd = statistical_parity_difference(y_test, y_pred, prot_attr=prot_test)
    eod = equal_opportunity_difference(y_test, y_pred, prot_attr=prot_test)
    
    print(f"\n=== Fairness Metrics ===")
    print(f"Statistical Parity Difference: {spd:.4f}")
    print(f"Equal Opportunity Difference: {eod:.4f}")
    
    # Composite metric: Harmonic mean of F1 and (1 - max fairness deviation)
    max_unfairness = max(abs(spd), abs(eod))
    fairness_component = 1 - max_unfairness
    
    if f1 + fairness_component > 0:
        composite_metric = 2 * f1 * fairness_component / (f1 + fairness_component)
    else:
        composite_metric = 0
    
    print(f"Composite Metric (F1 & Fairness): {composite_metric:.4f}")
    
    # Prepare results
    results = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Statistical Parity Difference': spd,
        'Equal Opportunity Difference': eod,
        'Composite Metric': composite_metric
    }
    
    print(f"\n=== Summary ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        
except Exception as e:
    print(f"Error calculating fairness metrics: {e}")
    print("This might be due to the protected attribute encoding or dataset structure.")
    
    # Basic results without fairness metrics
    results = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
    }
    
    print(f"\n=== Basic Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")