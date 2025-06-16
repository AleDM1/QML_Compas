import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm

# Configurazione del dispositivo quantistico
nqubits = 9
dev = qml.device("lightning.gpu", wires=nqubits)

@qml.qnode(dev, interface="autograd")
def get_state(a):
    """Restituisce lo stato dopo AmplitudeEmbedding di a."""
    qml.AmplitudeEmbedding(a, wires=range(nqubits), pad_with=0, normalize=True)
    return qml.state()

def compute_states(X):
    """Batch: per ogni vettore in X, calcola lo statevector."""
    return np.stack([get_state(x) for x in X])

def qkernel(A, B):
    """Calcola la matrice kernel per batch di statevectors."""
    states_A = compute_states(A)
    states_B = compute_states(B)
    # Prodotto scalare a matrice: (A ⋅ B†), poi modulo quadro
    K = np.abs(states_A @ states_B.conj().T) ** 2
    return K


'''
# Crea dataset circles
print("Creazione dataset...")
X, y = make_circles(n_samples=100, noise=0.1, factor=0.6, random_state=42)

# Espandi a 401 dimensioni per utilizzare i 9 qubits (2^9 = 512 amplitudini)
n_features = 401
padding = np.random.randn(X.shape[0], n_features - X.shape[1]) * 0.1
X = np.hstack([X, padding])

print(f"Dimensioni dataset: {X.shape}")
print(f"Classi: {np.unique(y)}")
print(f"Numero di qubits: {nqubits}")
print(f"Spazio di Hilbert: 2^{nqubits} = {2**nqubits} dimensioni")
print(f"Features utilizzate: {n_features} su {2**nqubits} disponibili ({n_features/(2**nqubits)*100:.1f}%)")

# Normalizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
x_tr, x_test, y_tr, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print(f"\nDimensioni training set: {x_tr.shape}")
print(f"Dimensioni test set: {x_test.shape}")

# Confronto con kernel classico RBF
print(f"\n{'='*50}")
print("BASELINE - KERNEL RBF CLASSICO")
print(f"{'='*50}")

start_time = time.time()
svm_classical = SVC(kernel='rbf', gamma='scale')
svm_classical.fit(x_tr, y_tr)
y_pred_classical = svm_classical.predict(x_test)
classical_time = time.time() - start_time
accuracy_classical = accuracy_score(y_test, y_pred_classical)

print(f"Accuracy RBF classico: {accuracy_classical:.3f}")
print(f"Tempo totale: {classical_time:.2f} secondi")

# Training del SVM quantistico
print(f"\n{'='*50}")
print("TRAINING SVM - LIGHTNING.QUBIT")
print(f"{'='*50}")

start_time = time.time()

svm_quantum = SVC(kernel=qkernel)
svm_quantum.fit(x_tr, y_tr)

training_time = time.time() - start_time

# Predizioni
print("Calcolo predizioni...")
pred_start = time.time()
y_pred_quantum = svm_quantum.predict(x_test)
prediction_time = time.time() - pred_start
total_time = training_time + prediction_time

# Risultati
accuracy_quantum = accuracy_score(y_test, y_pred_quantum)

print(f"\n=== Risultati Quantum Kernel ===")
print(f"Accuracy: {accuracy_quantum:.3f}")
print(f"Tempo training: {training_time:.2f} secondi")
print(f"Tempo prediction: {prediction_time:.2f} secondi")
print(f"Tempo totale: {total_time:.2f} secondi")

# Confronto finale
print(f"\n{'='*60}")
print("CONFRONTO FINALE DELLE PERFORMANCE")
print(f"{'='*60}")

print(f"{'Metodo':<25} {'Accuracy':<10} {'Tempo (s)':<12} {'Speedup':<10}")
print("-" * 65)

# Baseline
print(f"{'Classical RBF':<25} {accuracy_classical:<10.3f} {classical_time:<12.2f} {'1.00x':<10}")

# Quantum
speedup = classical_time / total_time if total_time > 0 else 0
print(f"{'Quantum Lightning CPU':<25} {accuracy_quantum:<10.3f} {total_time:<12.2f} {speedup:<10.2f}x")

# Analisi dettagliata
print(f"\n{'='*50}")
print("ANALISI DETTAGLIATA")
print(f"{'='*50}")

diff = accuracy_quantum - accuracy_classical
print(f"Differenza accuracy: {diff:+.3f}")
print(f"Quantum vs Classical speedup: {speedup:.2f}x")

if speedup < 1:
    slowdown = total_time / classical_time
    print(f"⚠️  Quantum è {slowdown:.1f}x più lento del classico")
else:
    print(f"🚀 Quantum è {speedup:.1f}x più veloce del classico")

print(f"\nBreakdown tempi quantum:")
print(f"  Training: {training_time:.2f}s ({training_time/total_time*100:.1f}%)")
print(f"  Prediction: {prediction_time:.2f}s ({prediction_time/total_time*100:.1f}%)")

# Statistiche computazionali
n_train = len(x_tr)
n_test = len(x_test)
kernel_computations_train = n_train * n_train
kernel_computations_test = n_test * n_train
total_quantum_ops = (kernel_computations_train + kernel_computations_test) * (2**nqubits)

print(f"\n📊 Complessità computazionale:")
print(f"   Kernel computations training: {kernel_computations_train:,}")
print(f"   Kernel computations prediction: {kernel_computations_test:,}")
print(f"   Operazioni quantistiche totali: {total_quantum_ops:,}")
print(f"   Operazioni per secondo: {total_quantum_ops/total_time:,.0f}")

# Test del kernel su alcuni esempi
print(f"\n=== Test Kernel Values ===")
print("Valori del kernel per alcuni esempi:")
for i in range(min(3, len(x_test))):
    for j in range(min(3, len(x_test))):
        kernel_val = kernel_circ(x_test[i], x_test[j])[0]
        print(f"K(x_{i}, x_{j}) = {kernel_val:.4f}")

# Test proprietà del kernel
print(f"\n🔍 Proprietà del kernel quantistico:")

# Simmetria
k_01 = kernel_circ(x_test[0], x_test[1])[0]
k_10 = kernel_circ(x_test[1], x_test[0])[0]
print(f"   Simmetria K(0,1)={k_01:.6f}, K(1,0)={k_10:.6f}, diff={abs(k_01-k_10):.8f}")

# Auto-kernel
k_00 = kernel_circ(x_test[0], x_test[0])[0]
print(f"   Auto-kernel K(0,0)={k_00:.6f} (dovrebbe essere ~1.0)")

# Range valori
sample_kernels = [kernel_circ(x_test[i], x_test[j])[0] 
                 for i in range(min(5, len(x_test))) 
                 for j in range(min(5, len(x_test)))]
print(f"   Range valori: [{min(sample_kernels):.6f}, {max(sample_kernels):.6f}]")

print(f"\n{'='*60}")
print("COMPLETATO! 🎉")
print(f"{'='*60}")
'''