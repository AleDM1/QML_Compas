import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import time

print("🧪 TEST COMPARATIVO DI DUE IMPLEMENTAZIONI KERNEL QUANTISTICO")
print("="*70)

# Test con 4 qubits (come nel primo esempio)
nqubits = 4
print(f"📊 Configurazione: {nqubits} qubits (spazio di Hilbert: 2^{nqubits} = {2**nqubits})")

# Setup dispositivi
dev1 = qml.device("lightning.qubit", wires=nqubits)
dev2 = qml.device("lightning.qubit", wires=nqubits)

# ========================
# IMPLEMENTAZIONE 1: Con probs e adjoint (la tua prima versione)
# ========================

@qml.qnode(dev1)
def kernel_circ(a, b):
    qml.AmplitudeEmbedding(a, wires=range(nqubits), pad_with=0, normalize=True)
    qml.adjoint(qml.AmplitudeEmbedding(b, wires=range(nqubits), pad_with=0, normalize=True))
    return qml.probs(wires=range(nqubits))

def qkernel_v1(A, B):
    """Versione 1: Con probs()[0]"""
    return np.array([[kernel_circ(a, b)[0] for b in B] for a in A])

# ========================
# IMPLEMENTAZIONE 2: Con state e prodotto scalare (la tua seconda versione)
# ========================

@qml.qnode(dev2)
def get_state(a):
    qml.AmplitudeEmbedding(a, wires=range(nqubits), pad_with=0, normalize=True)
    return qml.state()

def compute_states(X):
    return np.stack([get_state(x) for x in X])

def qkernel_v2(A, B):
    """Versione 2: Con state e prodotto scalare"""
    states_A = compute_states(A)
    states_B = compute_states(B)
    K = np.abs(states_A @ states_B.conj().T) ** 2
    return K

# ========================
# GENERAZIONE DATI DI TEST
# ========================

print(f"\n📋 Generazione dati di test...")
np.random.seed(42)

# Crea dati con dimensione adatta ai 4 qubits
n_samples = 4
n_features = 8  # Meno di 2^4=16 per testare il padding

# Dati casuali normalizzati
test_data = np.random.randn(n_samples, n_features) * 0.5

print(f"   Dati: {test_data.shape} (campioni: {n_samples}, features: {n_features})")
print(f"   Primi 2 campioni:")
for i in range(min(2, n_samples)):
    print(f"     x_{i}: {test_data[i]}")

# ========================
# TEST 1: Confronto su piccolo subset
# ========================

print(f"\n🔍 TEST 1: Confronto kernel su subset (2x2)")
subset = test_data[:2]

print("Calcolo kernel v1 (probs)...")
start_time = time.time()
K1 = qkernel_v1(subset, subset)
time1 = time.time() - start_time

print("Calcolo kernel v2 (state)...")
start_time = time.time()
K2 = qkernel_v2(subset, subset)
time2 = time.time() - start_time

print(f"\n⏱️  Tempi di esecuzione:")
print(f"   Kernel v1 (probs): {time1:.4f}s")
print(f"   Kernel v2 (state): {time2:.4f}s")
print(f"   Speedup: {time1/time2:.2f}x")

print(f"\n📊 Matrici kernel risultanti:")
print(f"Kernel v1 (probs):")
print(K1)
print(f"\nKernel v2 (state):")
print(K2)

print(f"\n📐 Confronto numerico:")
diff_matrix = np.abs(K1 - K2)
max_diff = np.max(diff_matrix)
mean_diff = np.mean(diff_matrix)

print(f"   Differenza massima: {max_diff:.8f}")
print(f"   Differenza media: {mean_diff:.8f}")
print(f"   Matrici identiche: {np.allclose(K1, K2, atol=1e-10)}")

if not np.allclose(K1, K2, atol=1e-6):
    print(f"   ⚠️  DIFFERENZE SIGNIFICATIVE rilevate!")
    print(f"   Matrice delle differenze:")
    print(diff_matrix)
else:
    print(f"   ✅ Matrici praticamente identiche!")

# ========================
# TEST 2: Proprietà dei kernel
# ========================

print(f"\n🔍 TEST 2: Verifica proprietà dei kernel")

# Simmetria
print("Verifica simmetria K(i,j) = K(j,i):")
for i in range(2):
    for j in range(2):
        if i != j:
            sym_diff_v1 = abs(K1[i,j] - K1[j,i])
            sym_diff_v2 = abs(K2[i,j] - K2[j,i])
            print(f"   K1({i},{j}) - K1({j},{i}) = {sym_diff_v1:.8f}")
            print(f"   K2({i},{j}) - K2({j},{i}) = {sym_diff_v2:.8f}")

# Auto-kernel (dovrebbe essere 1.0)
print(f"\nVerifica auto-kernel K(i,i) (dovrebbe essere ~1.0):")
for i in range(2):
    print(f"   K1({i},{i}) = {K1[i,i]:.6f}")
    print(f"   K2({i},{i}) = {K2[i,i]:.6f}")

# Range dei valori
print(f"\nRange valori kernel:")
print(f"   K1: [{K1.min():.6f}, {K1.max():.6f}]")
print(f"   K2: [{K2.min():.6f}, {K2.max():.6f}]")

# ========================
# TEST 3: Test su SVM con dataset più grande
# ========================

print(f"\n🧪 TEST 3: Confronto accuracy SVM su dataset sintetico")

# Genera dataset circles piccolo per test veloce
X, y = make_circles(n_samples=50, noise=0.1, factor=0.6, random_state=42)

# Adatta le dimensioni per 4 qubits (2^4 = 16 features)
target_features = 12  # Meno di 16 per testare padding
if X.shape[1] < target_features:
    padding = np.random.randn(X.shape[0], target_features - X.shape[1]) * 0.1
    X = np.hstack([X, padding])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dataset: train {X_train.shape}, test {X_test.shape}")

# Test SVM con entrambi i kernel (su subset per velocità)
subset_size = 15
X_train_subset = X_train[:subset_size]
y_train_subset = y_train[:subset_size]
X_test_subset = X_test[:10]
y_test_subset = y_test[:10]

print(f"\nTesting SVM con kernel v1 (probs)...")
start_time = time.time()
svm1 = SVC(kernel=qkernel_v1)
svm1.fit(X_train_subset, y_train_subset)
y_pred1 = svm1.predict(X_test_subset)
acc1 = accuracy_score(y_test_subset, y_pred1)
time_svm1 = time.time() - start_time

print(f"Testing SVM con kernel v2 (state)...")
start_time = time.time()
svm2 = SVC(kernel=qkernel_v2)
svm2.fit(X_train_subset, y_train_subset)
y_pred2 = svm2.predict(X_test_subset)
acc2 = accuracy_score(y_test_subset, y_pred2)
time_svm2 = time.time() - start_time

print(f"\n📊 RISULTATI FINALI:")
print(f"{'Metodo':<20} {'Accuracy':<10} {'Tempo':<10} {'Predizioni identiche'}")
print("-" * 55)
print(f"{'Kernel v1 (probs)':<20} {acc1:<10.4f} {time_svm1:<10.2f}s")
print(f"{'Kernel v2 (state)':<20} {acc2:<10.4f} {time_svm2:<10.2f}s")

pred_identical = np.array_equal(y_pred1, y_pred2)
print(f"\nPredizioni identiche: {pred_identical}")
if not pred_identical:
    print("Differenze nelle predizioni:")
    for i, (p1, p2) in enumerate(zip(y_pred1, y_pred2)):
        if p1 != p2:
            print(f"   Campione {i}: v1={p1}, v2={p2}")

print(f"\n" + "="*70)
print("CONCLUSIONI:")
if np.allclose(K1, K2, atol=1e-6):
    print("✅ I due metodi sono matematicamente EQUIVALENTI")
    print("   - Producono le stesse matrici kernel")
    if pred_identical:
        print("   - Producono le stesse predizioni SVM")
else:
    print("⚠️  I due metodi sono DIVERSI")
    print("   - Producono matrici kernel diverse")
    
print(f"🚀 Performance: kernel v2 è {time1/time2:.1f}x più veloce di v1")
print("="*70)