"""
Quantum Neural Network con Amplitude Embedding e Strongly Entangling Layers
Implementazione semplificata usando PennyLane + PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

# =============================================
# CONFIGURAZIONE DISPOSITIVO QUANTISTICO
# =============================================

N_QUBITS = 9  # 2^9 = 512 features max
DEVICE = qml.device("default.qubit", wires=N_QUBITS)

# =============================================
# CIRCUITO QUANTISTICO
# =============================================

@qml.qnode(DEVICE, interface="torch")
def quantum_circuit(inputs, weights):
    """
    Circuito quantistico con Amplitude Embedding + 3 Strongly Entangling Layers
    """
    # 1. AMPLITUDE EMBEDDING
    qml.AmplitudeEmbedding(
        features=inputs, 
        wires=range(N_QUBITS), 
        normalize=True,
        pad_with=0
    )
    
    # 2. STRONGLY ENTANGLING LAYERS (3 layers)
    qml.StronglyEntanglingLayers(
        weights=weights, 
        wires=range(N_QUBITS)
    )
    
    # 3. MEASUREMENT
    return qml.expval(qml.PauliZ(0))

# =============================================
# QUANTUM NEURAL NETWORK CLASS
# =============================================

class QuantumNeuralNetwork(nn.Module, BaseEstimator, ClassifierMixin):
    """
    Quantum Neural Network compatibile con scikit-learn
    """
    
    def __init__(self, n_qubits=N_QUBITS, n_layers=3):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.max_features = 2 ** n_qubits
        
        # Parametri trainabili del circuito quantistico
        weight_shapes = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.weights = nn.Parameter(
            torch.randn(weight_shapes, requires_grad=True) * 0.1
        )
        
        # Per compatibilità scikit-learn
        self.classes_ = np.array([0, 1])
        
    def forward(self, x):
        """
        Forward pass attraverso il circuito quantistico
        """
        # Converte in torch tensor se necessario
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Gestisce input singolo o batch
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        outputs = []
        for sample in x:
            # Prepara input per amplitude embedding
            sample_prepared = self._prepare_input(sample)
            
            # Esegui circuito quantistico
            output = quantum_circuit(sample_prepared, self.weights)
            outputs.append(output)
        
        return torch.stack(outputs)
    
    def _prepare_input(self, x):
        """
        Prepara input per amplitude embedding
        """
        # Converte in numpy
        if isinstance(x, torch.Tensor):
            x_np = x.detach().numpy()
        else:
            x_np = np.array(x)
        
        # Padding o troncamento per 2^n_qubits features
        if len(x_np) > self.max_features:
            x_prepared = x_np[:self.max_features]
        elif len(x_np) < self.max_features:
            x_prepared = np.pad(x_np, (0, self.max_features - len(x_np)), mode='constant')
        else:
            x_prepared = x_np
        
        # Normalizzazione per amplitude embedding
        norm = np.linalg.norm(x_prepared)
        if norm > 0:
            x_prepared = x_prepared / norm
        
        return torch.tensor(x_prepared, dtype=torch.float32)
    
    def predict_proba(self, X):
        """
        Predice probabilità (compatibilità scikit-learn)
        """
        self.eval()
        with torch.no_grad():
            raw_outputs = self.forward(X)
            
            # Converte output quantistico (-1, 1) in probabilità (0, 1)
            proba_class_1 = (raw_outputs + 1) / 2
            proba_class_0 = 1 - proba_class_1
            
            probabilities = torch.stack([proba_class_0, proba_class_1], dim=1)
            return probabilities.numpy()
    
    def predict(self, X):
        """
        Predice classi (compatibilità scikit-learn)
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def score(self, X, y):
        """
        Calcola accuracy (compatibilità scikit-learn)
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def fit(self, X, y):
        """
        Placeholder per compatibilità scikit-learn
        (Il training vero va implementato separatamente)
        """
        # Per ora solo salva le classi
        self.classes_ = np.unique(y)
        return self

# =============================================
# FACTORY FUNCTION
# =============================================

def create_quantum_nn(n_qubits=9, n_layers=3):
    """
    Crea una Quantum Neural Network
    """
    qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
    
    print(f"⚛️  Quantum Neural Network creata:")
    print(f"   🔢 Qubits: {n_qubits}")
    print(f"   🧠 Strongly Entangling Layers: {n_layers}")
    print(f"   📊 Max features: {2**n_qubits}")
    print(f"   🎛️  Parametri trainabili: {sum(p.numel() for p in qnn.parameters())}")
    
    return qnn