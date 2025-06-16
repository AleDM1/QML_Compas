"""
Modulo semplificato per interpretabilità Quantum vs Classical SVM
Solo Permutation Importance con barre di progresso
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import shap
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PERMUTATION IMPORTANCE
# =============================================

def compute_permutation_importance(model, X_test, y_test, feature_names=None, n_repeats=5, max_samples=50):
    """
    Calcola permutation importance con subset ridotto - VERSIONE CORRETTA per DataFrame
    
    Come funziona Permutation Importance:
    1. Calcola accuracy baseline del modello
    2. Per ogni feature:
       - Permuta (mescola) casualmente i valori di quella feature
       - Ricalcola l'accuracy con feature permutata
       - Importance = accuracy_baseline - accuracy_permutata
    3. Se feature importante → permutarla fa calare molto l'accuracy
    4. Se feature irrilevante → permutarla non cambia l'accuracy
    """
    # Usa subset per ridurre chiamate al quantum kernel
    n_samples = min(max_samples, len(X_test))
    subset_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # CORREZIONE: Usa .iloc per DataFrame pandas o .values per numpy
    if hasattr(X_test, 'iloc'):  # È un DataFrame pandas
        X_subset = X_test.iloc[subset_indices]
        y_subset = y_test.iloc[subset_indices]
    else:  # È un array numpy
        X_subset = X_test[subset_indices]
        y_subset = y_test[subset_indices]
    
    if feature_names is None:
        if hasattr(X_test, 'columns'):  # DataFrame pandas
            feature_names = X_test.columns.tolist()
        else:  # Array numpy
            feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    print(f"   Calcolo permutation importance su {n_samples} campioni...")
    perm_importance = permutation_importance(
        model, X_subset, y_subset, n_repeats=n_repeats, random_state=42, scoring='accuracy'
    )
    
    results_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return results_df


def compute_shapley_values(model, X_test, n_samples=20):

    X_subset = X_test[:n_samples]
    background = shap.sample(X_test, 50)
    
    # Explainer
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # SHAP values

    result = explainer.shap_values(X_subset)
    
    # Se binario, prendi classe positiva
    if isinstance(result, list):
        result = result[1]
    
    return result


# =============================================
# VISUALIZZAZIONE
# =============================================

def plot_permutation_importance_comparison(quantum_results, classical_results, figsize=(14, 8), top_n=10):
    """
    Plot semplificato del confronto permutation importance
    """
    print("📊 Generazione visualizzazioni...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prendi top N feature per ogni modello (già ordinate)
    q_top = quantum_results['permutation'].head(top_n)
    c_top = classical_results['permutation'].head(top_n)
    
    # 1. Quantum Model
    ax1.barh(range(len(q_top)), q_top['importance_mean'], 
             color='purple', alpha=0.8, edgecolor='darkmagenta')
    ax1.set_yticks(range(len(q_top)))
    ax1.set_yticklabels(q_top['feature'], fontsize=10)
    ax1.set_title('⚛️ Quantum Model', fontsize=14, fontweight='bold', color='purple')
    ax1.set_xlabel('Feature Importance', fontsize=12)
    ax1.invert_yaxis()  # Feature più importante in alto
    
    # 2. Classical Model  
    ax2.barh(range(len(c_top)), c_top['importance_mean'], 
             color='blue', alpha=0.8, edgecolor='darkblue')
    ax2.set_yticks(range(len(c_top)))
    ax2.set_yticklabels(c_top['feature'], fontsize=10)
    ax2.set_title('🔷 Classical Model', fontsize=14, fontweight='bold', color='blue')
    ax2.set_xlabel('Feature Importance', fontsize=12)
    ax2.invert_yaxis()  # Feature più importante in alto
    
    # Titolo principale
    plt.suptitle(f'🔍 Top {top_n} Feature più Importanti', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.show()
    
    return fig



def plot_shapley_values_comparison(shap_q, shap_c, feature_names, figsize=(14, 8), top_n=10):
    """
    Plot diretto degli SHAP values senza conversioni - VERSIONE PULITA
    
    Args:
        shap_q: array numpy (n_samples, n_features) - SHAP quantum
        shap_c: array numpy (n_samples, n_features) - SHAP classical  
        feature_names: lista nomi features
        top_n: numero di top features da mostrare
    """
    
    # Chiudi eventuali figure precedenti per evitare duplicazioni
    plt.close('all')
    
    # Calcola importanza media assoluta per feature
    if len(shap_q.shape) == 3 and shap_q.shape[2] == 2:
        # Classificazione binaria: prendi classe positiva (indice 1)
        quantum_importance = np.abs(shap_q[:, :, 1]).mean(axis=0)
        classical_importance = np.abs(shap_c[:, :, 1]).mean(axis=0)
    else:
        # Caso generale
        quantum_importance = np.abs(shap_q).mean(axis=0)
        classical_importance = np.abs(shap_c).mean(axis=0)
    
    # Ordina features per importanza
    quantum_order = np.argsort(quantum_importance)[::-1][:top_n]
    classical_order = np.argsort(classical_importance)[::-1][:top_n]
    
    # Crea UNA SOLA figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Quantum SHAP
    q_features = [feature_names[i] for i in quantum_order]
    q_values = quantum_importance[quantum_order]
    
    ax1.barh(range(len(q_features)), q_values, 
             color='purple', alpha=0.8, edgecolor='darkmagenta')
    ax1.set_yticks(range(len(q_features)))
    ax1.set_yticklabels(q_features, fontsize=10)
    ax1.set_title('⚛️ Quantum Model (SHAP)', fontsize=13, fontweight='bold', color='purple', pad=15)
    ax1.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax1.invert_yaxis()
    
    # 2. Classical SHAP
    c_features = [feature_names[i] for i in classical_order]
    c_values = classical_importance[classical_order]
    
    ax2.barh(range(len(c_features)), c_values, 
             color='blue', alpha=0.8, edgecolor='darkblue')
    ax2.set_yticks(range(len(c_features)))
    ax2.set_yticklabels(c_features, fontsize=10)
    ax2.set_title('🔷 Classical Model (SHAP)', fontsize=13, fontweight='bold', color='blue', pad=15)
    ax2.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax2.invert_yaxis()
    
    # Titolo principale con spacing corretto
    plt.suptitle(f'🧠 Top {top_n} Feature - SHAP Values', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Layout ottimizzato per evitare sovrapposizioni
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Mostra SOLO UNA VOLTA e non restituire la figura
    plt.show()
    plt.close(fig)  # Chiudi la figura dopo averla mostrata


# =============================================
# SUMMARY
# =============================================

def print_summary(quantum_results, classical_results):
    """
    Summary del confronto permutation importance
    """
    print("\n" + "="*60)
    print("📋 SUMMARY PERMUTATION IMPORTANCE")
    print("="*60)
    
    q_perm = quantum_results['permutation']
    c_perm = classical_results['permutation']
    
    # Top features per modello
    print(f"\n🏆 TOP 5 FEATURES:")
    print("-" * 30)
    
    print(f"⚛️  QUANTUM MODEL:")
    for i, row in q_perm.head(5).iterrows():
        print(f"   {i+1}. {row['feature']:25s}: {row['importance_mean']:+.6f}")
    
    print(f"\n🔷 CLASSICAL MODEL:")
    for i, row in c_perm.head(5).iterrows():
        print(f"   {i+1}. {row['feature']:25s}: {row['importance_mean']:+.6f}")
    
    # Consenso
    q_top5 = set(q_perm.head(5)['feature'])
    c_top5 = set(c_perm.head(5)['feature'])
    consensus = q_top5 & c_top5
    
    print(f"\n🤝 CONSENSO ANALYSIS:")
    print("-" * 20)
    print(f"   Features comuni nei top 5: {len(consensus)}/5")
    
    if consensus:
        print(f"   ✅ Accordo su:")
        for feature in consensus:
            q_val = q_perm[q_perm['feature'] == feature]['importance_mean'].iloc[0]
            c_val = c_perm[c_perm['feature'] == feature]['importance_mean'].iloc[0]
            print(f"      • {feature}: Q={q_val:.4f}, C={c_val:.4f}")
    
    # Features specifiche
    q_unique = q_top5 - c_top5
    c_unique = c_top5 - q_top5
    
    if q_unique:
        print(f"\n⚛️  Solo Quantum top 5:")
        for feature in q_unique:
            val = q_perm[q_perm['feature'] == feature]['importance_mean'].iloc[0]
            print(f"      • {feature}: {val:.6f}")
    
    if c_unique:
        print(f"\n🔷 Solo Classical top 5:")
        for feature in c_unique:
            val = c_perm[c_perm['feature'] == feature]['importance_mean'].iloc[0]
            print(f"      • {feature}: {val:.6f}")
    
    # Statistiche
    print(f"\n📈 STATISTICHE:")
    print("-" * 15)
    print(f"   Quantum importance range:  {q_perm['importance_mean'].min():.6f} to {q_perm['importance_mean'].max():.6f}")
    print(f"   Classical importance range: {c_perm['importance_mean'].min():.6f} to {c_perm['importance_mean'].max():.6f}")
    
    # Correlazione overall
    merged = q_perm[['feature', 'importance_mean']].merge(
        c_perm[['feature', 'importance_mean']], on='feature', suffixes=('_q', '_c')
    )
    
    if len(merged) > 0:
        correlation = np.corrcoef(merged['importance_mean_q'], merged['importance_mean_c'])[0, 1]
        print(f"   Correlazione feature importance: {correlation:.4f}")
        
        if correlation > 0.7:
            print(f"   ✅ Alta correlazione - modelli concordi")
        elif correlation > 0.4:
            print(f"   ⚠️  Media correlazione - parzialmente concordi")
        else:
            print(f"   ❌ Bassa correlazione - modelli discordanti")
    
    print(f"\n✅ Analisi Permutation Importance completata!")
    print("="*60)