from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def print_performance_metrics(y_test, y_pred, title="RISULTATI MODELLO"):
    """
    Calcola e stampa le metriche di performance per un modello di classificazione binaria.
    
    Parameters:
    -----------
    y_test : array-like
        Etichette vere
    y_pred : array-like  
        Etichette predette
    title : str, optional
        Titolo da stampare (default: "RISULTATI MODELLO")
    """
    # Calcola le metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Stampa i risultati
    print("\n" + "="*50)
    print(title)
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred))

def get_performance_metrics(y_test, y_pred):
    """
    Calcola e restituisce le metriche di performance senza stamparle.
    
    Parameters:
    -----------
    y_test : array-like
        Etichette vere
    y_pred : array-like  
        Etichette predette
        
    Returns:
    --------
    dict : Dizionario con le metriche calcolate
    """
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary')
    }