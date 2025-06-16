import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference, 
    equalized_odds_difference,
    MetricFrame
)

def fairness_check(y_true, y_pred, sensitive_features, plot=True):
    """
    🎯 Calcola le 4 metriche fairness essenziali + plot opzionale
    
    Args:
        y_true: target reali (array)
        y_pred: predizioni (array)
        sensitive_features: dict {'race': array, 'sex': array, ...}
        plot: se True mostra i grafici
    
    Returns:
        dict: risultati {attr: {metric: value}}
    """
    print("🔍 FAIRNESS METRICS")
    print("="*40)
    
    results = {}
    
    for attr_name, attr_values in sensitive_features.items():
        print(f"\n📊 {attr_name.upper()}:")
        
        try:
            # 4 metriche core
            spd = demographic_parity_difference(y_true, y_pred, sensitive_features=attr_values)
            eod = equal_opportunity_difference(y_true, y_pred, sensitive_features=attr_values)
            di = demographic_parity_ratio(y_true, y_pred, sensitive_features=attr_values)
            aod = equalized_odds_difference(y_true, y_pred, sensitive_features=attr_values, agg='mean')
            
            # Status
            def status_diff(val):
                return "✅" if abs(val) < 0.1 else "⚠️" if abs(val) < 0.2 else "❌"
            def status_ratio(val):
                return "✅" if val >= 0.8 else "⚠️" if val >= 0.6 else "❌"
            
            print(f"   Statistical Parity Diff: {spd:+.3f} {status_diff(spd)}")
            print(f"   Equal Opportunity Diff:  {eod:+.3f} {status_diff(eod)}")
            print(f"   Disparate Impact Ratio:  {di:.3f} {status_ratio(di)}")
            print(f"   Average Odds Diff:       {aod:+.3f} {status_diff(aod)}")
            
            results[attr_name] = {
                'statistical_parity_diff': spd,
                'equal_opportunity_diff': eod, 
                'disparate_impact': di,
                'average_odds_diff': aod
            }
            
        except Exception as e:
            print(f"   ❌ Errore: {e}")
            results[attr_name] = {'error': str(e)}
    
    # Summary
    print(f"\n💡 SUMMARY:")
    fair_count = 0
    total_count = 0
    
    for attr, metrics in results.items():
        if 'error' not in metrics:
            total_count += 1
            spd_fair = abs(metrics['statistical_parity_diff']) < 0.1
            eod_fair = abs(metrics['equal_opportunity_diff']) < 0.1  
            aod_fair = abs(metrics['average_odds_diff']) < 0.1
            di_fair = metrics['disparate_impact'] >= 0.8
            
            fair_metrics = sum([spd_fair, eod_fair, aod_fair, di_fair])
            
            if fair_metrics >= 3:  # 3/4 metriche fair
                fair_count += 1
                print(f"   ✅ {attr}: Fair ({fair_metrics}/4)")
            else:
                print(f"   ❌ {attr}: Biased ({fair_metrics}/4)")
    
    fairness_rate = fair_count / total_count if total_count > 0 else 0
    print(f"\n🎯 Overall: {fairness_rate:.1%} fair")
    
    # Plot opzionale
    if plot:
        plot_fairness(results)
    
    return results

def plot_fairness(results):
    """
    📈 Grafici delle metriche fairness
    """
    # Filtra errori
    valid = {k: v for k, v in results.items() if 'error' not in v}
    if not valid:
        print("❌ Nessun dato da visualizzare")
        return
    
    features = list(valid.keys())
    spd_vals = [abs(valid[f]['statistical_parity_diff']) for f in features]
    eod_vals = [abs(valid[f]['equal_opportunity_diff']) for f in features]
    aod_vals = [abs(valid[f]['average_odds_diff']) for f in features]
    di_vals = [1 - valid[f]['disparate_impact'] for f in features]  # Distanza da 1
    
    x = np.arange(len(features))
    width = 0.2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('🔍 Fairness Metrics', fontsize=14, fontweight='bold')
    
    # Difference metrics
    ax1.bar(x - 1.5*width, spd_vals, width, label='Statistical Parity', alpha=0.8)
    ax1.bar(x - 0.5*width, eod_vals, width, label='Equal Opportunity', alpha=0.8)
    ax1.bar(x + 0.5*width, aod_vals, width, label='Average Odds', alpha=0.8)
    
    ax1.set_title('Difference Metrics (Lower = Better)')
    ax1.set_ylabel('Absolute Difference')
    ax1.set_xlabel('Sensitive Features')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7)
    ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.7)
    
    # Disparate Impact
    ax2.bar(features, di_vals, alpha=0.8, color='purple')
    ax2.set_title('Disparate Impact (Distance from 1.0)')
    ax2.set_ylabel('Distance from Perfect Parity')
    ax2.set_xlabel('Sensitive Features')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.4, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()