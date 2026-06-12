import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Résultats OIHACDB
oihacdb_results = {
    "Centralisé": 0.96,
    "FL IID": 0.98,
    "FL Non-IID": 0.62
}


# Résultats AHCD
ahcd_results = {
    "Centralisé": 0.9006,
    "FL IID": 0.9182,
    "FL Non-IID": 0.7268
}

# Tableau comparatif global
df = pd.DataFrame({
    'Dataset': ['OIHACDB', 'OIHACDB', 'OIHACDB', 'AHCD', 'AHCD', 'AHCD'],
    'Méthode': ['Centralisé', 'FL IID', 'FL Non-IID'] * 2,
    'Accuracy': [
        oihacdb_results['Centralisé'],
        oihacdb_results['FL IID'],
        oihacdb_results['FL Non-IID'],
        ahcd_results['Centralisé'],
        ahcd_results['FL IID'],
        ahcd_results['FL Non-IID']
    ]
})

print("\n" + "="*70)
print("COMPARAISON GLOBALE - OIHACDB vs AHCD")
print("="*70)
print("\n", df.to_string(index=False))


# Graphique 1 : Barres groupées
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

methods = ['Centralisé', 'FL IID', 'FL Non-IID']
x = np.arange(len(methods))
width = 0.35

oihacdb_acc = [
    oihacdb_results['Centralisé'],
    oihacdb_results['FL IID'],
    oihacdb_results['FL Non-IID']
]

ahcd_acc = [
    ahcd_results['Centralisé'],
    ahcd_results['FL IID'],
    ahcd_results['FL Non-IID']
]

# Subplot 1 : Barres groupées
bars1 = ax1.bar(x - width/2, oihacdb_acc, width, label='OIHACDB', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, ahcd_acc, width, label='AHCD', alpha=0.8, edgecolor='black')

ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Comparaison par Dataset', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 1)

# Subplot 2 : Impact Non-IID
datasets = ['OIHACDB', 'AHCD']
iid_accs = [oihacdb_results['FL IID'], ahcd_results['FL IID']]
non_iid_accs = [oihacdb_results['FL Non-IID'], ahcd_results['FL Non-IID']]
degradations = [iid - non_iid for iid, non_iid in zip(iid_accs, non_iid_accs)]

colors_deg = ['#e74c3c', '#e74c3c']
bars = ax2.bar(datasets, degradations, color=colors_deg, alpha=0.8, edgecolor='black')

ax2.set_ylabel('Dégradation IID → Non-IID', fontsize=12, fontweight='bold')
ax2.set_title('Impact de l\'Hétérogénéité', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.grid(axis='y', alpha=0.3)

for bar, deg in zip(bars, degradations):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f'{deg:.2f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../../results/FINAL_COMP.png', dpi=300, bbox_inches='tight')
print(f"\n Graphique sauvegardé : results/FINAL_COMP.png")


# Analyse des dégradations
print("\n" + "="*70)
print("ANALYSE DES DÉGRADATIONS IID → Non-IID")
print("="*70)

oihacdb_deg = oihacdb_results['FL IID'] - oihacdb_results['FL Non-IID']
ahcd_deg = ahcd_results['FL IID'] - ahcd_results['FL Non-IID']

print(f"\nOIHACDB :")
print(f"  FL IID     : {oihacdb_results['FL IID']:.4f}")
print(f"  FL Non-IID : {oihacdb_results['FL Non-IID']:.4f}")
print(f"  Dégradation: {oihacdb_deg:.4f} ({(oihacdb_deg/oihacdb_results['FL IID']*100):.1f}%)")

print(f"\nAHCD :")
print(f"  FL IID     : {ahcd_results['FL IID']:.4f}")
print(f"  FL Non-IID : {ahcd_results['FL Non-IID']:.4f}")
print(f"  Dégradation: {ahcd_deg:.4f} ({(ahcd_deg/ahcd_results['FL IID']*100):.1f}%)")


# Tableau récapitulatif final
print("\n" + "="*70)
print("TABLEAU RÉCAPITULATIF FINAL")
print("="*70)

summary = pd.DataFrame({
    'Dataset': ['OIHACDB', 'AHCD'],
    'Centralisé': [oihacdb_results['Centralisé'], ahcd_results['Centralisé']],
    'FL IID': [oihacdb_results['FL IID'], ahcd_results['FL IID']],
    'FL Non-IID': [oihacdb_results['FL Non-IID'], ahcd_results['FL Non-IID']],
    'Dégradation': [oihacdb_deg, ahcd_deg]
})

print("\n", summary.to_string(index=False))

# Sauvegarder le tableau
summary.to_csv('../../results/FINAL_COMP_table.csv', index=False)
print(f"\n Tableau sauvegardé : results/FINAL_COMP_table.csv")

print("\n" + "="*70)
print(" COMPARAISON GLOBALE TERMINÉE")
print("="*70)

plt.show()