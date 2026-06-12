import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Résultats finaux 
results = {
    "Centralisé": 0.96,   
    "FL IID": 0.98,       
    "FL Non-IID": 0.62    
}


# Tableau comparatif
df = pd.DataFrame.from_dict(
    results, orient="index", columns=["Accuracy finale"]
)

print("\n COMPARAISON FINALE OIHACDB \n")
print(df)


# Visualisation
plt.figure()
plt.bar(df.index, df["Accuracy finale"])
plt.ylabel("Accuracy")
plt.title("Comparaison finale — Centralisé vs FL IID vs FL Non-IID (OIHACDB)")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.savefig("../../results/final_comparison_oihacdb.png", dpi=300, bbox_inches='tight')
print(f"\n Graphique sauvegardé : results/final_comparison_oihacdb.png")
plt.show()
