import os
import numpy as np
import matplotlib.pyplot as plt


RESULTS_PATH = "../../results"


print("Chargement des historiques...")

iid_history = np.load(os.path.join(RESULTS_PATH, "fl_iid_ahcd_history.npz"))
non_iid_history = np.load(os.path.join(RESULTS_PATH, "fl_non_iid_ahcd_history.npz"))

#Extraire les accuracies de test
acc_iid = iid_history['test_accuracy']
acc_non_iid = non_iid_history['test_accuracy']

print(f"IID - {len(acc_iid)} rounds chargés")
print(f"Non-IID - {len(acc_non_iid)} rounds chargés")


#Plot

plt.figure()
plt.plot(acc_iid, label="IID", marker="o")
plt.plot(acc_non_iid, label="Non-IID", marker="s")
plt.xlabel("Rounds fédérés")
plt.ylabel("Accuracy (jeu de test)")
plt.title("FL IID vs Non-IID – AHCD")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_PATH, "compare_iid_non_iid_ahcd.png"), dpi=300, bbox_inches='tight')
print(f"\n Graphique sauvegardé : {RESULTS_PATH}/compare_iid_non_iid_ahcd.png")
plt.show()


#Résultats finaux

print("\n" + "=",60)
print("RÉSULTATS FINAUX")
print("=",60)
print(f"FL IID     - Accuracy finale : {acc_iid[-1]:.4f}")
print(f"FL Non-IID - Accuracy finale : {acc_non_iid[-1]:.4f}")
print(f"Dégradation : {(acc_iid[-1] - acc_non_iid[-1]):.4f}")