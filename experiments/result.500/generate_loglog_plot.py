import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

# Caminho para o CSV com resultados (ajusta se necessário)
csv_path = Path("heuristic/benchmark_results.csv")

# Lê o CSV
df = pd.read_csv(csv_path)

# Mantém apenas linhas com dados do algoritmo guloso
df = df[df["greedy_operations"].notnull()]

# Agrupa por número de vértices e calcula média das operações (para diferentes densidades)
grouped = df.groupby("n_vertices")["greedy_operations"].mean().reset_index()

# Extrai dados
x = grouped["n_vertices"].values
y = grouped["greedy_operations"].values

# Ajuste log-log: log(y) = p*log(x) + log(c)
logx = np.log(x)
logy = np.log(y)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
p = slope
c = np.exp(intercept)

print(f"Coeficiente angular (p): {p:.3f}")
print(f"Constante (c): {c:.3f}")
print(f"R² do ajuste: {r_value**2:.4f}")

# Calcula valores ajustados
y_fit = c * x**p

# Cria figura
plt.figure(figsize=(8, 6))
plt.loglog(x, y, 'o', label='Dados experimentais (médias)')
plt.loglog(x, y_fit, '--', label=f"Ajuste: $y = {c:.2f} \\cdot n^{{{p:.2f}}}$")
plt.xlabel("Número de vértices (n)", fontsize=12)
plt.ylabel("Operações básicas (média)", fontsize=12)
plt.title("Ajuste log-log das operações do algoritmo guloso", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

# Guarda a figura
output_dir = Path("../plots.500")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "loglog_greedy_fit.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")


# I want the Op(15), Op(10), Op(20) 
print(f"Op(15): {y_fit[15]}")
print(f"Op(10): {y_fit[10]}")
print(f"Op(20): {y_fit[20]}")

# or Op(15) / Op(10) and Op(20) / Op(10)
print(f"Op(15) / Op(10): {y_fit[15] / y_fit[10]}")
print(f"Op(20) / Op(10): {y_fit[20] / y_fit[10]}")


print(f"✓ Figura gerada em {output_path}")
