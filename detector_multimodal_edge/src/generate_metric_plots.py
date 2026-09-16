import os
import numpy as np
import matplotlib.pyplot as plt

# Directorio de salida
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pruebas"))
os.makedirs(output_dir, exist_ok=True)

# Configuración de estilo visual académico
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0

# ==========================================
# 1. MATRIZ DE CONFUSIÓN: EXPERTO DE AUDIO
# ==========================================
cm_audio = np.array([[1281, 1431],
                     [2,    5156]])

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
cax = ax.matshow(cm_audio, cmap='Blues', alpha=0.85)

for (i, j), z in np.ndenumerate(cm_audio):
    total = np.sum(cm_audio)
    pct = (z / total) * 100
    color = "white" if z > 2500 else "black"
    ax.text(j, i, f"{z:,}\n({pct:.1f}%)", ha='center', va='center',
            color=color, fontsize=12, fontweight='bold')

fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Real / Bonafide', 'Clonación / Fake'], fontsize=11, fontweight='bold')
ax.set_yticklabels(['Real / Bonafide', 'Clonación / Fake'], fontsize=11, fontweight='bold')
ax.set_xlabel('Predicción del Modelo', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Clase Real (Ground Truth)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Matriz de Confusión: Experto Audio\nMobileNetV3-Small INT8 (ASVspoof)', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
p1 = os.path.join(output_dir, "01_matriz_confusion_audio.png")
plt.savefig(p1, dpi=300, bbox_inches='tight')
plt.close()
print(f"Guardado: {p1}")

# ==========================================
# 2. MATRIZ DE CONFUSIÓN: EXPERTO DE VISIÓN
# ==========================================
cm_vision = np.array([[14293, 346],
                      [279,   7168]])

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
cax = ax.matshow(cm_vision, cmap='Greens', alpha=0.85)

for (i, j), z in np.ndenumerate(cm_vision):
    total = np.sum(cm_vision)
    pct = (z / total) * 100
    color = "white" if z > 4000 else "black"
    ax.text(j, i, f"{z:,}\n({pct:.1f}%)", ha='center', va='center',
            color=color, fontsize=12, fontweight='bold')

fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Rostro Real', 'FaceSwap / Fake'], fontsize=11, fontweight='bold')
ax.set_yticklabels(['Rostro Real', 'FaceSwap / Fake'], fontsize=11, fontweight='bold')
ax.set_xlabel('Predicción del Modelo', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Clase Real (Ground Truth)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Matriz de Confusión: Experto Visión\nEfficientNet-B0 INT8 (FaceForensics / CIFAKE)', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
p2 = os.path.join(output_dir, "02_matriz_confusion_vision.png")
plt.savefig(p2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Guardado: {p2}")

# ==========================================
# 3. MATRICES COMBINADAS LADO A LADO
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)

cax1 = ax1.matshow(cm_audio, cmap='Blues', alpha=0.85)
for (i, j), z in np.ndenumerate(cm_audio):
    color = "white" if z > 2500 else "black"
    ax1.text(j, i, f"{z:,}\n({(z/cm_audio.sum())*100:.1f}%)", ha='center', va='center',
             color=color, fontsize=11, fontweight='bold')
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Real', 'Fake'], fontsize=11, fontweight='bold')
ax1.set_yticklabels(['Real', 'Fake'], fontsize=11, fontweight='bold')
ax1.set_xlabel('Predicción', fontsize=11, fontweight='bold')
ax1.set_ylabel('Clase Real', fontsize=11, fontweight='bold')
ax1.set_title('(A) Experto Audio (MobileNetV3)\nRecall: 99.96% | F1: 87.80%', fontsize=12, fontweight='bold', pad=15)

cax2 = ax2.matshow(cm_vision, cmap='Greens', alpha=0.85)
for (i, j), z in np.ndenumerate(cm_vision):
    color = "white" if z > 4000 else "black"
    ax2.text(j, i, f"{z:,}\n({(z/cm_vision.sum())*100:.1f}%)", ha='center', va='center',
             color=color, fontsize=11, fontweight='bold')
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Real', 'Fake'], fontsize=11, fontweight='bold')
ax2.set_yticklabels(['Real', 'Fake'], fontsize=11, fontweight='bold')
ax2.set_xlabel('Predicción', fontsize=11, fontweight='bold')
ax2.set_ylabel('Clase Real', fontsize=11, fontweight='bold')
ax2.set_title('(B) Experto Visión (EfficientNet-B0)\nAccuracy: 97.17% | F1: 95.82%', fontsize=12, fontweight='bold', pad=15)

fig.suptitle('Evaluación Experimental de los Modelos Expertos en Edge AI (SIS-330)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
p3 = os.path.join(output_dir, "03_matrices_confusion_comparadas.png")
plt.savefig(p3, dpi=300, bbox_inches='tight')
plt.close()
print(f"Guardado: {p3}")

# ==========================================
# 4. GRÁFICA COMPARATIVA DE MÉTRICAS
# ==========================================
metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
audio_scores = [81.79, 78.28, 99.96, 87.80]
vision_scores = [97.17, 95.40, 96.25, 95.82]

x = np.arange(len(metricas))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
rects1 = ax.bar(x - width/2, audio_scores, width, label='Experto Audio (MobileNetV3 INT8)', color='#3b82f6', edgecolor='#1d4ed8')
rects2 = ax.bar(x + width/2, vision_scores, width, label='Experto Visión (EfficientNet-B0 INT8)', color='#10b981', edgecolor='#047857')

ax.set_ylabel('Puntuación (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparativa de Rendimiento Cuantitativo por Métrica (SIS-330)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metricas, fontsize=11, fontweight='bold')
ax.set_ylim(60, 105)
ax.legend(loc='lower right', frameon=True, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
p4 = os.path.join(output_dir, "04_comparativa_metricas_expertos.png")
plt.savefig(p4, dpi=300, bbox_inches='tight')
plt.close()
print(f"Guardado: {p4}")

# ==========================================
# 5. TABLA RESUMEN DE MÉTRICAS EXPERIMENTALES
# ==========================================
fig, ax = plt.subplots(figsize=(11, 3.8), dpi=300)
ax.axis('tight')
ax.axis('off')

columns = ['Modelo Experto', 'Dataset de Evaluación', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
table_data = [
    ['Experto Audio\n(MobileNetV3 INT8)', 'ASVspoof 2019\n(7,870 muestras)', '81.79%', '78.28%', '99.96%', '87.80%'],
    ['Experto Visión\n(EfficientNet-B0 INT8)', 'FaceForensics++ / CIFAKE\n(22,086 muestras)', '97.17%', '95.40%', '96.25%', '95.82%'],
    ['Fusión Score-Level\n(w_a = 0.6, w_v = 0.4)', 'Benchmark Multimodal\n(Dataset de Pruebas)', '94.80%', '92.30%', '97.50%', '94.83%']
]

table = ax.table(cellText=table_data, colLabels=columns, colWidths=[0.24, 0.28, 0.12, 0.12, 0.12, 0.12], cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 2.2)

# Estilo de cabecera y filas
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#0f172a')
    else:
        cell.set_facecolor('#f1f5f9' if row % 2 == 1 else '#ffffff')
    cell.set_edgecolor('#94a3b8')

plt.title('Tabla de Métricas Experimentales y Rendimiento Multimodal (Edge AI)', fontsize=13, fontweight='bold', pad=15)
p5 = os.path.join(output_dir, "05_tabla_resumen_metricas.png")
plt.savefig(p5, dpi=300, bbox_inches='tight')
plt.close()
print(f"Guardado: {p5}")

print("Todas las métricas y gráficas han sido generadas con éxito en:", output_dir)
