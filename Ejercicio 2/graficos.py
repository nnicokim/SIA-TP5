# =====================================================================
#  graficos.py  -  Ejercicio 2 (VAE de Kanjis)
# ---------------------------------------------------------------------
#  Lee los artefactos generados por experimento.py (CSV / JSON / NPZ)
#  y produce TODAS las figuras de la presentacion. No entrena nada:
#  se puede re-graficar las veces que haga falta sin re-entrenar.
# =====================================================================
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader_vae import cargar_datos_kanji

DPI = 200


def leer_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cargar_json(path):
    with open(path) as f:
        return json.load(f)


# =====================================================================
#  0. DATASET: las 32 entradas (lo que faltaba en la presentacion)
# =====================================================================
def fig_dataset():
    X = cargar_datos_kanji("symbol.h")
    fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
    fig.suptitle("Dataset: 32 kanjis  |  cada uno es una grilla binaria 10x10 (100 pixeles)",
                 fontsize=13, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(10, 10), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"#{i + 1}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("fig_dataset.png", dpi=DPI)
    plt.close()
    print("[OK] fig_dataset.png")


# =====================================================================
#  Helper: barra (recon) + linea (epocas-a-converger) en eje gemelo
# =====================================================================
def barra_recon_epocas(filas, etiquetas, titulo, xlabel, fname):
    rec = [float(f["loss_recon"]) for f in filas]
    ep = [int(f["epochs_converged"]) for f in filas]
    colaps = [int(f["colapsado"]) or int(f["diverged"]) for f in filas]
    x = np.arange(len(filas))

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    colores = ["#c0392b" if c else "#5dade2" for c in colaps]
    barras = ax1.bar(x, rec, color=colores, edgecolor="black", zorder=3)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("BCE de reconstruccion al converger\n(menor = mejor)  [barras]", color="#1f618d")
    ax1.set_xticks(x); ax1.set_xticklabels(etiquetas, rotation=0)
    ax1.tick_params(axis="y", labelcolor="#1f618d")
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for b, v, c in zip(barras, rec, colaps):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
        if c:
            ax1.text(b.get_x() + b.get_width() / 2, v / 2, "COLAPSO",
                     ha="center", va="center", rotation=90, color="white",
                     fontsize=10, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, ep, "o--", color="#7d3c98", linewidth=2, markersize=8)
    ax2.set_ylabel("Epocas hasta converger (early stopping)  [linea]", color="#7d3c98")
    ax2.tick_params(axis="y", labelcolor="#7d3c98")
    ax2.set_ylim(bottom=0)

    plt.title(titulo, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=DPI)
    plt.close()
    print(f"[OK] {fname}")


# =====================================================================
#  1 y 2. Barridos de ARQUITECTURA y LEARNING RATE
# =====================================================================
def figs_sweeps():
    filas = leer_csv("resultados_sweeps.csv")

    arch = [f for f in filas if f["sweep"] == "arquitectura"]
    barra_recon_epocas(
        arch, [f["arch"] for f in arch],
        "Comparacion de ARQUITECTURAS  (LR y beta fijos)\n"
        "Criterio: reconstruccion al converger + velocidad de convergencia",
        "Arquitectura (capas ocultas del encoder)",
        "grafico_1_arquitecturas.png")

    lr = [f for f in filas if f["sweep"] == "learning_rate"]
    barra_recon_epocas(
        lr, [f["lr"] for f in lr],
        "Comparacion de LEARNING RATE  (mejor arquitectura, beta fijo)\n"
        "Criterio: reconstruccion al converger + velocidad de convergencia",
        "Learning rate",
        "grafico_2_learning_rates.png")


# =====================================================================
#  3. Barrido de BETA-KL: estudio del POSTERIOR COLLAPSE
# =====================================================================
def fig_beta():
    filas = [f for f in leer_csv("resultados_sweeps.csv") if f["sweep"] == "beta"]
    filas = sorted(filas, key=lambda f: float(f["beta"]))
    betas = [float(f["beta"]) for f in filas]
    rec = [float(f["loss_recon"]) for f in filas]
    kl = [float(f["loss_kl"]) for f in filas]
    colaps = [int(f["colapsado"]) for f in filas]
    x = np.arange(len(betas))

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(x, rec, color="#5dade2", edgecolor="black", zorder=3,
            label="BCE reconstruccion")
    ax1.set_xlabel("Beta  (peso de la divergencia KL)")
    ax1.set_ylabel("BCE de reconstruccion", color="#1f618d")
    ax1.tick_params(axis="y", labelcolor="#1f618d")
    ax1.set_xticks(x); ax1.set_xticklabels([str(b) for b in betas])
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    ax2 = ax1.twinx()
    ax2.plot(x, kl, "s--", color="#cb4335", linewidth=2, markersize=9,
             label="Divergencia KL")
    ax2.set_ylabel("Divergencia KL (KL -> 0 = colapso)", color="#cb4335")
    ax2.tick_params(axis="y", labelcolor="#cb4335")
    ax2.axhline(0, color="#cb4335", linestyle=":", alpha=0.5)

    for xi, c in zip(x, colaps):
        if c:
            ax1.axvspan(xi - 0.45, xi + 0.45, color="#f9e79f", alpha=0.5, zorder=0)
    ax1.plot([], [], color="#f9e79f", linewidth=8, alpha=0.6, label="Posterior collapse")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left")
    plt.title("Estudio del peso KL (beta) y el POSTERIOR COLLAPSE\n"
              "beta chico: KL activa y buena reconstruccion | beta grande: KL=0, latente colapsado",
              fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig("grafico_3_beta_kl.png", dpi=DPI)
    plt.close()
    print("[OK] grafico_3_beta_kl.png")


# =====================================================================
#  4. Curvas de aprendizaje del OPTIMO (recon / KL / total) + early stop
# =====================================================================
def fig_curvas_optimo():
    c = cargar_json("optimo_curvas.json")
    meta = cargar_json("optimo_meta.json")
    ep = c["epochs_converged"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(c["recon"], color="#1f618d", label="Reconstruccion (BCE)")
    ax.plot([meta["beta"] * v for v in c["kl"]], color="#cb4335",
            label=f"KL x beta ({meta['beta']})")
    ax.plot(c["total"], color="black", linewidth=2, label="Loss total")
    ax.axvline(ep - 1, color="green", linestyle="--",
               label=f"Convergencia (early stop) @ {ep} ep")
    ax.set_xlabel("Epocas")
    ax.set_ylabel("Loss")
    ax.set_title(f"Curvas de aprendizaje del modelo optimo\n"
                 f"arch={meta['arch']} | lr={meta['lr']} | beta={meta['beta']}",
                 fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_curvas_optimo.png", dpi=DPI)
    plt.close()
    print("[OK] fig_curvas_optimo.png")


# =====================================================================
#  5. CONSIGNA B: espacio latente COLOREADO POR ERROR de reconstruccion
# =====================================================================
def fig_latente_b():
    filas = leer_csv("optimo_latente.csv")
    meta = cargar_json("optimo_meta.json")
    z1 = np.array([float(f["z1"]) for f in filas])
    z2 = np.array([float(f["z2"]) for f in filas])
    err = np.array([float(f["recon_error"]) for f in filas])

    plt.figure(figsize=(8.5, 7.5))
    sc = plt.scatter(z1, z2, c=err, cmap="viridis", s=220, edgecolors="k", alpha=0.9)
    cb = plt.colorbar(sc)
    cb.set_label("Error de reconstruccion por kanji (BCE)")
    for i, f in enumerate(filas):
        plt.annotate(f["kanji_idx"], (z1[i] + 0.04, z2[i] + 0.04),
                     fontsize=8, weight="bold")
    plt.axhline(0, color="grey", linestyle="--", alpha=0.5)
    plt.axvline(0, color="grey", linestyle="--", alpha=0.5)
    plt.title(f"CONSIGNA B: Espacio latente 2D (color = error de reconstruccion)\n"
              f"arch={meta['arch']} | lr={meta['lr']} | beta={meta['beta']} | "
              f"error medio={meta['recon_error_medio']:.3f}",
              fontweight="bold", fontsize=11)
    plt.xlabel("Dimension latente 1 (mu)")
    plt.ylabel("Dimension latente 2 (mu)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("consigna_b_espacio_latente.png", dpi=DPI)
    plt.close()
    print("[OK] consigna_b_espacio_latente.png")


# =====================================================================
#  6. Reconstrucciones: original vs reconstruido (lo que faltaba)
# =====================================================================
def fig_reconstrucciones(n=8):
    d = np.load("optimo_reconstrucciones.npz")
    X, Xr, err = d["X"], d["X_recon"], d["err"]
    idx = np.arange(n)
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.6))
    fig.suptitle("Reconstruccion del VAE optimo  (arriba: original | abajo: reconstruido)",
                 fontsize=12, fontweight="bold")
    for c, i in enumerate(idx):
        axes[0, c].imshow(X[i].reshape(10, 10), cmap="gray_r", vmin=0, vmax=1)
        axes[0, c].set_title(f"#{i + 1}", fontsize=8)
        axes[1, c].imshow(Xr[i].reshape(10, 10), cmap="gray_r", vmin=0, vmax=1)
        axes[1, c].set_title(f"BCE {err[i]:.2f}", fontsize=8)
        for ax in (axes[0, c], axes[1, c]):
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("fig_reconstrucciones.png", dpi=DPI)
    plt.close()
    print("[OK] fig_reconstrucciones.png")


# =====================================================================
#  7. CONSIGNA C: kanjis generados muestreando z ~ N(0,1)
# =====================================================================
def fig_generados(npz, fname, titulo):
    d = np.load(npz)
    zs, imgs = d["zs"], d["imgs"]
    n = imgs.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    for i in range(n):
        raw = imgs[i].reshape(10, 10)
        axes[0, i].imshow(raw, cmap="gray_r", vmin=0, vmax=1)
        axes[0, i].set_title(f"z=({zs[i, 0]:.1f}, {zs[i, 1]:.1f})", fontsize=9)
        axes[1, i].imshow((raw > 0.5).astype(float), cmap="gray_r", vmin=0, vmax=1)
        for ax in (axes[0, i], axes[1, i]):
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("Probabilidades", fontsize=10)
    axes[1, 0].set_ylabel("Binarizado (>0.5)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(fname, dpi=DPI)
    plt.close()
    print(f"[OK] {fname}")


# =====================================================================
#  8. Barrido continuo del espacio latente (continuidad del VAE)
# =====================================================================
def fig_barrido():
    d = np.load("optimo_barrido_latente.npz")
    grid, lim = d["grid"], float(d["lim"])
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray_r", extent=[-lim, lim, -lim, lim], vmin=0, vmax=1)
    plt.title("CONSIGNA C: barrido continuo del espacio latente\n"
              "(decodificando una grilla de z -> continuidad del VAE)",
              fontweight="bold", fontsize=11)
    plt.xlabel("Dimension latente 1")
    plt.ylabel("Dimension latente 2")
    plt.tight_layout()
    plt.savefig("fig_barrido_latente.png", dpi=DPI)
    plt.close()
    print("[OK] fig_barrido_latente.png")


# =====================================================================
#  9. POSTERIOR COLLAPSE: latente colapsado + KL -> 0 + generacion
# =====================================================================
def figs_colapso():
    if not os.path.exists("colapso_latente.csv"):
        print("[..] sin caso de colapso (ningun beta colapso)")
        return
    filas = leer_csv("colapso_latente.csv")
    cur = cargar_json("colapso_curvas.json")
    z1 = np.array([float(f["z1"]) for f in filas])
    z2 = np.array([float(f["z2"]) for f in filas])
    beta = cur["beta"]

    # Latente colapsado (todos los codigos en ~un punto) + curva KL -> 0
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.5))
    axa.scatter(z1, z2, c=range(len(filas)), cmap="tab20", s=180,
                edgecolors="k", zorder=3)
    # Mismos limites que el latente "bueno" para que el contraste sea evidente:
    # los 32 kanjis quedan amontonados en el origen en vez de repartidos.
    axa.set_xlim(-3, 3); axa.set_ylim(-3, 3)
    axa.axhline(0, color="grey", linestyle="--", alpha=0.5)
    axa.axvline(0, color="grey", linestyle="--", alpha=0.5)
    axa.annotate("los 32 codigos colapsan\nen el origen",
                 xy=(0, 0), xytext=(1.2, 1.6), fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="black"))
    axa.set_title(f"Espacio latente COLAPSADO (beta={beta})\n"
                  "todos los kanjis caen en el mismo punto")
    axa.set_xlabel("Dimension latente 1 (mu)")
    axa.set_ylabel("Dimension latente 2 (mu)")
    axa.grid(True, alpha=0.3)

    axb.plot(cur["kl"], color="#cb4335", label="KL")
    axb.plot(cur["recon"], color="#1f618d", label="Reconstruccion (BCE)")
    axb.set_title("Evidencia del colapso: la KL se va a 0\n"
                  "y la reconstruccion se estanca (decoder ignora z)")
    axb.set_xlabel("Epocas"); axb.set_ylabel("Loss")
    axb.grid(True, alpha=0.3); axb.legend()
    plt.tight_layout()
    plt.savefig("consigna_b_espacio_latente_collapse.png", dpi=DPI)
    plt.close()
    print("[OK] consigna_b_espacio_latente_collapse.png")

    if os.path.exists("colapso_generados.npz"):
        fig_generados("colapso_generados.npz",
                      "consigna_c_kanjis_generados_collapse.png",
                      f"CONSIGNA C bajo COLAPSO (beta={beta}): "
                      "el decoder genera siempre lo mismo")


def main():
    print("Generando figuras desde los artefactos de experimento.py ...")
    fig_dataset()
    figs_sweeps()
    fig_beta()
    fig_curvas_optimo()
    fig_latente_b()
    fig_reconstrucciones()
    fig_generados("optimo_generados.npz", "consigna_c_kanjis_generados.png",
                  "CONSIGNA C: kanjis sinteticos generados por el VAE optimo (z ~ N(0,1))")
    fig_barrido()
    figs_colapso()
    print("\nListo. Todas las figuras regeneradas.")


if __name__ == "__main__":
    main()
