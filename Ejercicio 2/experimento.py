# =====================================================================
#  experimento.py  -  Ejercicio 2 (VAE de Kanjis)
# ---------------------------------------------------------------------
#  Entrena TODO en CPU (NumPy puro) con EARLY STOPPING y vuelca los
#  resultados a CSV / JSON / NPZ. NO grafica nada: de eso se encarga
#  graficos.py. Asi se puede re-graficar sin re-entrenar.
#
#  Metodologia (barrido de un factor por vez, en vez de epocas fijas):
#    1. Arquitectura : se varia arch, fijando LR_BASE y BETA_BASE.
#    2. Learning rate: se varia lr,   fijando la mejor arch y BETA_BASE.
#    3. Beta (KL)    : se varia beta, fijando la mejor arch y el mejor lr.
#                      -> estudio del POSTERIOR COLLAPSE.
#  El "cuanto entrenar" lo decide early stopping (epocas-a-converger),
#  no un numero magico de epocas.
# =====================================================================
import os
import csv
import json
import time
import numpy as np

from data_loader_vae import cargar_datos_kanji
from models_vae import build_vae
from config_vae import load_config_experimento


# ---------------------------------------------------------------------
#  Utilidades
# ---------------------------------------------------------------------
def bce_por_muestra(X, X_recon):
    """BCE sumada sobre los 100 pixeles, por cada kanji. Shape (N,)."""
    eps = 1e-12
    xr = np.clip(X_recon, eps, 1.0 - eps)
    bce = -(X * np.log(xr) + (1.0 - X) * np.log(1.0 - xr))
    return np.sum(bce, axis=1)


def entrenar_con_early_stopping(core, X, max_epochs, patience, min_delta):
    """
    Entrena llamando a core.train_step hasta convergencia.
    Corta si la loss total no mejora mas de min_delta durante `patience`
    epocas seguidas, o si diverge (NaN/Inf), o al llegar a max_epochs.
    """
    tot, rec, kl = [], [], []
    best = np.inf
    best_epoch = 0
    wait = 0
    diverged = False

    for e in range(max_epochs):
        t, r, k = core.train_step(X)
        if not np.isfinite(t):
            diverged = True
            break
        tot.append(float(t)); rec.append(float(r)); kl.append(float(k))

        if best - t > min_delta:
            best = t
            best_epoch = e
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    epochs_run = len(tot)
    epochs_converged = (best_epoch + 1) if epochs_run > 0 else 0
    return {
        "total": tot, "recon": rec, "kl": kl,
        "epochs_run": epochs_run,
        "epochs_converged": epochs_converged,
        "diverged": diverged,
    }


def metricas_en_convergencia(res):
    """Snapshot de las metricas en la epoca de convergencia (best epoch)."""
    if res["epochs_run"] == 0:
        return float("nan"), float("nan"), float("nan")
    idx = res["epochs_converged"] - 1
    return res["total"][idx], res["recon"][idx], res["kl"][idx]


# ---------------------------------------------------------------------
#  Una corrida = entrenar + medir + extraer latente
# ---------------------------------------------------------------------
def correr(sweep, arch, lr, beta, X, latent, cfg, curvas, filas):
    np.random.seed(cfg["seed"])  # misma init/ruido por config -> reproducible y comparable
    vae, encoder, decoder = build_vae(
        X.shape[1], arch, latent, lr, use_cuda=False, kl_weight=beta
    )
    res = entrenar_con_early_stopping(
        vae.core, X, cfg["max_epochs"], cfg["patience"], cfg["min_delta"]
    )
    loss_tot, loss_rec, loss_kl = metricas_en_convergencia(res)

    # Actividad del latente (para detectar colapso): dispersion de los codigos z_mean.
    z_mean, z_log_var, _ = encoder.predict(X)
    z_std_medio = float(np.mean(np.std(z_mean, axis=0))) if res["epochs_run"] else 0.0
    # Colapso = los codigos casi no se separan entre kanjis distintos.
    colapsado = (not res["diverged"]) and (z_std_medio < 0.1)

    cid = f"{sweep}|arch={arch}|lr={lr}|beta={beta}"
    curvas[cid] = {"total": res["total"], "recon": res["recon"], "kl": res["kl"]}

    filas.append({
        "sweep": sweep,
        "arch": str(arch),
        "lr": lr,
        "beta": beta,
        "latent_dim": latent,
        "seed": cfg["seed"],
        "epochs_run": res["epochs_run"],
        "epochs_converged": res["epochs_converged"],
        "loss_total": loss_tot,
        "loss_recon": loss_rec,
        "loss_kl": loss_kl,
        "z_std_medio": z_std_medio,
        "colapsado": int(colapsado),
        "diverged": int(res["diverged"]),
    })

    estado = "DIVERGIO" if res["diverged"] else (f"recon={loss_rec:.3f} kl={loss_kl:.1f} "
                                                 f"conv@{res['epochs_converged']}ep "
                                                 f"{'COLAPSO' if colapsado else 'ok'}")
    print(f"  [{sweep:12s}] arch={str(arch):14s} lr={lr:<7} beta={beta:<6} -> {estado}")
    return res, encoder, decoder, vae


def main():
    t0 = time.time()
    cfg = load_config_experimento()
    print("=" * 70)
    print("EXPERIMENTO VAE (CPU / NumPy)  -  Early stopping, semilla fija")
    print(f"  seed={cfg['seed']}  latente={cfg['latent_dim']}  "
          f"max_epochs={cfg['max_epochs']} patience={cfg['patience']} "
          f"min_delta={cfg['min_delta']}")
    print("=" * 70)

    X = cargar_datos_kanji("symbol.h")
    latent = cfg["latent_dim"]

    curvas = {}   # cid -> curvas de loss
    filas = []    # filas del CSV de barridos

    # ----- BARRIDO 1: ARQUITECTURA -----
    print("\n[1/3] Barrido de ARQUITECTURAS "
          f"(LR={cfg['lr_base']}, beta={cfg['beta_base']})")
    for arch in cfg["archs"]:
        correr("arquitectura", arch, cfg["lr_base"], cfg["beta_base"],
               X, latent, cfg, curvas, filas)
    cand = [f for f in filas if f["sweep"] == "arquitectura" and not f["diverged"]]
    mejor_arch = json.loads(min(cand, key=lambda f: f["loss_recon"])["arch"])
    print(f"  >>> Mejor arquitectura: {mejor_arch} (menor BCE de reconstruccion)")

    # ----- BARRIDO 2: LEARNING RATE -----
    print(f"\n[2/3] Barrido de LEARNING RATE (arch={mejor_arch}, beta={cfg['beta_base']})")
    for lr in cfg["lrs"]:
        correr("learning_rate", mejor_arch, lr, cfg["beta_base"],
               X, latent, cfg, curvas, filas)
    cand = [f for f in filas if f["sweep"] == "learning_rate"
            and not f["diverged"] and not f["colapsado"]]
    mejor_lr = min(cand, key=lambda f: f["loss_recon"])["lr"]
    print(f"  >>> Mejor learning rate: {mejor_lr}")

    # ----- BARRIDO 3: BETA (KL) / POSTERIOR COLLAPSE -----
    print(f"\n[3/3] Barrido de BETA-KL (arch={mejor_arch}, lr={mejor_lr})")
    for beta in cfg["betas"]:
        correr("beta", mejor_arch, mejor_lr, beta,
               X, latent, cfg, curvas, filas)
    activos = [f for f in filas if f["sweep"] == "beta"
               and not f["diverged"] and not f["colapsado"]]
    if activos:
        min_rec = min(f["loss_recon"] for f in activos)
        # Para un VAE GENERATIVO interesa la MAYOR regularizacion (beta mas alto)
        # que NO colapse: asi el posterior queda lo mas cerca posible del prior
        # N(0,1) -> mejor muestreo en la consigna C. Se exige que la reconstruccion
        # siga siendo buena (dentro de 10x de la mejor) para no irse al colapso.
        elegibles = [f for f in activos if f["loss_recon"] <= 10.0 * min_rec]
        mejor_beta = max(elegibles, key=lambda f: f["beta"])["beta"]
    else:
        mejor_beta = cfg["beta_base"]
    print(f"  >>> Beta optimo (max regularizacion sin colapso): {mejor_beta}")

    # Caso de colapso para las slides: el beta mas alto que SI colapso (si existe).
    colapsos = [f for f in filas if f["sweep"] == "beta" and f["colapsado"]]
    beta_colapso = max(colapsos, key=lambda f: f["beta"])["beta"] if colapsos else None

    # ----- GUARDAR CSV DE BARRIDOS + CURVAS -----
    campos = ["sweep", "arch", "lr", "beta", "latent_dim", "seed",
              "epochs_run", "epochs_converged", "loss_total", "loss_recon",
              "loss_kl", "z_std_medio", "colapsado", "diverged"]
    with open("resultados_sweeps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    with open("curvas_sweeps.json", "w") as f:
        json.dump(curvas, f)
    print("\n[OK] resultados_sweeps.csv + curvas_sweeps.json")

    # ----- MODELO OPTIMO (arch + lr + beta ganadores) -----
    print(f"\n[FINAL] Entrenando modelo OPTIMO: arch={mejor_arch} "
          f"lr={mejor_lr} beta={mejor_beta}")
    res, encoder, decoder, vae = correr(
        "optimo", mejor_arch, mejor_lr, mejor_beta, X, latent, cfg, curvas, filas
    )

    # Consigna B: codigos latentes (z_mean) + error de reconstruccion por kanji
    z_mean, z_log_var, _ = encoder.predict(X)
    X_recon = decoder.predict(z_mean)            # recon deterministica (sin ruido)
    err = bce_por_muestra(X, X_recon)            # error por kanji
    with open("optimo_latente.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kanji_idx", "z1", "z2", "recon_error"])
        for i in range(X.shape[0]):
            w.writerow([i + 1, float(z_mean[i, 0]), float(z_mean[i, 1]), float(err[i])])
    np.savez("optimo_reconstrucciones.npz", X=X, X_recon=X_recon, err=err)

    # Consigna C: 5 kanjis nuevos muestreando z ~ N(0,1)
    np.random.seed(cfg["seed"] + 1)
    zs = np.random.normal(0.0, 1.0, size=(5, latent))
    gen = decoder.predict(zs)
    np.savez("optimo_generados.npz", zs=zs, imgs=gen)

    # Consigna C (extra): barrido continuo del espacio latente (continuidad del VAE)
    n = 12
    lim = float(np.max(np.abs(z_mean))) * 1.1
    lim = max(lim, 2.5)
    gx = np.linspace(-lim, lim, n)
    gy = np.linspace(lim, -lim, n)
    grid = np.zeros((n * 10, n * 10))
    for r, yy in enumerate(gy):
        for c, xx in enumerate(gx):
            img = decoder.predict(np.array([[xx, yy]]))[0].reshape(10, 10)
            grid[r * 10:(r + 1) * 10, c * 10:(c + 1) * 10] = img
    np.savez("optimo_barrido_latente.npz", grid=grid, lim=lim)

    # Curvas del optimo (para la figura de convergencia con marca de early stopping)
    with open("optimo_curvas.json", "w") as f:
        json.dump({"total": res["total"], "recon": res["recon"], "kl": res["kl"],
                   "epochs_converged": res["epochs_converged"]}, f)

    # Caso de COLAPSO (beta alto) para las slides de posterior collapse
    if beta_colapso is not None:
        np.random.seed(cfg["seed"])
        vae_c, enc_c, dec_c = build_vae(X.shape[1], mejor_arch, latent,
                                        mejor_lr, use_cuda=False, kl_weight=beta_colapso)
        res_c = entrenar_con_early_stopping(vae_c.core, X, cfg["max_epochs"],
                                            cfg["patience"], cfg["min_delta"])
        zc, _, _ = enc_c.predict(X)
        with open("colapso_latente.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kanji_idx", "z1", "z2"])
            for i in range(X.shape[0]):
                w.writerow([i + 1, float(zc[i, 0]), float(zc[i, 1])])
        with open("colapso_curvas.json", "w") as f:
            json.dump({"total": res_c["total"], "recon": res_c["recon"],
                       "kl": res_c["kl"], "beta": beta_colapso}, f)
        # Generacion bajo colapso: todas las muestras salen casi identicas (el "kanji promedio")
        np.random.seed(cfg["seed"] + 1)
        zc_gen = np.random.normal(0.0, 1.0, size=(5, latent))
        np.savez("colapso_generados.npz", zs=zc_gen, imgs=dec_c.predict(zc_gen))

    # ----- META DEL OPTIMO -----
    meta = {
        "arch": mejor_arch,
        "lr": mejor_lr,
        "beta": mejor_beta,
        "latent_dim": latent,
        "seed": cfg["seed"],
        "epochs_converged": res["epochs_converged"],
        "loss_total": res["total"][res["epochs_converged"] - 1],
        "loss_recon": res["recon"][res["epochs_converged"] - 1],
        "loss_kl": res["kl"][res["epochs_converged"] - 1],
        "recon_error_medio": float(np.mean(err)),
        "beta_colapso": beta_colapso,
        "z_range": lim,
    }
    with open("optimo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print("RESUMEN DEL MODELO OPTIMO")
    print(f"  Arquitectura : {mejor_arch}")
    print(f"  Learning rate: {mejor_lr}")
    print(f"  Beta (KL)    : {mejor_beta}")
    print(f"  Converge en  : {res['epochs_converged']} epocas")
    print(f"  BCE recon    : {meta['loss_recon']:.3f}  |  KL: {meta['loss_kl']:.2f}")
    print(f"  Error recon medio por kanji: {meta['recon_error_medio']:.3f}")
    if beta_colapso is not None:
        print(f"  Caso de colapso documentado con beta={beta_colapso}")
    print("=" * 70)
    print(f"[TIME] {time.time() - t0:.1f}s  ->  ahora corre:  python3 graficos.py")


if __name__ == "__main__":
    main()
