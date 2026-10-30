"""
Generator dataset sintetis: AI Task Prioritizer (Student Edition)
=================================================================

Kenapa sintetis?
- Kita TAHU aturan asli yang dipakai untuk memberi label (fungsi `true_priority_score`).
- Nanti (Bab 8) kita bisa mengecek: apakah Random Forest + SHAP berhasil "menemukan"
  aturan ini? Itu bukti dari luar diri kita, bukan sekadar angka akurasi.

Cara pakai:
    python generate_dataset.py
Menghasilkan:
    tugas_kuliah_dataset_raw.csv   (data MENTAH, sengaja ada kotoran untuk latihan cleaning)

Semua acakan dikunci dengan SEED supaya hasilnya selalu sama (reproducible).
"""

import numpy as np
import pandas as pd

SEED = 42
N_TASKS = 200
HOURS_PER_DAY = 3.0  # ASUMSI: jam efektif per hari yang bisa dipakai untuk SATU tugas

rng = np.random.default_rng(SEED)

COURSES = ["Alin", "Statprob", "PBP", "SDA", "Matdis 2", "Basdat", "OS", "Jarkom"]

CATEGORIES = ["Tugas Kuliah", "Praktikum", "Kuis/Ujian", "Organisasi", "Pribadi"]
CATEGORY_PROBS = [0.30, 0.20, 0.20, 0.15, 0.15]


# ---------------------------------------------------------------------------
# 1. Membuat ciri-ciri (feature) tiap tugas, realistis per kategori
# ---------------------------------------------------------------------------
def make_task(i):
    cat = rng.choice(CATEGORIES, p=CATEGORY_PROBS)
    course = rng.choice(COURSES)

    if cat == "Tugas Kuliah":
        name = f"{rng.choice(['Tugas', 'PR', 'Tugas Kelompok', 'Paper'])} {course}"
        deadline = int(rng.integers(1, 22))
        hours = rng.uniform(1, 10)
        impact = int(rng.choice([2, 3, 4], p=[0.3, 0.45, 0.25]))
        deps = int(rng.choice([0, 1, 2, 3], p=[0.55, 0.2, 0.15, 0.1]))
    elif cat == "Praktikum":
        name = f"{rng.choice(['Laporan Praktikum', 'Lab', 'Tutorial'])} {course}"
        deadline = int(rng.integers(1, 10))
        hours = rng.uniform(1.5, 6)
        impact = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.2]))
        deps = int(rng.choice([0, 1], p=[0.8, 0.2]))
    elif cat == "Kuis/Ujian":
        kind = rng.choice(["Kuis", "UTS", "UAS"], p=[0.6, 0.2, 0.2])
        name = f"Belajar {kind} {course}"
        deadline = int(rng.integers(1, 22))
        if kind == "Kuis":
            hours, impact = rng.uniform(1, 4), int(rng.choice([2, 3], p=[0.4, 0.6]))
        else:
            hours, impact = rng.uniform(4, 14), 5
        deps = 0
    elif cat == "Organisasi":
        name = rng.choice(["Proposal acara", "Rapat divisi", "Desain poster", "Notulensi rapat",
                           "LPJ kegiatan", "Survei lokasi", "Rekap absensi"])
        deadline = int(rng.integers(1, 15))
        hours = rng.uniform(0.5, 5)
        impact = int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
        deps = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.1]))
    else:  # Pribadi
        name = rng.choice(["Cuci baju", "Update CV", "Belajar bahasa Inggris", "Bayar kos",
                           "Beres-beres kamar", "Olahraga", "Belanja bulanan"])
        deadline = int(rng.integers(1, 31))
        hours = rng.uniform(0.5, 3)
        impact = int(rng.choice([1, 2], p=[0.6, 0.4]))
        deps = 0

    hours = round(hours * 2) / 2  # bulatkan ke kelipatan 0.5 jam
    hours = max(hours, 0.5)

    # complexity: tugas yang lama cenderung lebih sulit (dengan sedikit acak)
    if hours >= 7:
        complexity = rng.choice(["Medium", "High"], p=[0.3, 0.7])
    elif hours >= 3:
        complexity = rng.choice(["Low", "Medium", "High"], p=[0.2, 0.6, 0.2])
    else:
        complexity = rng.choice(["Low", "Medium"], p=[0.75, 0.25])

    # urgency: PERASAAN "mendesak" manusia, terutama dipicu dekatnya deadline (+ noise)
    if deadline <= 1:
        urgency = 5
    elif deadline <= 3:
        urgency = 4
    elif deadline <= 7:
        urgency = 3
    elif deadline <= 14:
        urgency = 2
    else:
        urgency = 1
    if rng.random() < 0.30:  # 30% orang menilai beda 1 tingkat
        urgency += int(rng.choice([-1, 1]))
    urgency = int(np.clip(urgency, 1, 5))

    return {
        "task_id": f"TASK-{i:03d}",
        "task_name": name,
        "task_category": cat,
        "deadline_days": deadline,
        "urgency": urgency,
        "impact_score": impact,
        "estimated_hours": hours,
        "complexity": complexity,
        "dependency_count": deps,
    }


# ---------------------------------------------------------------------------
# 2. ATURAN TERSEMBUNYI (kunci jawaban). Model TIDAK pernah melihat fungsi ini.
# ---------------------------------------------------------------------------
def true_priority_score(row):
    """
    Skor 0..1. Semakin besar, semakin prioritas.

    pressure = jam_dibutuhkan / jam_tersedia
             = estimated_hours / (deadline_days * HOURS_PER_DAY)
      > 1 artinya waktunya sudah tidak cukup kalau baru mulai sekarang.
    """
    pressure = row["estimated_hours"] / (row["deadline_days"] * HOURS_PER_DAY)
    pressure_n = min(pressure, 1.5) / 1.5          # 0..1
    impact_n = (row["impact_score"] - 1) / 4        # 0..1
    urgency_n = (row["urgency"] - 1) / 4            # 0..1
    deps_n = min(row["dependency_count"], 4) / 4    # 0..1

    return 0.40 * pressure_n + 0.30 * impact_n + 0.15 * urgency_n + 0.15 * deps_n


LABELS = ["Low", "Medium", "High", "Critical"]
THRESHOLDS = [0.21, 0.32, 0.44]  # batas skor antar kelas (dikalibrasi ~ Low 35%, Med 30%, High 22%, Crit 13%)


def score_to_label(score):
    idx = int(np.searchsorted(THRESHOLDS, score, side="right"))
    return LABELS[idx]


# ---------------------------------------------------------------------------
# 3. Rakit dataset + noise label (manusia tidak 100% konsisten)
# ---------------------------------------------------------------------------
def build():
    df = pd.DataFrame([make_task(i + 1) for i in range(N_TASKS)])
    score = df.apply(true_priority_score, axis=1)
    labels = score.apply(score_to_label)

    # 8% label bergeser 1 tingkat: meniru ketidakkonsistenan manusia saat melabeli
    flip = rng.random(N_TASKS) < 0.08
    for i in np.where(flip)[0]:
        k = LABELS.index(labels[i]) + int(rng.choice([-1, 1]))
        labels[i] = LABELS[int(np.clip(k, 0, 3))]
    df["priority_level"] = labels

    # ---- 4. Sengaja "mengotori" data mentah untuk latihan Bab 2 ----
    raw = df.copy()
    raw.loc[rng.choice(N_TASKS, 5, replace=False), "estimated_hours"] = np.nan
    raw.loc[rng.choice(N_TASKS, 3, replace=False), "complexity"] = np.nan
    # salah ketik: kategori ASLI-nya tetap, hanya penulisannya yang berantakan
    # (huruf kecil/besar tidak konsisten, spasi berlebih di depan/belakang)
    typo_style = [str.lower, lambda s: s + " ", lambda s: " " + s, str.upper]
    typo_rows = rng.choice(N_TASKS, 4, replace=False)
    for r, style in zip(typo_rows, typo_style):
        raw.loc[r, "task_category"] = style(raw.loc[r, "task_category"])
    outlier_rows = rng.choice(N_TASKS, 2, replace=False)
    raw.loc[outlier_rows[0], "estimated_hours"] = 80.0  # salah ketik "8.0"
    raw.loc[outlier_rows[1], "urgency"] = 7             # di luar skala 1-5
    dup = raw.loc[rng.choice(N_TASKS, 4, replace=False)]
    raw = pd.concat([raw, dup], ignore_index=True)
    raw = raw.sample(frac=1, random_state=SEED).reset_index(drop=True)  # acak urutan

    return df, raw


if __name__ == "__main__":
    clean_truth, raw = build()
    raw.to_csv("tugas_kuliah_dataset_raw.csv", index=False)
    print("Shape mentah:", raw.shape)
    print(clean_truth["priority_level"].value_counts(normalize=True).round(3) * 100)
