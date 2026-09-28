"""Раскрытие слепой оценки и расчёт метрик. Запускать ПОСЛЕ заполнения blind_eval.xlsx.

Запуск:  python analyze.py
Результат:  report.md, results_unblinded.xlsx
"""
import random
import shutil
import statistics
import tempfile
from math import comb
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
MDE = 0.5            # минимальный интересный прирост Completeness, баллов
ALPHA = 0.05
LATENCY_LIMIT = 1.5  # медиана латентности B не выше 1.5 x A
N_PERM = 20_000


def paired_permutation_p(diffs: list[float]) -> float:
    """Двусторонний парный перестановочный тест: случайно меняем знаки разниц."""
    rng = random.Random(0)
    observed = abs(statistics.mean(diffs))
    hits = sum(
        abs(statistics.mean(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12
        for _ in range(N_PERM)
    )
    return (hits + 1) / (N_PERM + 1)


def bootstrap_ci(diffs: list[float], level: float = 0.95) -> tuple[float, float]:
    rng = random.Random(1)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(N_PERM))
    lo = means[int((1 - level) / 2 * N_PERM)]
    hi = means[int((1 + level) / 2 * N_PERM) - 1]
    return lo, hi


def sign_test_p(wins: int, losses: int) -> float:
    """Двусторонний биномиальный тест побед B против A (ничьи исключены)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def unblind() -> pd.DataFrame:
    # Excel держит открытый файл заблокированным — читаем сохранённую версию через временную копию
    with tempfile.TemporaryDirectory() as tmp:
        copy = shutil.copy(ROOT / "blind_eval.xlsx", Path(tmp) / "blind_eval.xlsx")
        blind = pd.read_excel(copy, sheet_name="Оценка")
    key = pd.read_csv(ROOT / "_key" / "answer_key.csv")
    dataset = pd.read_csv(ROOT / "dataset.csv")
    raw = pd.read_csv(ROOT / "results_raw.csv")

    required = ["Completeness X", "Completeness Y", "Галлюцинация X", "Галлюцинация Y", "Лучше"]
    missing = blind[blind[required].isna().any(axis=1)]["№"].tolist()
    if missing:
        raise SystemExit(f"Не заполнены оценки в строках №: {missing}")

    df = blind.merge(key, on="№")
    rows = []
    for _, r in df.iterrows():
        side = {r["X"]: "X", r["Y"]: "Y"}  # версия -> колонка
        pref = str(r["Лучше"]).strip().upper()
        winner = "=" if pref == "=" else (r[pref] if pref in ("X", "Y") else None)
        if winner is None:
            raise SystemExit(f"Строка №{r['№']}: в «Лучше» должно быть X, Y или =")
        rows.append({
            "id": r["id"],
            "compl_A": int(r[f"Completeness {side['A']}"]),
            "compl_B": int(r[f"Completeness {side['B']}"]),
            "hall_A": int(r[f"Галлюцинация {side['A']}"]),
            "hall_B": int(r[f"Галлюцинация {side['B']}"]),
            "winner": winner,
            "comment": r["Комментарий"] if pd.notna(r["Комментарий"]) else "",
        })
    res = pd.DataFrame(rows).merge(dataset[["id", "category", "query"]], on="id")
    lat = raw.pivot(index="id", columns="version", values="latency_s").add_prefix("latency_")
    tok = raw.pivot(index="id", columns="version", values="completion_tokens").add_prefix("tokens_")
    return res.merge(lat, on="id").merge(tok, on="id").sort_values("id")


def main() -> None:
    res = unblind()
    n = len(res)
    diffs = (res["compl_B"] - res["compl_A"]).tolist()

    mean_a, mean_b = res["compl_A"].mean(), res["compl_B"].mean()
    delta = mean_b - mean_a
    p_compl = paired_permutation_p(diffs)
    ci_lo, ci_hi = bootstrap_ci(diffs)

    hall_a, hall_b = res["hall_A"].mean(), res["hall_B"].mean()
    lat_a, lat_b = res["latency_A"].median(), res["latency_B"].median()
    tok_a, tok_b = res["tokens_A"].mean(), res["tokens_B"].mean()

    wins_b = (res["winner"] == "B").sum()
    wins_a = (res["winner"] == "A").sum()
    ties = (res["winner"] == "=").sum()
    winrate_b = (wins_b + 0.5 * ties) / n
    p_sign = sign_test_p(wins_b, wins_a)

    checks = [
        ("Δ Completeness ≥ MDE (+0,5)", delta >= MDE, f"{delta:+.2f}"),
        (f"Значимость p < {ALPHA}", p_compl < ALPHA, f"p = {p_compl:.4f}"),
        ("Guard: галлюцинации B ≤ A", hall_b <= hall_a, f"A {hall_a:.0%} → B {hall_b:.0%}"),
        (f"Guard: медиана латентности B ≤ {LATENCY_LIMIT}×A", lat_b <= LATENCY_LIMIT * lat_a,
         f"A {lat_a:.2f}s → B {lat_b:.2f}s (×{lat_b / lat_a:.2f})"),
    ]
    accepted = all(ok for _, ok, _ in checks)

    by_cat = res.groupby("category").agg(
        n=("id", "size"), compl_A=("compl_A", "mean"), compl_B=("compl_B", "mean"),
        hall_A=("hall_A", "sum"), hall_B=("hall_B", "sum"),
    ).reset_index()

    hall_rows = res[(res["hall_A"] == 1) | (res["hall_B"] == 1)]

    lines = [
        "# Результаты A/B-теста: бот «Объясняю термины ИИ»",
        "",
        f"Запросов: **{n}**. Оценка: ручная, слепая (X/Y). План и критерии: [plan.md](plan.md).",
        "",
        "## Сводка метрик",
        "",
        "| Метрика | A (контроль) | B (структура) | Разница |",
        "|---|---|---|---|",
        f"| **Completeness (1–5), Primary** | {mean_a:.2f} | {mean_b:.2f} | {delta:+.2f} (95% ДИ {ci_lo:+.2f}…{ci_hi:+.2f}), p = {p_compl:.4f} |",
        f"| **Галлюцинации, Guard** | {hall_a:.0%} ({res['hall_A'].sum()}/{n}) | {hall_b:.0%} ({res['hall_B'].sum()}/{n}) | {(hall_b - hall_a) * 100:+.0f} п.п. |",
        f"| Латентность, медиана | {lat_a:.2f} с | {lat_b:.2f} с | ×{lat_b / lat_a:.2f} |",
        f"| Длина ответа, токенов (среднее) | {tok_a:.0f} | {tok_b:.0f} | ×{tok_b / tok_a:.2f} |",
        f"| Win-rate B (ничья = 0,5) | — | {winrate_b:.0%} | B {wins_b} / A {wins_a} / = {ties}, sign-test p = {p_sign:.4f} |",
        "",
        "## Проверка критериев успеха",
        "",
        "| Критерий | Результат | Значение |",
        "|---|---|---|",
        *[f"| {name} | {'✅' if ok else '❌'} | {val} |" for name, ok, val in checks],
        "",
        f"**Решение: {'принимаем B в продакшен' if accepted else 'B в продакшен не принимаем'}.**",
        "",
        "## По категориям запросов",
        "",
        "| Категория | N | Completeness A | Completeness B | Галлюц. A | Галлюц. B |",
        "|---|---|---|---|---|---|",
        *[f"| {r.category} | {r.n} | {r.compl_A:.2f} | {r.compl_B:.2f} | {r.hall_A} | {r.hall_B} |" for r in by_cat.itertuples()],
        "",
        "## Ответы с галлюцинациями",
        "",
        *([f"- #{r.id} «{r.query}» — A: {r.hall_A}, B: {r.hall_B}{f' — {r.comment}' if r.comment else ''}"
           for r in hall_rows.itertuples()] or ["- Нет"]),
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    res.to_excel(ROOT / "results_unblinded.xlsx", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
