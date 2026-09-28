"""Прогон датасета через промпты A и B + подготовка файла для слепой оценки.

Запуск:  python run_ab.py
Результат:
  results_raw.csv       — все ответы с версиями, латентностью и токенами (не открывать до оценки!)
  blind_eval.xlsx       — перемешанные ответы X/Y для ручной оценки
  _key/answer_key.csv   — ключ X/Y -> A/B (не открывать до оценки!)
  dataset.xlsx          — датасет в читаемом виде (для скриншотов)
"""
import random
import re
import secrets
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from prompts import GEN_PARAMS, MODEL, PROMPTS

ROOT = Path(__file__).parent
HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
SCORE_FILL = PatternFill("solid", fgColor="FFF2CC")

RUBRIC = [
    ("Completeness (1–5) — полнота относительно эталона", ""),
    ("5", "Все ключевые пункты эталона + понятный пример/применение"),
    ("4", "Все ключевые пункты, но без примера или с мелким пропуском"),
    ("3", "Примерно половина ключевых пунктов"),
    ("2", "Поверхностно, главное упущено"),
    ("1", "Не отвечает на вопрос или по сути неверно"),
    ("Несуществующий термин", "5 — честно сказал, что термин неизвестен/не общепринят; 1 — уверенно объяснил выдуманное"),
    ("Вне темы / инъекция", "Оцениваем по ожидаемому поведению из колонки «Эталон»"),
    ("", ""),
    ("Галлюцинация (0/1)", "1 — есть хоть одно выдуманное или неверное утверждение; 0 — нет"),
    ("Лучше (X / Y / =)", "Какой ответ ты бы хотел(а) получить как новичок; = — ничья"),
    ("", ""),
    ("Важно", "Не открывай _key/answer_key.csv и results_raw.csv, пока не закончишь оценку."),
]


def strip_markdown(text: str) -> str:
    """Убираем разметку, чтобы жирные заголовки не выдавали версию промпта."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def generate(client: OpenAI, system: str, query: str) -> dict:
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
        **GEN_PARAMS,
    )
    latency = time.perf_counter() - start
    return {
        "answer": resp.choices[0].message.content,
        "latency_s": round(latency, 3),
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }


def run(dataset: pd.DataFrame) -> pd.DataFrame:
    client = OpenAI()
    rows = []
    for i, item in dataset.iterrows():
        # Чередуем порядок вызовов, чтобы колебания латентности API не били в одну версию
        order = ["A", "B"] if i % 2 == 0 else ["B", "A"]
        for version in order:
            res = generate(client, PROMPTS[version], item["query"])
            rows.append({"id": item["id"], "version": version, **res})
            print(f"#{item['id']:>2} {version}: {res['latency_s']:.2f}s, {res['completion_tokens']} tok")
    return pd.DataFrame(rows)


def build_blind(dataset: pd.DataFrame, raw: pd.DataFrame) -> None:
    rng = random.Random(secrets.randbits(64))  # сид не сохраняется — ключ нельзя восстановить из кода
    answers = raw.pivot(index="id", columns="version", values="answer")

    blind_rows, key_rows = [], []
    ids = list(dataset["id"])
    rng.shuffle(ids)
    for n, qid in enumerate(ids, start=1):
        item = dataset.set_index("id").loc[qid]
        x_is = rng.choice(["A", "B"])
        y_is = "B" if x_is == "A" else "A"
        blind_rows.append({
            "№": n,
            "Запрос": item["query"],
            "Эталон (ключевые пункты)": item["reference"],
            "Ответ X": strip_markdown(answers.loc[qid, x_is]),
            "Ответ Y": strip_markdown(answers.loc[qid, y_is]),
            "Completeness X": None,
            "Completeness Y": None,
            "Галлюцинация X": None,
            "Галлюцинация Y": None,
            "Лучше": None,
            "Комментарий": None,
        })
        key_rows.append({"№": n, "id": qid, "X": x_is, "Y": y_is})

    path = ROOT / "blind_eval.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame(blind_rows).to_excel(w, sheet_name="Оценка", index=False)
        pd.DataFrame(RUBRIC, columns=["Критерий", "Описание"]).to_excel(w, sheet_name="Инструкция", index=False)

    wb = load_workbook(path)
    ws = wb["Оценка"]
    widths = {"A": 5, "B": 30, "C": 40, "D": 60, "E": 60, "F": 14, "G": 14, "H": 14, "I": 14, "J": 9, "K": 30}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in row[5:10]:
            cell.fill = SCORE_FILL
    ws.freeze_panes = "B2"
    last = ws.max_row
    for rng_ref, options in [(f"F2:G{last}", '"1,2,3,4,5"'), (f"H2:I{last}", '"0,1"'), (f"J2:J{last}", '"X,Y,="')]:
        dv = DataValidation(type="list", formula1=options, allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(rng_ref)
    wi = wb["Инструкция"]
    wi.column_dimensions["A"].width = 45
    wi.column_dimensions["B"].width = 90
    for cell in wi[1]:
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
    wb.save(path)

    (ROOT / "_key").mkdir(exist_ok=True)
    pd.DataFrame(key_rows).to_csv(ROOT / "_key" / "answer_key.csv", index=False, encoding="utf-8-sig")


def build_dataset_xlsx(dataset: pd.DataFrame) -> None:
    path = ROOT / "dataset.xlsx"
    out = dataset.rename(columns={"id": "ID", "category": "Категория", "query": "Запрос", "reference": "Эталон (ключевые пункты)"})
    out.to_excel(path, index=False, sheet_name="Датасет")
    wb = load_workbook(path)
    ws = wb.active
    for col, width in {"A": 5, "B": 22, "C": 55, "D": 90}.items():
        ws.column_dimensions[col].width = width
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    wb.save(path)


def main() -> None:
    load_dotenv()
    dataset = pd.read_csv(ROOT / "dataset.csv")
    build_dataset_xlsx(dataset)
    raw = run(dataset)
    raw.to_csv(ROOT / "results_raw.csv", index=False, encoding="utf-8-sig")
    build_blind(dataset, raw)
    print(f"\nГотово: {len(dataset)} запросов x 2 версии. Открывай blind_eval.xlsx и оценивай.")


if __name__ == "__main__":
    main()
