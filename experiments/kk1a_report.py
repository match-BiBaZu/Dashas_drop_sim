"""Audit and render the local Kk1a investigation, without catalogue writes."""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from dashas_drop_sim.catalog_batch import engine_fingerprint
from dashas_drop_sim.runner import wilson_interval


ROOT = Path("results/kk1a_study")
ORIGINAL = Path("results/catalog39/Kk1a/run_20261004T161351_282540Z_42")
LABELS = {
    "baseline": "Bisherige Einstellung", "wall_h050": "Wandimpulse: Höhe 0,5 mm",
    "mu_wall005": "Wandreibung 0,05", "early300": "Künstliche Impulse 300 mm/s, bis 3 s",
    "half_timestep": "Zeitschritt 0,5 ms",
}


def main():
    manifest = json.loads((ORIGINAL / "manifest.json").read_text())
    original_rows = [json.loads(line) for line in (ORIGINAL / "trials.jsonl").read_text().splitlines()]
    original_by_trial = {row["trial"]: row for row in original_rows}
    entries = []
    hashes = {}
    original_seed_reproductions = 0
    zero_control_verified = False
    for summary_path in sorted(ROOT.glob("*/summary.json")):
        summaries = json.loads(summary_path.read_text())
        baselines = {}
        for summary in summaries:
            case_path = summary_path.parent / (summary["case"] + ".json")
            data = json.loads(case_path.read_text())
            config, rows = data["config"], data["trials"]
            assert config["catalog_repo"] is None and config["catalog_push"] is False
            assert config["workers"] == 8
            assert len(rows) == summary["trials"] == config["trials"]
            assert [row["trial"] for row in rows] == list(range(len(rows)))
            assert all(row["status"] not in {"invalid", "cancelled"} for row in rows)
            assert dict(Counter(row["status"] for row in rows)) == summary["status_counts"]
            assert sum(summary["axis_status_counts"].values()) == len(rows)
            assert all(np.isfinite(row["final_quat_xyzw"]).all() and np.isfinite(row["final_pos_chute_mm"]).all() for row in rows)
            if "provenance" in data:
                assert data["provenance"]["engine_fingerprint"] == engine_fingerprint()
                assert data["provenance"]["source_sha256"] == manifest["mesh_source_sha256"]
            if summary["case"] == "baseline":
                baselines[config["seed"]] = rows
                if config["seed"] == 42:
                    for row in rows:
                        old = original_by_trial[row["trial"]]
                        assert row["status"] == old["status"] and row["seed"] == old["seed"]
                        assert row["final_quat_xyzw"] == old["final_quat_xyzw"]
                        assert row["final_pos_chute_mm"] == old["final_pos_chute_mm"]
                    original_seed_reproductions += len(rows)
            elif config["seed"] in baselines:
                assert all(row["seed"] == ref["seed"] and row["initial_quat_xyzw"] == ref["initial_quat_xyzw"]
                           for row, ref in zip(rows, baselines[config["seed"]]))
                if summary["case"] == "patch_zero":
                    assert rows == baselines[config["seed"]]
                    zero_control_verified = True
            counts = summary["axis_status_counts"]
            longitudinal = sum(n for key, n in counts.items() if key.startswith("longitudinal:"))
            transverse = sum(n for key, n in counts.items() if key.startswith("transverse:"))
            entry = {"group": summary_path.parent.name, "case": summary["case"], "seed": config["seed"],
                     "trials": len(rows), "settled_percent": summary["settled_percent"],
                     "settled_ci95_percent": summary["settled_ci95_percent"],
                     "longitudinal_percent": 100 * longitudinal / len(rows),
                     "transverse_percent": 100 * transverse / len(rows),
                     "transverse_ci95_percent": [100 * x for x in wilson_interval(transverse, len(rows))],
                     "diagnostic_positive_energy_j": sum(row.get("diagnostic_positive_energy_j", 0) for row in rows),
                     "axis_status_counts": counts, "parameters": summary["parameters"],
                     "raw_file": str(case_path.relative_to(ROOT)).replace("\\", "/")}
            entries.append(entry)
            hashes[entry["raw_file"]] = hashlib.sha256(case_path.read_bytes()).hexdigest()
    axes = Rotation.from_quat([row["final_quat_xyzw"] for row in original_rows]).apply([0, 0, 1])
    original_counts = Counter(("longitudinal" if abs(axis[0]) >= 0.9 else "transverse" if abs(axis[0]) <= 0.3 else "oblique")
                              + ":" + row["status"] for axis, row in zip(axes, original_rows))
    audit = {"original_trials": len(original_rows), "original_axis_status_counts": dict(original_counts),
             "study_trials": sum(row["trials"] for row in entries), "case_runs": len(entries),
             "unique_cases": len({row["case"] for row in entries}),
             "exact_original_seed_reproductions": original_seed_reproductions,
             "engine_fingerprint": engine_fingerprint(), "mesh_sha256": manifest["mesh_source_sha256"],
             "mass_kg": manifest["mass_kg"], "all_complete_valid_and_paired": True,
             "friction_patch_zero_control_exact": zero_control_verified,
             "diagnostic_axis_definition": "abs(axis.X)>=0.9 longitudinal; <=0.3 transverse; otherwise oblique",
             "results": entries, "raw_file_sha256": hashes}
    (ROOT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    fields = ["group", "case", "seed", "trials", "settled_percent", "longitudinal_percent", "transverse_percent", "diagnostic_positive_energy_j", "raw_file"]
    with (ROOT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    confirmation = [row for row in entries if row["group"] == "confirmation"]
    if confirmation:
        categories = [("longitudinal:settled", "Längs, eingependelt", "#18856c"),
                      ("longitudinal:unsettled", "Längs, nicht eingependelt", "#f2ad44"),
                      ("transverse:settled", "Quer, eingependelt", "#4099cd"),
                      ("transverse:unsettled", "Quer, nicht eingependelt", "#ce5867")]
        fig, ax = plt.subplots(figsize=(12, 5.8), layout="constrained")
        starts = np.zeros(len(confirmation))
        positions = np.arange(len(confirmation))
        for key, label, color in categories:
            values = np.array([100 * row["axis_status_counts"].get(key, 0) / row["trials"] for row in confirmation])
            ax.barh(positions, values, left=starts, label=label, color=color, height=0.65)
            starts += values
        ax.barh(positions, 100 - starts, left=starts, label="Andere Ergebnisstatus", color="#8d929b", height=0.65)
        ax.set_yticks(positions, [LABELS.get(row["case"], row["case"]) for row in confirmation])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Anteil an allen 1.000 Fallversuchen [%]")
        ax.set_title("Kk1a: Endlage und Einpendeln — unabhängiger Seed 43")
        ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.14), ncol=2, frameon=False)
        fig.savefig(ROOT / "confirmation.png", dpi=160)
        fig.savefig(ROOT / "confirmation.svg")
        plt.close(fig)
    dose_rows = [row for row in entries if row["group"] == "thresholds" and (row["case"] == "baseline" or row["case"].startswith("early"))]
    if dose_rows:
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        xs = np.array([row["parameters"].get("diagnostic_dv_mm_s", 0) for row in dose_rows])
        for value, ci, label, color in (("settled_percent", "settled_ci95_percent", "Eingependelt", "#18856c"),
                                      ("transverse_percent", "transverse_ci95_percent", "Achse quer", "#ce5867")):
            ys = np.array([row[value] for row in dose_rows])
            bounds = np.array([row[ci] for row in dose_rows]).T
            ax.errorbar(xs, ys, yerr=np.vstack([ys - bounds[0], bounds[1] - ys]), color=color, marker="o", label=label, capsize=4)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Impuls J / Bauteilmasse als äquivalentes Δv [mm/s]")
        ax.set_ylabel("Anteil aller 128 Versuche [%]")
        ax.set_title("Künstliche Wandimpulse bis 3 s — Diagnose, keine Kratzerkalibrierung")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(ROOT / "pulse_threshold.png", dpi=160)
        fig.savefig(ROOT / "pulse_threshold.svg")
        plt.close(fig)
    text_rows = "".join(f"<tr data-group='{html.escape(row['group'])}'><td>{html.escape(row['group'])}</td><td>{html.escape(row['case'])}</td><td>{row['trials']}</td><td>{row['settled_percent']:.1f}</td><td>{row['longitudinal_percent']:.1f}</td><td>{row['transverse_percent']:.1f}</td><td>{row['diagnostic_positive_energy_j']/row['trials']*1e6:.1f}</td><td><a href='{html.escape(row['raw_file'])}'>JSON</a></td></tr>" for row in entries)
    groups = sorted({row["group"] for row in entries})
    options = "".join(f"<option>{html.escape(group)}</option>" for group in groups)
    report = f"""<!doctype html><html lang="de"><meta charset="utf-8"><title>Kk1a Kontaktstudie</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;background:#f5f7fa;color:#203044}}h1{{font-size:30px}}p{{line-height:1.6}}.card{{background:white;padding:22px;border-radius:12px;margin:20px 0}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}th{{position:sticky;top:0;background:#e5edf4;cursor:pointer}}img.chart{{width:100%}}.poses{{display:flex;gap:12px}}.poses img{{width:100%}}figure{{margin:0;flex:1}}figcaption{{font-size:14px}}select{{padding:8px}}</style>
<h1>Kk1a: Kontaktstörungen und Endlagen</h1>
<p>{audit['case_runs']} lokale Vergleichsläufe, {audit['study_trials']:,} neue Fallversuche. 8 Prozesse, gekoppelte Abwürfe je Vergleich. Original: 1.000 Versuche, 267 eingependelt; 481 quer und nicht eingependelt, 252 längs und nicht eingependelt, 266 längs eingependelt, 1 quer eingependelt.</p>
<div class="card"><p><b>Ergebnis:</b> Die geprüften kleinen Kontaktstörungen senken den Queranteil kaum. Kleinere Wandreibung verringert den Anteil „unsettled“, ohne die Querlagen entsprechend in Längslagen zu überführen. Erst die sehr großen künstlichen Diagnoseimpulse erzwingen häufig das Kippen. Dabei erzeugen sie Energie an der ruhenden Wand; diese Reihe eignet sich deshalb nicht als physikalisches Kratzermodell und wird nicht als Standard übernommen.</p>
<p>Die Klassifizierung reagiert auf Reibung, Kollisionskörper und Zeitschritt. Eine Bewertung braucht den tatsächlich beobachteten Bewegungszustand und die Endorientierung. Ein neuer Kataloglauf wurde nicht veröffentlicht; seine Physik und Defaults wurden nicht geändert.</p></div>
<div class="card"><img class="chart" src="confirmation.png"><p>Je Einstellung 1.000 identische neue Abwürfe mit Seed 43. „Quer“ bezeichnet die Achsrichtung, einschließlich stehender Querlagen. Die künstlichen Impulse sind nur eine Schwellenmessung.</p></div>
<div class="card"><img class="chart" src="pulse_threshold.png"><p>Die Fehlerbalken sind Wilson-Intervalle (95 %). Jeder künstliche Impuls greift an einem belasteten Wandkontakt an; J = m · Δv. „Δv“ ist hier eine Impulseinheit für das gesamte Bauteil und keine vorgegebene Änderung der Bandgeschwindigkeit. Impulse nach 3 s sind ausgeschaltet, um anschließendes Einpendeln zu beobachten.</p></div>
<div class="card poses"><figure><img src="images/transverse_unsettled.png"><figcaption>Quer: Umfang an der Wand, Ende auf dem Band; Bewegung bleibt bestehen.</figcaption></figure><figure><img src="images/longitudinal_unsettled.png"><figcaption>Längs, dennoch als „unsettled“ eingestuft.</figcaption></figure><figure><img src="images/longitudinal_settled.png"><figcaption>Längs und eingependelt.</figcaption></figure></div>
<div class="card"><p><b>Weitere Messungen:</b> Grobe reale Rollhäufigkeit nach derselben Strecke, Wand-/Bandreibung mit dem echten Bauteil, Zeitverlauf der ±3-mm/s-Bandschwankung, Film eines typischen Kippvorgangs. Damit lässt sich unterscheiden, ob lokale Hindernisse, nicht ebene Bauteil-Endflächen oder der Antrieb das fehlende Kippmoment erzeugen.</p>
<p>Die vorhandene Unebenheitsfunktion erzeugt Ereignisse nach <em>relativem Schlupf</em>. Ein rollender Kontakt mit sehr wenig Schlupf kann jedoch neue Oberflächenstellen überfahren. Ein zusätzlicher Test mit örtlich wechselnder Kontaktreibung erfasst die Bewegung des belasteten Kontaktbereichs. Auch damit änderte sich der Queranteil kaum. Der Kontrolllauf mit Nullvariation reproduzierte sämtliche Versuchsdaten exakt.</p>
<p>Der Zeitschrittvergleich (Seed 43, je 1.000 Abwürfe) änderte den eingependelten Anteil von 27,8 % bei 1 ms auf 37,9 % bei 0,5 ms; der Queranteil blieb bei 48,3 % beziehungsweise 47,5 %. Ein weiterer Kontrolllauf bei 0,25 ms (Seed 42, 128 Abwürfe) ergab 41,4 % eingependelt und 49,2 % quer. Ein „settled“-Grenzwert ist damit für diese Bewegung noch nicht numerisch abgesichert. Diese Werte sind keine nachgewiesene Übereinstimmung mit realen Posenhäufigkeiten.</p></div>
<div class="card"><label>Vergleich: <select id="group"><option value="">Alle</option>{options}</select></label><p>Alle Prozentwerte beziehen sich auf alle abgeschlossenen Versuche. Achse längs: |X| ≥ 0,9; quer: |X| ≤ 0,3. Diese Diagnosegruppen ersetzen keine Posen-IDs oder Erkennungstoleranzen.</p><table id="table"><thead><tr><th>Reihe</th><th>Einstellung</th><th>N</th><th>Settled %</th><th>Längs %</th><th>Quer %</th><th>Zusätzliche positive Diagnoseenergie pro Versuch [µJ]</th><th>Daten</th></tr></thead><tbody>{text_rows}</tbody></table></div>
<p>Nachweise: <a href="audit.json">Prüfung, Hashes und alle Zusammenfassungen</a>, <a href="comparison.csv">CSV</a>. Grundlagen: <a href="https://mujoco.readthedocs.io/en/3.14.0/XMLreference.html#contact-pair">MuJoCo-Kontakte</a>, <a href="https://mujoco.readthedocs.io/en/3.14.0/modeling.html#solver-parameters">Kontaktmodell und Solver</a>.</p>
<script>const select=document.querySelector('#group');select.onchange=()=>document.querySelectorAll('tbody tr').forEach(r=>r.hidden=select.value&&r.dataset.group!==select.value);document.querySelectorAll('th').forEach((h,i)=>h.onclick=()=>{{const body=document.querySelector('tbody'),rows=[...body.rows],descending=h.dataset.direction!=='desc';h.dataset.direction=descending?'desc':'asc';rows.sort((a,b)=>{{const av=a.cells[i].textContent,bv=b.cells[i].textContent;return (descending?-1:1)*(isNaN(Number(av))?av.localeCompare(bv):Number(av)-Number(bv))}});rows.forEach(r=>body.append(r))}});</script></html>"""
    (ROOT / "report.html").write_text(report, encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("study_trials", "case_runs", "unique_cases", "exact_original_seed_reproductions", "all_complete_valid_and_paired")}))


if __name__ == "__main__":
    main()
