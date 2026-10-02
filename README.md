# Dashas Drop Sim

Wiederholbare Falltests für ein einzelnes Bauteil in der BiBaZu-L-Rutsche. Das
Programm simuliert je Versuch eine gleichverteilt zufällige 3D-Abwurforientierung
und erfasst die Endlage. Anschließend prüft es gefundene Lagen mit definierten
Drehimpulsen. **Auftretenshäufigkeit** und **Pose-Erhalt nach einer Störung** sind
getrennte Ergebnisse.

## Installation

Voraussetzungen: Windows oder ein System mit MuJoCo-Unterstützung, Python 3.12,
[`uv`](https://docs.astral.sh/uv/) und für die Roadmap-Abhängigkeit Zugriff auf
GitHub. Im geklonten Repository:

```powershell
uv sync --python 3.12 --all-extras
uv run dashas-drop-sim gui
```

`--all-extras` installiert die PyQt6-Oberfläche, OpenCascade für STEP-Dateien,
die fest gepinnte Version von `bibazu_geometry_to_pose` für Roadmap-Vergleiche
und pytest. Nach der Installation lassen sich GUI und CLI auch mit
`uv run --no-sync` starten, wenn keine erneute Synchronisierung gewünscht ist.

Für einen Testlauf ohne Oberfläche:

```powershell
uv run dashas-drop-sim run --config config.json
```

Ein einzelner Versuch mit sichtbarer MuJoCo-Ansicht:

```powershell
uv run dashas-drop-sim preview --config config.json --trial 0
```

Die GUI bietet dieselben Eingaben und startet Serien ohne laufende 3D-Ansicht.
Der Knopf **Einzelfall anzeigen** öffnet die Ansicht für einen Fall. Eine Serie
kann über **Abbrechen** nach dem laufenden Versuch beendet werden.
Im Dropdown **Werkstückkatalog** stehen 39 STL-Modelle aus
[`Werkstücke_STL_grob`](https://github.com/match-BiBaZu/bibazu_geometry_to_pose/tree/main/Werkst%C3%BCcke_STL_grob)
als mitgelieferter, offline nutzbarer Stand `02d3fbcf` bereit. **Durchsuchen**
bleibt für eigene STL- und STEP-Dateien verfügbar. Ist die lokale
`bibazu_geometry_to_pose`-Arbeitskopie vorhanden, wird eine eindeutig passende
Roadmap beim Auswählen automatisch eingetragen. In der Häufigkeitstabelle
öffnet **Bild** eine schematische 3D-Ansicht der Pose; beobachtete Endlagen und
unbeobachtete Katalogorientierungen sind beschriftet.

Serien und Störversuche laufen über getrennte MuJoCo-Prozesse auf mehreren
CPU-Kernen. Die GUI bietet dafür **Parallele Prozesse** (Vorgabe: bis zu vier).
Jeder Versuch behält seinen eigenen Seed und Physikzustand; mit `1` läuft alles
nacheinander. Der Gewinn hängt von der verfügbaren CPU-Leistung ab.
Ein Vergleich mit acht Qk1a-Abwürfen auf 1,3 m dauerte hier 26,2 s mit einem
Prozess und 13,2 s mit vier Prozessen. Einzelversuche, Häufigkeiten und
Störresultate waren in beiden Läufen bytegleich.

## Eingaben

Die Bauteildatei muss ein geschlossenes, volumenhaltiges STL- oder STEP-Modell
**in Millimetern** enthalten. STL wird direkt gelesen; STEP wird mit
OpenCascade trianguliert. Masse, Schwerpunkt und Trägheit werden aus dem
geschlossenen Mesh und einer homogenen Dichte berechnet. Konkave Formen werden
für MuJoCo in konvexe Kontaktkörper zerlegt. Die Kennzahlen zur Abweichung der
Kontaktgeometrie stehen in `manifest.json`.

Beispiel für `config.json` (Pfade anpassen):

```json
{
  "mesh_path": "C:/Bauteile/Qk1a.stl",
  "roadmap_path": "C:/Bauteile/Qk1a_roadmap.yaml",
  "output_dir": "C:/Falltests/Ergebnisse",
  "trials": 100,
  "workers": 4,
  "seed": 42,
  "belt_speed_mm_s": 100.0,
  "drop_height_mm": 100.0,
  "lateral_mm": 0.0,
  "density_g_cm3": 1.15,
  "mu_belt": 0.40,
  "mu_wall": 0.20,
  "alpha_deg": 45.0,
  "beta_deg": 0.0,
  "length_mm": 1300.0,
  "timestep_s": 0.001,
  "disturbance_levels_mm": [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
}
```

Relative Eingabe- und Ausgabepfade in der JSON-Konfiguration werden relativ zum
**Ordner der Konfigurationsdatei** aufgelöst.
`roadmap_path` kann weggelassen oder auf `null` gesetzt werden. Bekannte
Roadmap-IDs erfordern das originale STL, mit dem die Roadmap erzeugt wurde;
eine STEP-Triangulierung erzeugt andere Katalog-IDs. Ohne Roadmap bleiben die
gefundenen Lagen als separate, unbekannte Gruppen sichtbar. Die Zuordnung
berücksichtigt erkannte Bauteilsymmetrien und alle Katalogorientierungen einer
Roadmap-Pose. Eine mehrdeutige oder zu weit entfernte Endlage bekommt keine
erzwungene ID.
Wird ohne das optionale `roadmap`-Paket installiert, stehen auch für unbekannte
Gruppen keine erkannten Bauteilsymmetrien zur Verfügung; dies wird im Manifest
als `symmetry_available: false` ausgewiesen.

Die Bereiche der GUI sind 0–200 mm/s Bandgeschwindigkeit, 0–200 mm
Abwurfhöhe und ±100 mm Querposition, gemessen von der Winkelhalbierenden
zwischen Band und Wand. Bei 0 mm/s endet die Beobachtung nach einer festen
Zeit statt an einem Streckenende. Die genannten Dichte- und Reibwerte sind
konfigurierbar; 1,15 g/cm³ ist der Startwert innerhalb der angegebenen
Materialspanne von 1,12–1,18 g/cm³.

## Ergebnisse

Jede Serie erhält einen eigenen Ordner `run_<UTC-Zeitstempel>_<Seed>` unter
`output_dir`:

| Datei | Inhalt |
| --- | --- |
| `config.json` | Tatsächlich verwendete Eingaben |
| `manifest.json` | Datei-Hashes, Softwareversionen und Geometriequalität |
| `trials.jsonl` | Während des Laufs fortlaufend geschrieben; am Ende mit fertiger Pose-Zuordnung |
| `trials.csv` | Jeder Abwurf mit Seed, Endorientierung, Status und fertiger Zuordnung |
| `frequencies.csv` | Anzahl, Anteil und 95-%-Konfidenzintervall je Pose oder Status |
| `disturbances.csv` | Einzelne Störversuche nach Pose, Richtung und Stärke |
| `stability.csv` | Pose-Erhalt je Störstärke als Störkurve |
| `stability_summary.csv` | Normierte Fläche unter der Störkurve und Rangfolge |
| `summary.json` | Maschinenlesbare Zusammenfassung des Laufs |

Ein sichtbarer Einzelfall erhält einen eigenen Laufordner mit `config.json`,
`manifest.json` und `preview.json` statt der Serientabellen.
Die Häufigkeitstabelle enthält zu jeder darstellbaren Pose auch eine
Beispielorientierung für den Bild-Button.

Eine Pose wird nur als Treffer gezählt, wenn ihre Orientierung und
Winkelgeschwindigkeit über ein Zeitfenster ruhig bleiben. Das Teil darf sich
dabei weiter in Bandrichtung bewegen. Nicht eingependelte, feststeckende oder
mehrdeutige Fälle erscheinen gesondert. Die zwölf Störrichtungen pro positiver
Stärke sind reproduzierbar; die Stufe `0 mm` wird als Basislauf nur einmal
simuliert. Die Störstärken sind als äquivalente Hubhöhe in Millimetern
angegeben: Ihre Rotationsenergie beträgt `Masse × 9,81 m/s² × Hubhöhe`. Die
Störkurve beschreibt den beobachteten Pose-Erhalt am Ende und in eingependelten
Zwischenlagen; die normierte Fläche dient
als Rangmaß. Es gibt keinen vorgegebenen Grenzwert für „stabil“.

Die Roadmap wird nur gelesen. Ihre Felder für Luftimpuls-Übergänge werden durch
Fallhäufigkeiten nicht verändert.

## Physikalische Annahmen und Grenzen

- Die virtuelle Rutsche hat 45° Querneigung, 0° Längsneigung und standardmäßig
  1,3 m Beobachtungsstrecke wie die reale Rutsche. Die Länge bleibt einstellbar.
  Häufigkeiten beziehen sich immer auf die konfigurierte Strecke und
  Abwurfverteilung. Ergebnisse alter 3-m-Läufe sind damit nicht direkt
  vergleichbar.
- PE-Band und PTFE-Wand sind jeweils 15 cm von der gemeinsamen Ecke bis zur
  freien Kante dargestellt. Idealisierte Kontaktflächen schließen seitliches
  Herunterfallen aus der Untersuchung aus.
- Die Kontaktoberfläche des PE-Bands bewegt sich mit MuJoCos `surfacevel` in
  Rutschenrichtung; die PTFE-Wand ruht. Ihre Reibwerte werden getrennt gesetzt.
  Die Startwerte `mu_belt = 0.40` und `mu_wall = 0.20` sind **vorläufige
  Simulationsannahmen**, keine aus Shore-Härte oder anderen Materialkennwerten
  abgeleiteten Messwerte. Für belastbare Vorhersagen müssen beide Kontakte
  anhand realer Gleit- und Fallversuche kalibriert werden.
- Die simulierten Teile sind starr, massiv und homogen. Durch die
  Kontaktzerlegung und den Zeitschritt kann sich das Ergebnis ändern. Für
  relevante Varianten sollten Zeitschritt und Kontaktgeometrie verglichen
  werden.

Ein erster Zeitschrittvergleich für Qk1a auf der vorherigen 3-m-Strecke mit zehn gleichen Abwurf-Seeds ergab
bei 1 ms und 0,5 ms in **acht von zehn** Fällen dieselbe Roadmap-ID. Alle 20
Läufe waren eingependelt; die zwei abweichenden Abwürfe gelangten in andere
Rocking-Posen. Die beobachteten Häufigkeiten unterschieden sich je ID um
höchstens einen Fall. Zehn Versuche reichen nicht aus, um Konvergenz der
Posenverteilung zu belegen. Für eine belastbare Rangfolge sind mehr Abwürfe und
eine Wiederholung mit kleinerem Zeitschritt nötig.

Die Häufigkeit beschreibt die gewählten Startbedingungen: eine zufällige
Orientierung auf SO(3) bei festem Abwurfpunkt, Höhe und Querposition. Andere
Zuführungen können andere Häufigkeiten liefern.
