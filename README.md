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

Unter Windows startet `DropSimulationGUI.cmd` die GUI direkt. Ein Doppelklick auf
`WindowsLaunchers\Verknuepfungen-installieren.cmd` erstellt eigene Verknüpfungen
auf dem Desktop und unter **Startmenü > BiBaZu > BiBaZu Drop Simulation**.
Den Startmenü-Eintrag kann man über das Kontextmenü an „Start“ anheften.
Die Simulation verwendet keine Kameras, Lampen, SPS oder anderen BiBaZu-Geräte
und kann parallel zu den anderen GUIs laufen. Bei parallelen Simulationen
teilen sich die Prozesse lediglich CPU und Arbeitsspeicher.

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

Eine ganze Serie mit durchgehend geöffneter MuJoCo-Ansicht:

```powershell
uv run dashas-drop-sim watch --config config.json
```

In der GUI startet **Simulation starten** die Serie ohne 3D-Ansicht.
**Falltests sichtbar starten** zeigt alle eingestellten Abwürfe nacheinander
im selben MuJoCo-Fenster; der Reiter **Letzter Versuch** zeigt das jüngste
Ergebnis. Ein geschlossenes MuJoCo-Fenster oder **Abbrechen** beendet den Lauf
mit den bis dahin gespeicherten Versuchen. Die anschließenden Störversuche
laufen ohne 3D-Ansicht. Sichtbare Abwürfe laufen einzeln und ungefähr in
Echtzeit; die Einstellung **Parallele Prozesse** gilt dabei nur für die
Störversuche.
Im Dropdown **Werkstückkatalog** stehen 39 STL-Modelle aus
[`Werkstücke_STL_grob`](https://github.com/match-BiBaZu/bibazu_geometry_to_pose/tree/main/Werkst%C3%BCcke_STL_grob)
als mitgelieferter, offline nutzbarer Stand `02d3fbcf` bereit. **Durchsuchen**
bleibt für eigene STL- und STEP-Dateien verfügbar. Ist die lokale
`bibazu_geometry_to_pose`-Arbeitskopie vorhanden, wird eine eindeutig passende
Roadmap beim Auswählen automatisch eingetragen. In der Häufigkeitstabelle
öffnet **Bild** eine schematische 3D-Ansicht der Pose; beobachtete Endlagen und
unbeobachtete Katalogorientierungen sind beschriftet.

Für mehrere Werkstücke gibt es den Bereich **Mehrere Werkstücke**. **Mehrere
Dateien laden** erlaubt die gemeinsame Auswahl mehrerer STL- oder STEP-Dateien;
**Aus Katalog laden** erlaubt Mehrfachauswahl oder **Alle auswählen**. Ein bereits
eingerichtetes Modell samt Roadmap lässt sich mit **Aktuelles Werkstück
hinzufügen** in die Liste übernehmen. Jeder Eintrag hat eine eigene optionale
Roadmap. Eindeutig passende Roadmaps werden automatisch eingetragen und können
in der Liste geändert oder geleert werden.

**Werkstück-Batch starten** verarbeitet die gesamte Liste nacheinander ohne
weitere Eingaben. Die aktuellen Simulationsparameter, Versuchszahl und Anzahl
paralleler Prozesse gelten für jedes Werkstück und werden beim Start gespeichert.
Die Parallelisierung findet innerhalb des jeweiligen Werkstücks statt. Ein
fehlgeschlagenes Werkstück wird markiert; der Batch läuft mit dem nächsten
weiter. **Abbrechen** beendet den aktuellen Lauf und startet keine weiteren
Werkstücke.

Ergebnisse liegen unter `results/batch_<Zeitstempel>/<Nummer>_<Werkstück>/run_*`.
`batch.json` im Batch-Ordner enthält Einstellungen, Status und Ergebnisordner
aller Einträge. Nach Abschluss öffnet **Ergebnisse** in der jeweiligen Tabellenzeile
die Häufigkeits- und Stabilitätstabellen dieses Werkstücks. Die Auswahl setzt
auch dessen Roadmap für die anschließende Neunummerierung der Pose-IDs.

Serien ohne 3D-Ansicht und Störversuche laufen über getrennte MuJoCo-Prozesse auf mehreren
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

Beim normalen Simulationslauf wird die Roadmap nur gelesen. Ihre Felder für
Luftimpuls-Übergänge werden durch Fallhäufigkeiten nicht verändert.

### Pose-Nummern nach Häufigkeit ordnen

Nach einer Serie kann die Schaltfläche **YAML und JSON nach Häufigkeit neu
nummerieren** die IDs einer geladenen, gepaarten Roadmap ändern. Dafür wird
die `summary.json` des Laufs ausgewählt. Die Simulation muss mit genau dieser
Version der YAML- oder JSON-Roadmap gelaufen sein. Pose 0 wird die am häufigsten
beobachtete Roadmap-Pose; weitere Posen folgen absteigend nach Trefferzahl.
Gleichstände werden nach bisheriger ID sortiert, Posen ohne Treffer stehen am
Ende. Unbekannte Lagen und nicht eingependelte Versuche werden nicht in die
Roadmap aufgenommen. Die Einordnung als robust/metastabil bleibt erhalten;
Pose-Listen und Übergänge in beiden Dateien werden auf die neuen IDs umgestellt.

Der Speicherdialog schlägt neue Dateien mit `_frequency_ordered` im Namen vor.
Wer die bisherigen Dateien überschreibt, erhält zuvor Sicherungskopien mit
`.bak.<Zeitstempel>` im selben Ordner. Bereits gespeicherte Simulationsergebnisse
behalten ihre bisherigen IDs; neue Läufe können die neu nummerierte Roadmap
verwenden.

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
