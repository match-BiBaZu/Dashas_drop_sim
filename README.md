# Dashas Drop Sim

Wiederholbare Falltests für ein einzelnes Bauteil in der BiBaZu-L-Rutsche. Das
Programm simuliert je Versuch eine gleichverteilt zufällige 3D-Abwurforientierung
und erfasst die Endlage. Anschließend prüft es gefundene Lagen mit definierten
Drehimpulsen. **Auftretenshäufigkeit** und **Pose-Erhalt nach einer Störung** sind
getrennte Ergebnisse.

## Installation

Voraussetzungen: Windows oder ein System mit MuJoCo-Unterstützung, Python 3.12
und [`uv`](https://docs.astral.sh/uv/). Geometriecode und CAD-Paare
werden mitgeliefert; ein zweites Repository ist nicht erforderlich.
Im geklonten Repository:

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

**Als Standard speichern** speichert die Eingaben für **Fallversuche**,
**Erweiterte Simulationsparameter** und **Unebenheiten und zufällige Kontaktkräfte** einschließlich
der Streubereiche sowie die ausgewählten Datei- und Ordnerpfade. Die GUI lädt
sie beim nächsten Start automatisch. Das Speichern der Parameter funktioniert
auch vor der Auswahl eines Werkstücks. Änderungen werden beim Drücken dieses
Knopfs gespeichert. Unter Windows liegt die Datei unter
`%APPDATA%\BiBaZu\DashasDropSim\gui_defaults.json`; auf anderen Systemen unter
`$XDG_CONFIG_HOME/dashas-drop-sim/gui_defaults.json` beziehungsweise
`~/.config/dashas-drop-sim/gui_defaults.json`.

Die vollständigen aktuellen GUI-Vorgaben sind außerdem im Repository unter
`src/dashas_drop_sim/assets/gui_defaults.json` versioniert: alle 40 Parameter,
einschließlich Dämpfung 0,19, 500 mm Strecke, Einzelimpulsen während der ersten
Sekunde und Ruheprüfung nach Bandstopp. Ohne eigene gespeicherte Einstellungen
verwendet die GUI diese Werte (100 Versuche, vier Prozesse). Explizit gespeicherte
Werte haben Vorrang. Persönliche absolute Datei- und Ordnerpfade bleiben lokal.
Die bisherigen Engine-Vorgaben für ältere CLI-Konfigurationen bleiben erhalten.

Vollständige portable CLI-Profile sichern die aktuellen Einstellungen und die
beiden zuletzt ausgeführten Serien. Die mitgelieferten STL-Dateien sind mit den
verwendeten Geometrien identisch; Pfade gelten relativ zur jeweiligen JSON-Datei.

| Profil | Bauteil | Versuche | Prozesse | Katalogexport |
| --- | --- | ---: | ---: | --- |
| `configs/gui-current.json` | Dk1i | 100 | 4 | Aus |
| `configs/dk1i-1000.json` | Dk1i | 1.000 | 16 | Aus |
| `configs/kf1i-10000.json` | Kf1i | 10.000 | 16 | Ins benachbarte `BiBaZu_StableComponentenPoses`, mit Push |

Beispiel: `uv run dashas-drop-sim run --config configs/dk1i-1000.json`.
Die ursprünglichen Beispiel- und Batchprofile bleiben als eigene Konfigurationen verfügbar.

`--all-extras` installiert die PyQt6-Oberfläche, OpenCascade für STEP-Dateien,
die lokale Geometriekopie unter `vendor/bibazu_chute_pose` für Symmetrieerkennung
und pytest. Die bisher verwendete Geometrieversion bleibt erhalten. Details
stehen unter [Eigenständige Geometrie](docs/standalone-geometry.md).
Nach der Installation lassen sich GUI und CLI auch mit
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
bleibt für eigene STL- und STEP-Dateien verfügbar. Roadmaps werden in der separaten Roadmap-App erstellt und nummeriert. In der Häufigkeitstabelle
öffnet **Bild** eine schematische 3D-Ansicht der Pose; beobachtete Endlagen und
unbeobachtete Katalogorientierungen sind beschriftet.

Für mehrere Werkstücke gibt es den Bereich **Mehrere Werkstücke**. **Mehrere
Dateien laden** erlaubt die gemeinsame Auswahl mehrerer STL- oder STEP-Dateien;
**Aus Katalog laden** erlaubt Mehrfachauswahl oder **Alle auswählen**. Ein bereits
eingerichtetes Modell lässt sich mit **Aktuelles Werkstück
hinzufügen** in die Liste übernehmen. Jeder Eintrag enthält ein Modell; Simulationsparameter werden gemeinsam gewählt.

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
die Häufigkeits- und Stabilitätstabellen dieses Werkstücks. Die Auswahl öffnet die gespeicherten Ergebnisse dieses Werkstücks.

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
  "output_dir": "C:/Falltests/Ergebnisse",
  "trials": 100,
  "workers": 4,
  "seed": 42,
  "belt_speed_mm_s": 100.0,
  "belt_speed_variation_mm_s": 3.0,
  "belt_variation_interval_s": 0.2,
  "drop_height_mm": 100.0,
  "drop_height_spread_mm": 20.0,
  "lateral_mm": 0.0,
  "lateral_spread_mm": 10.0,
  "density_g_cm3": 1.15,
  "mu_belt": 0.40,
  "mu_wall": 0.20,
  "alpha_deg": 45.0,
  "beta_deg": 0.0,
  "length_mm": 1300.0,
  "timestep_s": 0.001,
  "roughness_enabled": false,
  "roughness_model": "longitudinal_traction",
  "roughness_wall_height_mm": 0.1,
  "roughness_wall_ramp_mm": 3.0,
  "roughness_wall_spacing_mm": 10.0,
  "roughness_belt_height_mm": 0.0,
  "roughness_belt_ramp_mm": 3.0,
  "roughness_belt_spacing_mm": 10.0,
  "disturbance_levels_mm": [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
}
```

Relative Eingabe- und Ausgabepfade in der JSON-Konfiguration werden relativ zum
**Ordner der Konfigurationsdatei** aufgelöst.
Beobachtete Lagen werden unter erkannten Bauteilsymmetrien gruppiert und als
Pose-Katalog exportiert. Roadmaps werden weder eingelesen noch umnummeriert.
Alte `roadmap_path`-Einträge werden beim Laden einer Konfiguration ignoriert.
Ohne das optionale `geometry`-Extra stehen keine erkannten Bauteilsymmetrien
zur Verfügung; dies wird als `symmetry_available: false` ausgewiesen.

Die Bereiche der GUI sind 0–200 mm/s Bandgeschwindigkeit, 0–200 mm
Abwurfhöhe und ±100 mm Querposition, gemessen von der Winkelhalbierenden
zwischen Band und Wand. Bei 0 mm/s endet die Beobachtung nach einer festen
Zeit statt an einem Streckenende. Die genannten Dichte- und Reibwerte sind
konfigurierbar; 1,15 g/cm³ ist der Startwert innerhalb der angegebenen
Materialspanne von 1,12–1,18 g/cm³.

Abwurfhöhe und Querposition bieten jeweils einen **Streubereich ±** in mm.
Pro Abwurf werden beide Werte unabhängig gleichverteilt um den Sollwert
gezogen: **100 mm ±20 mm** Höhe ergibt 80–120 mm; **0 mm ±10 mm** Querposition
ergibt −10 bis +10 mm. **±0 mm** ergibt einen festen Wert. Die gesamten
Bereiche müssen innerhalb 0–200 mm Höhe beziehungsweise ±100 mm Querposition
liegen; die GUI begrenzt die Streubreite passend zum Sollwert. Ungültige
CLI-Konfigurationen werden abgewiesen. Die getrennten Zufallsfolgen verändern
die orientierungsbezogenen Zufallszahlen nicht und bleiben bei gleichem Seed
auch in parallelen Läufen reproduzierbar. Die Störkurve startet aus bereits
eingependelten Posen; die Abwurfstreuung gilt für die Fallversuche.

Die gezogenen Werte stehen als `actual_drop_height_mm` und `actual_lateral_mm`
in `trials.csv`, `trials.jsonl` und bei einer CLI-Einzelvorschau in
`preview.json`. `initial_pos_chute_mm` enthält die tatsächliche Startposition
des Schwerpunkts in Rutschenachsen. Die Anzeige **Letzter Versuch** zeigt diese
Werte ebenfalls; das Manifest dokumentiert die konfigurierte Abwurfverteilung.

### Zufällige Impulse durch Kratzer und Dellen

Im GUI-Bereich **Unebenheiten und zufällige Kontaktkräfte** aktiviert **Zufällige Kontaktstörungen**
ein statistisches Ersatzmodell für kleine Oberflächenfehler. Es wird kein
konkreter Kratzer und kein fester Oberflächenplan modelliert. Vorgaben für die
PTFE-Wand sind maximal 0,1 mm wirksame Kantenhöhe, 3 mm wirksame Kantenlänge und
10 mm mittlerer Abstand. Das sind unkalibrierte Startschätzungen. Für das Band
ist die Kantenhöhe zunächst 0 mm; beide Flächen sind unabhängig einstellbar.
In einer neuen GUI sind vollständige Einzelimpulse mit 49,05 mm/s,
0,2 s mittlerem Abstand und **1 s Anfangsanregung** als Startwerte aktiviert. Explizit gespeicherte
Einstellungen haben Vorrang. CLI-Konfigurationen ohne Störungsfelder behalten
die bisherigen deaktivierten Vorgaben. Die Funktion gilt bei Aktivierung für
Abwürfe sowie die anschließenden Störversuche. Die GUI enthält die Rauheitsmodi
**Kontaktimpulse entlang der Rutsche** (neue Hauptoption) und
**Flache Unebenheitsstöße mit Reibung** (Vergleichsoption) sowie die unten
beschriebenen Gewichtskraft-Pulse und vollständigen Einzelimpulse. Gespeicherte ältere
Konfigurationen ohne `roughness_model` verwenden jetzt die Hauptoption; der
Manifest-Eintrag dokumentiert die Modellversion. Für Vergleiche mit früheren
Ergebnissen ist diese Änderung zu berücksichtigen.

#### Kraftpulse als Anteil der Gewichtskraft

Die zusätzliche Option **Kraftpulse als Anteil der Gewichtskraft** wirkt über
eine endliche Dauer in Rutschenrichtung **+X**. Ein zufällig ausgewählter,
belasteter MuJoCo-Kontakt mit Band oder Wand erhält die Kraft
`F = roughness_force_fraction × Masse × 9,81 m/s²`. Der Hebelarm zum Schwerpunkt
erzeugt dabei ein Drehmoment. Es wird keine Pose bevorzugt oder erzwungen.

Einstellbare Startwerte sind **10 %**, **50 ms** Pulsdauer und **200 ms** mittlere
Pause. Die Pausen werden exponentiell verteilt gezogen; Pulse überlappen nicht.
Ein fälliger Start ohne belasteten Kontakt wird übersprungen. Der Kontaktpunkt
wird während des Pulses geometrisch nachgeführt. Kurzzeitig null berechnete
Normallast beendet einen weiterhin berührenden Kontakt nicht. Fehlt die
ursprüngliche Fläche, wird zum nächsten tatsächlichen Kontakt mit der anderen
Rutschenfläche gewechselt; solche Wechsel stehen im Ereignisprotokoll.
Erst der Verlust sämtlicher Rutschenkontakte beendet den Puls. Eine numerische
Berührungstoleranz von 0,1 µm fängt Rundungsfehler ab. Im Flug wird keine Kraft angelegt. Die
Kontaktprüfung erfolgt vor jedem Physikschritt. Innerhalb eines Schrittes
verlorener Kontakt wird beim nächsten Schritt erkannt.

Die Kraft wirkt auch bei fehlendem Schlupf. Höhe, Länge und Abstand der
Rauheit sind in diesem Modus ohne Wirkung. Das Modell stellt eine externe
mechanische Anregung dar und kann Energie zuführen. Es ist kein passives
Reibmodell und kein vermessenes Oberflächenprofil. Die Parameter sind
unkalibrierte Vergleichswerte; für realistische Häufigkeiten müssen sie anhand
realer Versuche angepasst werden. Eine longitudinale Kraft allein ist keine
zusätzliche vertikale Sprungkraft.

```json
{
  "roughness_enabled": true,
  "roughness_model": "weight_force",
  "roughness_force_fraction": 0.1,
  "roughness_force_duration_s": 0.05,
  "roughness_force_interval_s": 0.2,
  "contact_time_constant_s": 0.005,
  "contact_damping_ratio": 0.5,
  "contact_torsional_friction_mm": 5.0,
  "contact_rolling_friction_mm": 0.1
}
```

Das ist ein Parameterfragment; Bauteil und Ausgabeordner müssen ergänzt werden.
Im sichtbaren Lauf markieren ein Punkt und ein Pfeil den Angriffsort und die
Richtung. Rot bezeichnet einen aktiven Puls, Gelb einen kürzlich beendeten.
Die Markierung hat eine kurze Nachanzeige und eine schematische Länge;
ein neuer Puls ersetzt die vorherige Markierung. Die Nachanzeige verlängert
die physikalische Pulsdauer nicht.
`roughness_events.csv` und `.jsonl` enthalten Kraft in N, konfigurierte und
tatsächliche Dauer, tatsächlichen Impuls in N·s, Drehimpuls und Abbruchstatus.
Für einen ununterbrochenen Puls gilt `Impuls = F × Dauer`. Bei 10 % und 50 ms
entspricht dies ohne Kontaktwiderstände `Δv = 49,05 mm/s`. Die protokollierte
Kraftarbeit ist eine Trapezschätzung und keine isolierte Messung der Änderung
der gesamten kinetischen Energie (`energy_semantics`).

#### Vollständige Einzelimpulse

Die GUI-Option **Einzelimpulse mit vollständigem Kraftstoß** setzt einen
einmaligen Kraftstoß **J** am zufälligen belasteten Kontaktpunkt mit Band oder
Wand an. Die Stärke wird als **Freier Geschwindigkeitsstoß (J/m)** in mm/s
eingegeben: `J = Bauteilmasse × roughness_impulse_delta_v_mm_s / 1000` in N·s.
Die GUI zeigt für STL-Dateien den resultierenden Kraftstoß und die Modellmasse.
Bei STEP-Dateien erscheint der Wert nach der Modellvorbereitung; der Lauf
protokolliert stets die tatsächliche Masse und Stärke.

Der Startwert **49,05 mm/s** entspricht `0,1 × g × 50 ms`. Ein kräftiger
Diagnosewert **490,5 mm/s** entspricht `g × 50 ms`. Beide Werte beschreiben
den vollständigen Kraftstoß pro Masse, keine Anwendungsdauer. Die bisherige
Pulsdauer und der Gewichtskraftanteil haben in diesem Modus keine Wirkung.
Dies ist auch die GUI-Vorgabe ohne bereits gespeicherte Einstellungen.

**Dauer der Anfangsanregung** begrenzt alle zufälligen Kontaktstörungen auf
den Anfang des Versuchs. Mit `roughness_active_duration_s: 1.0` endet die
Anregung nach 1 s; anschließend läuft das Bauteil ohne weitere Kontaktstörungen
bis zum Ausgang. Das erleichtert die Einpendelprüfung der letzten 0,5 s.
Die Prüfkriterien werden dabei nicht gelockert. Das Zeitfenster beginnt mit
der Fallbeobachtung, einschließlich Freiflug beim Abwurf; bei Störversuchen
beginnt es nach dem Wiedereinsetzen. Aktive Kraftpulse werden am Fensterende
mit `pulse_status: window_end` beendet. Eine spätere erneute Berührung öffnet
das Fenster nicht wieder. Bandgeschwindigkeitsschwankungen bleiben aktiv.

`0` beziehungsweise **Ganzer Lauf** schaltet die Begrenzung aus. Ältere
CLI-Konfigurationen ohne das Feld behalten diese bisherige Betriebsart.
Es werden nur vollständige Physikschritte innerhalb des Fensters angeregt;
die tatsächliche Grenze kann deshalb um weniger als einen Zeitschritt vor
dem eingestellten Ende liegen. Die Angabe in `manifest.json` dokumentiert
das Fenster. Für kürzere Strecken oder schnellere Transporte muss genügend
Auslauf nach der gewählten Anfangszeit vorgesehen werden.

```json
{
  "roughness_enabled": true,
  "roughness_model": "weight_impulse",
  "roughness_impulse_delta_v_mm_s": 49.05,
  "roughness_force_interval_s": 0.2,
  "roughness_active_duration_s": 1.0
}
```

Ein fälliges Ereignis wird nur bei einem belasteten, geometrisch berührenden
Kontakt ausgelöst, im Flug übersprungen. Die Wartezeit ist exponentiell
verteilt. Der Stoß wird einmal vor einem Physikschritt über Kontakt-Jacobian
und Massenmatrix auf die Geschwindigkeiten übertragen; der Hebelarm erzeugt
Winkelgeschwindigkeit. Danach lösen MuJoCos reguläre Kontakte die Bewegung
weiter auf. Eine spätere Ablösung kürzt den bereits übertragenen Stoß nicht.
Die Bauteilorientierung wird nicht direkt gesetzt.

Ereigniszeiten liegen auf Integrationsgrenzen. `requested_time_s`, `time_s`
und `scheduling_delay_s` dokumentieren die Rundung um höchstens einen
Zeitschritt. Es wird höchstens ein Stoß je Schritt ausgeführt, ohne
nachgeholte Ereignisbündel. `actual_duration_s = 0` kennzeichnet den
Geschwindigkeitssprung; `impulse_ns` enthält dennoch den vollständigen
Kraftstoß. Die Energieänderung beim Stoß ist aus der Punktgeschwindigkeit
und der effektiven Masse berechnet (`instantaneous_velocity_jump`). Das
ist eine externe, unkalibrierte Anregung mit möglicher Energiezufuhr.

Im echten Regressionstest steht die originale Dk1i-Geometrie auf ihrer
dreieckigen Stirnfläche und gleitet ohne Störung bis zum Rutschenende mit
weniger als 1° Orientierungsänderung. Ein einmaliger Stoß von **490,5 mm/s**
nach 1 s kippt die Längsachse um etwa **90°**; die andere Lage bleibt bis
zum Ausgang erhalten. `tests/test_impulse_tipping.py` prüft diesen Ablauf.
Das ist ein Funktionsnachweis, keine Kalibrierung realer Unebenheiten.

#### Kontaktfederung und Drehreibung

**Kontakt-Dämpfungsverhältnis** und **Kontakt-Zeitkonstante** steuern MuJoCos
`solref`. Der bisher feste Wert 1,2 ist überdämpft; 1 ist kritisch gedämpft,
ein Wert unter 1 erlaubt mehr Rücksprung. Ein Startwert 0,5 ist ein Vergleich,
kein gemessener Rückprallkoeffizient. Die Zeitkonstante muss mindestens zwei
Zeitschritte betragen. Alte Konfigurationen ohne diese Felder behalten 1,2
und 5 ms. Siehe [MuJoCo Kontaktparameter](https://mujoco.readthedocs.io/en/3.14.0/modeling.html#solver-parameters).

Die bisherigen Werte für **Torsionsreibung** (5 mm) und **Rollreibung**
(0,1 mm) sind ebenfalls einstellbar. Diese effektiven Längen beschreiben
Widerstand gegen Kontaktrotation und können kleine Bauteile zusätzlich
stabilisieren. Beide auf null deaktiviert diese zusätzlichen
Rotationswiderstände, während Gleitreibung erhalten bleibt. Die bestehenden
Werte bleiben für Vergleiche verfügbar; sie sind keine vermessenen
Materialkennwerte. Siehe [MuJoCo Reibungsdimensionen](https://mujoco.readthedocs.io/en/3.14.0/computation/index.html#condim).

#### Kontaktimpulse entlang der Rutsche

Diese Option bildet lokale Änderungen der Kontaktreibung als kurze Impulse
in Rutschenrichtung **X** ab. Sie erzeugt keinen zusätzlichen Normalimpuls
von Band oder Wand weg. Die bestehende Kontaktreibung bleibt im
MuJoCo-Solver. Zufällige zusätzliche Brems-/Mitnahmeimpulse greifen an
belasteten Kontaktpunkten an: Wandimpulse bremsen den dortigen Schlupf;
Bandimpulse können je nach Relativgeschwindigkeit bremsen oder beschleunigen.
Mit dem Hebelarm zum Schwerpunkt entsteht ein Kippmoment um die Querachsen.

Für das statistische Ersatzmodell wird eine zusätzliche wirksame Reibung
`delta_mu = mu × gezogene Höhe / Kantenlänge` angenommen. Die überfahrene
Kontaktzone liefert die wirksame Dauer `tau = Kantenlänge / |X-Schlupf|`.
Der Impuls wird begrenzt auf
`|J_X| = min(delta_mu × Normalkraft × tau, effektive Masse × |X-Schlupf|)`.
Die Normalkraft stammt aus dem tatsächlich belasteten Kontakt; die
effektive Masse berücksichtigt Bauteilmasse, Trägheit und Hebelarm. Dadurch
kann der Impuls den Schlupf abbremsen, ihn aber nicht umkehren. Die Beziehung
zwischen Höhe/Länge und zusätzlicher Reibung ist eine **unkalibrierte
Modellannahme**; aus einer gemessenen Kratzertiefe folgt kein eindeutiger
Impuls. Die wirksame Dauer wird zu einem einzelnen Impuls integriert, nicht
als zeitlich aufgelöster Kratzer simuliert.

Die Ereignisrate folgt dem longitudinalen Schlupfweg. Impulsstärke und -rate
hängen damit von der Bewegung gegenüber der jeweiligen Kontaktfläche ab;
es werden keine willkürlichen Kräfte am Schwerpunkt oder Drehraten gesetzt.
Bei null Reibwert, fehlendem belasteten Kontakt oder fehlendem X-Schlupf
gibt es in dieser Option keine Zusatzimpulse.

#### Flache Unebenheitsstöße mit Reibung

Die Ereignisse werden mit exponentiellen Abständen entlang des relativen
Gleitwegs ausgelöst. Doppelter Gleitweg erzeugt im Mittel doppelt so viele
Ereignisse; die Zahl der Kollisionskörper vervielfacht die Ereignisrate nicht.
Ein Impuls greift an einem zufällig ausgewählten, tatsächlich belasteten
MuJoCo-Kontaktpunkt an. Die Auswahl berücksichtigt Normalkraft und lokalen
Gleitweg. Es werden die diskreten Kontaktpunkte des Solvers verwendet, keine
willkürlich interpolierten Flächen zwischen Kontakten eines konkaven Teils.
Die Kontaktgeschwindigkeit enthält Translation und Drehung; für das Band
wird dessen eingestellte Oberflächengeschwindigkeit abgezogen.

Die wirksame Höhe wird pro Ereignis gleichverteilt zwischen 0 und der
Maximalhöhe gezogen. Das Verhältnis Höhe/Kantenlänge bestimmt die Stärke
eines flachen virtuellen Stoßes. Für kleine Steigungen ist der Impuls ungefähr
`J = effektive Masse × relative Gleitgeschwindigkeit × Höhe / Kantenlänge`.
Die effektive Masse berücksichtigt Trägheit und Hebelarm; dadurch erzeugt
ein außermittiger Impuls auch Rotation. Der Stoß ist unelastisch und wird auf
den zusätzlichen Anteil der Unebenheit begrenzt. Nach dem Normalstoß wirkt
zusätzliche Gleitreibung auf der virtuellen Kontaktfläche, begrenzt durch
`mu × zusätzlicher Normalimpuls` und den zum Abbremsen des Schlupfs nötigen
Impuls. Sie berücksichtigt die getrennten Reibwerte von Wand und Band.
Der Stoß lenkt Gleitbewegung von der Fläche weg und bremst sie. Zur ruhenden Wand erhöht er die
kinetische Energie nicht; beim bewegten Band kann der Bandantrieb Energie
übertragen. Stillstand und fehlender belasteter Kontakt erzeugen keine Impulse.
Kontaktgeschwindigkeiten unter 0,1 mm/s werden als numerisches Rauschen
ausgeklammert. Die Erfassung erfolgt ungefähr alle 5 ms, bei gröberem
Physikzeitschritt einmal pro Schritt.

Die Geschwindigkeitsänderung wird direkt aus Kontakt-Jacobian und MuJoCo-
Massenmatrix berechnet; eine Kraft wird nicht willkürlich für einen
Zeitschritt festgelegt. Die verwendeten APIs sind in der
[MuJoCo-Referenz](https://mujoco.readthedocs.io/en/3.14.0/APIreference/APIfunctions.html#mj-jac)
dokumentiert. Separate Zufallsfolgen pro Versuch und Fläche machen die
Störungen auch bei parallelen Läufen reproduzierbar. Für die Störkurve wird
pro Ausgangspose dieselbe Zufallsfolge verwendet, wobei der tatsächlich
durchlaufene Gleitweg und die Kontakte von der Bewegung abhängen.

Das Modell bildet kleine Fehler während des Gleitens ab. Rein rollende
Kontakte ohne Schlupf, einzelne große Dellen, elastische Verformung und das
exakte Überfahren einer vermessenen Oberfläche werden damit nicht aufgelöst.
Höhe/Kantenlänge darf höchstens 0,25 betragen. Eine gemessene Kratzertiefe
bestimmt die reale Impulsstärke nicht allein: Bis zur Kalibrierung an
Transportversuchen sollten Höhe, Kantenlänge und Abstand variiert werden.

Die beiden Rauheitsmodi speichern zusätzlich die **X-Komponente des Kraftimpulses**,
das Drehimpuls-Vektorprodukt `Hebelarm × Impuls`, Normalkraft, Reibanteil,
wirksame Kontaktzonendauer sowie die aktuelle Bandgeschwindigkeit in den
Unebenheitsereignissen. Beide erzeugen keine zusätzliche kinetische Energie
im Bezugssystem der jeweiligen Kontaktfläche.

### Schwankende Bandgeschwindigkeit

**Bandgeschwindigkeitsschwankung** stellt eine zeitliche ±-Schwankung in mm/s
ein, getrennt von der Streuung des Abwurfs. **±0 mm/s** deaktiviert sie; als
Schätzung für das reale Band bietet sich **±3 mm/s** an: bei 100 mm/s bleibt
die Oberfläche im Bereich 97–103 mm/s. Der gesamte Bereich muss innerhalb
0–200 mm/s bleiben. **Änderungsintervall Band** legt den zeitlichen Abstand
zwischen zufälligen Geschwindigkeitszielen fest. **0,2 s** ist eine
ungemessene Startannahme, kein bestätigter Messwert.

Die Ziele werden gleichverteilt gezogen und mit glatten Kosinusübergängen
verbunden. Die erste Geschwindigkeit ist der Sollwert; Änderungen erfolgen
kontinuierlich, ohne Zeitschritt-Rauschen. MuJoCos `surfacevel` wird vor jedem
Physikschritt angepasst. Die Kräfte entstehen über die reguläre
Kontaktreibung am Band, einschließlich ihres Hebelarms; die PTFE-Wand ruht.
Das Attribut ist laut
[MuJoCo-Dokumentation](https://mujoco.readthedocs.io/en/3.14.0/XMLreference.html#body-geom-surfacevel)
während eines Laufs veränderbar und wirkt tangential auf die Kontakte.
Bandfluktuationen sind auch bei abgeschalteten Unebenheitsimpulsen verfügbar.

Eine eigene Seed-Folge macht den Geschwindigkeitsverlauf unabhängig von
Zeitschritt und paralleler Ausführung. Bei jedem Fallversuch wird er
zurückgesetzt; für Störversuche startet er nach dem Wiedereinsetzen aus der
eingependelten Pose. `belt_speed_control_points` in den Versuchsdateien
enthält die Stützpunkte mit Laufzeit und Geschwindigkeit (einschließlich
des nächsten Zielpunkts für den letzten Übergang); Konfiguration und
Manifest dokumentieren die Interpolation. Die Parameter werden auch mit
**Als Standard speichern** gesichert.

Bei idealem Rollen ohne Schlupf entstehen aus dem longitudinalen Reibmodell
allein keine Unebenheitsimpulse. Bandbeschleunigungen können trotzdem über
den Kontakt Kräfte und Kippmomente erzeugen. Die neue Anregung garantiert
kein Umkippen einer bestimmten Pose: reale Häufigkeiten und Kippübergänge
müssen mit Transportversuchen verglichen werden. Ein räumlich aufgelöstes
Rollmodell auf einer unebenen Fläche ist darin nicht enthalten.

## Automatischer Posenkatalog und 39er-Lauf

Der [Prozessvergleich auf dem aktuellen Rechner](docs/worker_benchmark.md) ergab
für Dk2a mit diesen Parametern keinen Vorteil durch mehr als acht Prozesse.
Das dort beschriebene Benchmark-Werkzeug prüft Laufzeit und Ergebnisgleichheit.

Im GUI-Bereich **Posenkatalog automatisch aktualisieren** den lokalen Klon von
`BiBaZu_StableComponentenPoses` und den Ordner mit den ursprünglichen STL-/STEP-Paaren
wählen. Export aktivieren; **Nach jedem Werkstück automatisch nach GitHub pushen**
veröffentlicht die Ergebnisse nach jedem vollständigen Lauf. **Alle 39 hinzufügen**
füllt die Werkstückliste, **Werkstück-Batch starten** arbeitet sie mit den aktuellen
Parametern ab. Für reine Falltests **Zusätzliche Drehimpuls-Störkurven berechnen**
ausschalten. Alle Export- und Physikeinstellungen lassen sich als Standard speichern.

Die abgestimmte Konfiguration liegt in `configs/catalog39.json`: 1.000 Versuche je
Werkstück, 8 Prozesse, 500 mm, 100 ±3 mm/s, 100 mm Abwurfhöhe, 0 mm Querposition,
keine Abwurfstreuung, longitudinale Wandimpulse, keine zusätzlichen Störkurven.
CLI-Start und Wiederaufnahme verwenden denselben Simulations- und Exportpfad:

```powershell
uv run --all-extras dashas-drop-sim catalog-batch --config configs/catalog39.json
```

Die GUI-Einstellungen für ein Einzelwerkstück werden genauso verwendet. Bei
aktiviertem Export werden vollständige frühere Läufe mit identischer Geometrie,
Physikkonfiguration und Simulationscode wiederverwendet. Sichtbare Serien werden
immer neu simuliert. Ein laufender Versuch speichert seine eingefrorene Konfiguration.
`results/catalog39/catalog_batch.json` protokolliert den CLI-Gesamtlauf; die lokale
Wiederaufnahmeinformation liegt zusätzlich im `.git/catalog-progress` des Ziel-Repos.
Nach Abbruch wird nur das noch unvollständige Werkstück erneut berechnet.

Je Werkstück entstehen originale STL und STEP, eine bebilderte README, `poses.json`,
die inhaltsgleiche `poses.yaml`, `frequencies.csv`, `trials.csv`, Konfiguration und
Manifest. Nur beobachtete, eingependelte Endlagen stehen in der Posenliste; die
Hauptprozentwerte und Wilson-Intervalle beziehen sich auf **alle** abgeschlossenen
Abwürfe. Nicht eingependelte Fälle bleiben als Ergebnisstatus sichtbar. Zusätzlich
steht der Prozentanteil unter den eingependelten Versuchen dabei. Ausführliche
Impulsereignisse und Geschwindigkeitsstützpunkte bleiben im lokalen Rohdatenlauf;
Konfiguration, Seeds, Modellversionen und Software-Commit ermöglichen die Zuordnung.

Bei aktivierter Ruheprüfung nach Bandstopp werden auch die ruhigen stationären
Endlagen automatisch unter Bauteilsymmetrie und derselben 5°-Toleranz klassifiziert.
Die bebilderte README zeigt diese Klassen in einem eigenen Abschnitt. Sie stehen
in `poses_after_stop.json` und `.yaml`, `frequencies_after_stop.csv` sowie
`trials_after_stop.csv`. Diese Versuchstabelle umfasst auch die weiterhin
bewegten Teile und die Fälle ohne Ankunft; nur ruhige Endlagen erhalten eine ID.
Die IDs enthalten `_stop_` und werden in `pose_registry_after_stop.json`
unabhängig von den Fahrposen gespeichert. Die Hauptanteile beziehen sich auf
alle Abwürfe; zusätzlich sind die Anteile unter den Ankünften und unter den
ruhigen Endlagen angegeben. Bei einem erneuten Export bereits berechneter
Läufe werden bisher fehlende Ruheklassen ergänzt; vorhandene Rohdaten und
Archive bleiben erhalten. Ein neuer Simulationslauf ist dafür nicht nötig.

Die exportierten Pose-IDs bleiben unabhängig von der Häufigkeit reserviert.
`pose_registry.json` hält auch frühere, aktuell nicht beobachtete IDs bereit.
Orientierungen werden unter der erkannten Bauteilsymmetrie gruppiert.
Die exportierten Pose-Dateien können anschließend in der separaten Roadmap-App verwendet werden:

```python
from dashas_drop_sim.catalog_export import recognize_pose
result = recognize_pose(quaternion_xyzw, catalogue_json)
```

`recognition` enthält die diskreten Symmetriequaternionen beziehungsweise die
gerichtete kontinuierliche Symmetrieachse, Winkeltoleranz und Mehrdeutigkeitsabstand.
Quaternionen sind **xyzw**, Bauteil → Rutsche. `transform_chute_from_source_mm`
transformiert Punkte der ursprünglichen CAD-/STL-Koordinaten nach Rutschenkoordinaten;
die Translation berücksichtigt den Originalschwerpunkt. Rutschenachsen: X entlang
des Bandes, Y von der Wand weg, Z vom Band weg. Die Bilder zeigen tatsächlich
beobachtete Repräsentanten; die Längsposition wird für die Nahansicht ausgeblendet.

Erst der vollständig geprüfte Export ersetzt den bisherigen Werkstückordner; pro
Werkstück wird committed. Netzwerkfehler lassen einen lokalen Commit mit Status
`pending` zurück. **Abgeschlossenen Lauf erneut exportieren …** oder der CLI-Befehl
unten wiederholen Export/Push ohne Simulation. Ungesicherte manuelle Änderungen
werden nicht überschrieben. Andere Exporte werden über eine Prozesssperre serialisiert.

```powershell
uv run --all-extras dashas-drop-sim export --run-dir results/DEIN_LAUF --catalog-repo ../BiBaZu_StableComponentenPoses
```

`--no-push` erstellt nur den lokalen Commit. `--cad-dir` wählt einen anderen
Original-CAD-Ordner. Der Export verweigert unvollständige Läufe und eine seit dem
Versuch geänderte Geometrie. Die Katalogübersicht nennt je Werkstück Versuchszahl,
Strecke und beobachtete Posen; alle Aussagen gelten für die jeweilige Simulation.

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
| `stop_observations.csv` | Zusätzliche Ruheprüfung nach Bandstopp mit Fahrstatus, Ruhepose und Nachlaufweg |
| `disturbances.csv` | Einzelne Störversuche nach Pose, Richtung und Stärke |
| `roughness_events.csv` / `.jsonl` | Jeder Unebenheitsimpuls mit Phase, Versuch, Kontaktpunkt, Relativgeschwindigkeit, Impuls in N·s und Energieänderung |
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

Roadmaps und deren Nummerierung werden ausschließlich in der separaten Roadmap-App bearbeitet.

### Ruheprüfung nach Bandstopp

Die GUI-Option **Am Streckenende anhalten und Ruhe prüfen** ist zunächst
eingeschaltet. Nach der regulären Messung am virtuellen Streckenende bremst
das Band mit einer glatten Kosinusrampe auf null (Startwert **0,2 s**). Danach
läuft die Simulation noch **2 s** ohne Bandgeschwindigkeitsschwankungen oder
Kontaktstörungen. Die Bremsdauer und Beobachtungsdauer sind einstellbar.
Das Bauteil wird dabei weder fixiert noch seine Geschwindigkeit zurückgesetzt.
Die idealisierten Kontaktflächen setzen sich hinter der Messlinie fort;
ein physischer Abwurf von einer Bandkante wird nicht simuliert.

Der Reiter **Nach Bandstopp** zeigt je Abwurf den Status während der Fahrt und
nach dem Anhalten sowie die dann gemessene Pose. Als ruhig zählt nur eine
Lage mit stabilem Kontakt und stabiler Orientierung über das letzte
0,5-s-Fenster; zusätzlich wird die Bewegung in allen drei Raumrichtungen
geprüft (Positionsspanne höchstens 1 mm und gefilterte Geschwindigkeit
höchstens 1 mm/s). Weiterrollende oder gleitende Teile bleiben `unsettled`.
Die volle Beobachtungsdauer läuft auch nach einem ersten Einpendeln weiter.

Die Fahrmessung und ihre Häufigkeiten bleiben unverändert. Die Ruheprüfung
steht separat in `trials.jsonl` als `stop_observation`, in
`stop_observations.csv` und in `summary.json` unter `belt_stop_observation`.
Der Katalogexport enthält die Zusatzprüfung, die klassifizierten Ruheposen
mit Bildern und eine getrennte Zusammenfassung; sie ersetzt nicht die Fahrposen im Katalog.
Die Störkurven prüfen weiterhin das laufende Band.
Für JSON-Konfigurationen heißen die Optionen `belt_stop_enabled`,
`belt_stop_ramp_s` und `belt_stop_observation_s`; ohne Angabe bleibt die
Zusatzprüfung dort aus, damit ältere Konfigurationen dieselben Versuche ausführen.

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
