# Prozessvergleich vom 4. Oktober 2026

Rechner: Intel Core i9-13900E, 24 Kerne / 24 logische Prozessoren.
Werkstück: Dk2a. Physikparameter: `configs/catalog39.json`.
Der 39er-Gesamtlauf war für den Vergleich angehalten.

Je Einstellung wurden zweimal dieselben 384 Abwürfe gerechnet; im zweiten
Durchgang war die Reihenfolge umgekehrt. Die Geometrie war bereits vorbereitet.
Die Zeit enthält Start und Beenden der Worker. Die vollständigen sortierten
Versuchsergebnisse einschließlich Impulsereignissen wurden per SHA-256 verglichen.

| Prozesse | Median für 384 Abwürfe | Ergebnisvergleich |
| ---: | ---: | --- |
| 8 | 34,50 s | exakt identisch |
| 12 | 36,95 s | exakt identisch |
| 16 | 38,80 s | exakt identisch |

Alle sechs Läufe enthielten 384 vollständige, numerisch gültige Ergebnisse.
8 Prozesse waren bei dieser Messung am schnellsten; der Gesamtlauf bleibt daher
bei 8 Prozessen. Die Werte sind ein Vergleich für Dk2a unter dem aktuellen
Rechnerbetrieb, keine allgemeine Aussage über alle Bauteile und Rechner.

Ein vorausgegangener kürzerer Vergleich mit 96 Abwürfen scheiterte bei
24 Prozessen während der Worker-Initialisierung mit MuJoCos Meldung
`Could not allocate memory` und anschließend `BrokenProcessPool`.
Diese Einstellung wird für den Gesamtlauf nicht verwendet.

Wiederholbarer Aufruf:

```powershell
uv run --all-extras python -m dashas_drop_sim.benchmark --config configs/catalog39.json --workpiece Dk2a --trials 384 --workers 8 12 16 --repeats 2 --output results/worker_benchmark_long.json
```

Das Werkzeug dokumentiert auch fehlgeschlagene Prozesszahlen. Eine Empfehlung
setzt gültige, identische Ergebnisse voraus; bei höchstens 5 % Zeitunterschied
wird die kleinere Prozesszahl bevorzugt. Die Einstellung der eigentlichen
Falltests wird dadurch nicht automatisch verändert.
