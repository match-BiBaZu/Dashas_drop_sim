# Eigenständige Geometrie

- `vendor/bibazu_chute_pose/` enthält die bisher festgelegte Geometrieversion `94cffbca9e4a227beea266bb098f8186e1d1029d` als lokale Kopie.
- `uv sync --python 3.12 --all-extras` installiert diese Kopie ohne Geometrie-Git-Download.
- Das optionale Extra `geometry` enthält die Geometrie-/Symmetrie-Abhängigkeit.
- `src/dashas_drop_sim/assets/geometry/cad/` enthält 39 STL-/STEP-Paare.
- `SOURCE.json` und die Lizenz dokumentieren Herkunft, Revision und Dateiprüfsummen.
- CAD-Export und GUI verwenden standardmäßig die lokalen CAD-Paare.
- Alte CAD-Pfade werden bei einer exakt vorhandenen lokalen Kopie umgeleitet.
- Roadmap-Auswahl, Roadmap-Abgleich und die Neunummerierung von Roadmaps gehören nicht mehr zur Simulationsanwendung.
- Alte `roadmap_path`-Einträge in Laufkonfigurationen und GUI-Einstellungen werden ignoriert.
- Beobachtete Pose-IDs, Symmetriegruppierung und Pose-Exporte bleiben erhalten; sie sind Eingaben für die separate Roadmap-App.
- Historische Rohdaten-Feldnamen wie `roadmap_pose_id` bleiben aus Kompatibilitätsgründen erhalten; sie lösen keine Roadmap-Zugriffe aus.
- Reorientation Control und das ursprüngliche Geometrie-Repository werden nicht benötigt.
- Ein separat ausgewähltes Ergebnis-Repository bleibt nur für optionale Veröffentlichungen erforderlich.
