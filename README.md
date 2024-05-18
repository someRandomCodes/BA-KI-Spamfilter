# Bachelorarbeit: Thomas Guede Stork

## Struktur des Projekts

Dieser Code wurde für die Bachelorarbeit von Thomas Guede Stork entwickelt. Im Repository finden sich verschiedene Skripte und Verzeichnisse, die für die Sortierung und Verarbeitung von E-Mail-Daten verwendet werden.

### Wichtige Dateien und Verzeichnisse

- `__mails`: Enthält alle E-Mail-Daten.
- `email_processing_and_saving.py`: Skript zur Sortierung der E-Mails in entsprechende Ordner.
- `__mails_cleaned_new`, `__mails_cleaned_new_testdata`, `__mails_new`, `__mails_new_testdata`, `__mails_new_db`: Verzeichnisse mit sortierten E-Mails. "cleaned" beinhaltet nur Subjekt und Body der E-Mails. "testdata" enthält Validierungsdaten mit je 4.906 E-Mails klassifiziert als Spam oder Ham.

### Modell-Training

- `NaiveBayes_twoModelsPlot.py`: Trainiert das Naive Bayes Modell. Wichtig: Pfade in den Zeilen 238 und 288 anpassen.
- `NaiveBayes_twoModelPlot_v2.py`: Naive Bayes Modell mit angepassten Hyperparametern.
- `tensorflow_way_load-batched_optuna.py`: Neuronales Netzwerk-Modell mit Tensorflow.

### Tools

- **Optuna Dashboard**: Start mit `optuna-dashboard sqlite:///Model_NN_bateched_full_Ergebnisse/_optuna_studies/BatchedEmails.db`
- **TensorBoard**: Befehl muss noch ergänzt werden.

> **Hinweis**: Tools müssen installiert sein, bevor sie genutzt werden können.
