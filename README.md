
# Bachelorarbeit: Thomas Guede Stork

Dieses Repository enthält den Code und die Dokumentation für die Bachelorarbeit von Thomas Guede Stork. Im Folgenden wird die Struktur des Projekts erläutert und Anweisungen zur Verwendung der enthaltenen Skripte gegeben.

## Projektstruktur

### Wichtige Dateien und Verzeichnisse

- `__mails`: Enthält alle E-Mail-Daten.
- `email_processing_and_saving.py`: Skript zur Sortierung der E-Mails in entsprechende Ordner.
- `__mails_cleaned_new`, `__mails_cleaned_new_testdata`, `__mails_new`, `__mails_new_testdata`, `__mails_new_db`: Verzeichnisse mit sortierten E-Mails.
  - "cleaned" beinhaltet nur Subjekt und Body der E-Mails.
  - "testdata" enthält Validierungsdaten mit je 4.906 E-Mails klassifiziert als Spam oder Ham.

### Modell-Training

- `NaiveBayes_twoModelsPlot.py`: Trainiert das Naive Bayes Modell.
  - Wichtig: Pfade sowie Namen in den Zeilen 227, 228 und 240 anpassen.
- `NaiveBayes_twoModelPlot_v2.py`: Naive Bayes Modell mit angepassten Hyperparametern.
  - Wichtig: Pfade sowie Namen in den Zeilen 206, 207 und 219 anpassen.
- `tensorflow_way_load-batched_optuna.py`: Neuronales Netzwerk-Modell mit TensorFlow.

> **Hinweis**: Es muss zuerst email_processing_and_saving.py ausgeführt worden sein, damit die Trainingsdaten zur verfügung stehen!



### Tools

- **Optuna Dashboard**: Start mit `optuna-dashboard sqlite:///Model_NN_batched_Ergebnisse/_optuna_studies/BatchedEmails.db`
- **TensorBoard**: Starten mit `tensorboard --logdir=Model_NN_batched_Ergebnisse/tensorboard_logs`

> **Hinweis**: Diese Tools müssen installiert sein, bevor sie genutzt werden können. Die Pfade müssten evtl. angepasst werden


### Ansehen der Wichtigsten Spam / Ham wörter

- `NaiveBayes_significant_words.py`: Gibt die X wichtigsten Wörter für eine Spam / Ham klassifizierung aus.

> **Hinweis**: Die Modelle müssen erstellt worden sein bzw. die Vectorizer müssen vorhanden sein. In der Datei müssen die Namen der Vectoriser angepasst werden!

# Rspamd Docker Setup

## Einleitung

Dieser Abschnitt beschreibt die Einrichtung und Nutzung von Rspamd innerhalb eines Docker Containers.

## Voraussetzungen

- Docker muss installiert sein.
- Docker Compose ist erforderlich.

## Starten von Rspamd

1. Öffne das Terminal und navigiere zum Ordner `./rspamd`.
2. Führe den folgenden Befehl aus, um Rspamd in einem Docker Container zu starten:
   ```sh
   docker-compose up -d
   ```
3. Die Weboberfläche ist nun unter `localhost:11334` erreichbar.

## Passwort einrichten

Nach dem Starten des Containers kann das Passwort mit folgenden Befehlen gesetzt werden:
```sh
docker exec -ti -u root rspamd bash -c 'echo password="rspamadm pw -q -p YOUR_PASSWORD" > /etc/rspamd/local.d/worker-controller.inc'
docker exec -ti -u root rspamd bash -c "kill -HUP 1"
```
Ersetze `YOUR_PASSWORD` durch das gewünschte Passwort.

## Bayes Filter und Redis

1. Stelle sicher, dass Redis installiert und verbunden ist.
2. Führe den Config-Wizard aus:
   ```sh
   docker exec -ti -u root rspamd bash
   rspamadm configwizard
   ```
   Folge den Anweisungen, um Redis einzurichten.

## Training und Testen

- Führe `email_processing_and_saving.py` aus, um die benötigten Ordner zu erstellen (falls dies nicht schon geschehen ist).
- Zum Trainieren des Modells: `rspamd_train_emails.py`.
- Zum Testen der Leistung von Rspamd: `rspamd_check_emails.py`.



# Weboberfläche für E-Mail Klassifizierung

Es befindet sich eine `.env.sample` Datei im `server` Ordner. Diese beinhaltet einen Secret Key, um eine CSRF-Protection umzusetzen. Aus der `.env.sample` muss eine `.env` Datei erstellt werden. Hier kann das Passwort bei Bedarf angepasst werden.


Im Ordner `server` befindet sich eine `server.py` Datei. Diese muss mit dem Befehl `python3 server.py` gestartet werden. Anschließend ist die Weboberfläche unter `localhost:8080` zu erreichen.

**Wichtig:** Die Modelle müssen vortrainiert sein und existieren, damit der Server die Modelle findet. In der `server.py` Datei in den Zeilen 28-32 kann angegeben werden, welche Modelle genutzt werden sollen.
