import os
import pandas as pd
import numpy as np
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from joblib import dump, load
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.pruners import MedianPruner
from sklearn.metrics import accuracy_score, confusion_matrix
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Funktion, die Trainings- und Validierungsgeneratoren zurückgibt
def create_email_generators(folder_name, vectorizer, batch_size=32, validation_split=0.2, seed=12345, limit=None):
    random.seed(seed)
    spam_folder_path = os.path.join(folder_name, 'spam')
    ham_folder_path = os.path.join(folder_name, 'ham')

    # Erstellen einer Liste von Pfaden und Labels
    email_paths = []
    labels = []

    for filename in os.listdir(spam_folder_path):
        email_paths.append(os.path.join(spam_folder_path, filename))
        labels.append(1)  # Spam

    for filename in os.listdir(ham_folder_path):
        email_paths.append(os.path.join(ham_folder_path, filename))
        labels.append(0)  # Ham

    # Zufälliges Mischen der Daten
    combined = list(zip(email_paths, labels))
    random.shuffle(combined)
    email_paths[:], labels[:] = zip(*combined)

    if limit:
        email_paths = email_paths[:limit]
        labels = labels[:limit]

    # Daten in Trainings- und Validierungssets aufteilen
    paths_train, paths_val, labels_train, labels_val = train_test_split(email_paths, labels, test_size=validation_split, random_state=seed)

    def batch_generator(paths, labels, batch_size):
        while True:
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                batch_emails = [read_email(path) for path in batch_paths]
                batch_x = vectorizer.transform(batch_emails).toarray()
                batch_y = to_categorical(batch_labels, num_classes=2)
                yield batch_x, batch_y

    # Erstelle separate Generatoren für Trainings- und Validierungsdaten
    train_gen = batch_generator(paths_train, labels_train, batch_size)
    val_gen = batch_generator(paths_val, labels_val, batch_size)

    return train_gen, val_gen


def plot_to_image(figure):
    """Konvertiert einen matplotlib Plot in ein PNG-Bild und gibt dieses zurück."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def log_confusion_matrix_and_metrics(conf_matrix, test_labels, predicted_labels, file_writer):
    """Loggt die Konfusionsmatrix als Bild in TensorBoard."""
    figure = plt.figure(figsize=(8, 8))
    plt.matshow(conf_matrix, fignum=figure.number, cmap='Blues')
    plt.title('Konfusionsmatrix')
    plt.colorbar()
    plt.xlabel('Vorhergesagte Werte')
    plt.ylabel('Tatsächliche Werte')

    for (i, j), val in np.ndenumerate(conf_matrix):
        plt.text(j, i, f'{val}', ha='center', va='center', color='red')

    image = plot_to_image(figure)

    # Zusätzliche Metriken berechnen
    report = classification_report(test_labels, predicted_labels, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']

    with file_writer.as_default():
        # Logge die Konfusionsmatrix
        tf.summary.image("Konfusionsmatrix", image, step=0)
        # Logge zusätzliche Metriken
        tf.summary.scalar("Precision", precision, step=0)
        tf.summary.scalar("Recall", recall, step=0)
        tf.summary.scalar("F1-Score", f1_score, step=0)


def visualize_study(study):
    # Optimierungsgeschichte
    optimization_history = plot_optimization_history(study)
    optimization_history.show()

    # Wichtigkeit der Hyperparameter
    param_importances = plot_param_importances(study)
    param_importances.show()


# Funktion zum Einlesen einer einzelnen E-Mail
def read_email(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='iso-8859-1') as file:
                return file.read()
        except FileNotFoundError:
            return None


# Funktion zum Laden der Daten aus Textdateien in den Ordnern Spam und Ham mit Upsampling
def load_data(folder_name, seed=12345, limit=None, active_upsampling=False):
    emails, labels = [], []
    spam_folder_path = os.path.join(folder_name, 'spam')
    ham_folder_path = os.path.join(folder_name, 'ham')

    # Laden und begrenzen der Spam-E-Mails
    spam_emails = [read_email(os.path.join(spam_folder_path, filename)) for filename in os.listdir(spam_folder_path) if
                   os.path.isfile(os.path.join(spam_folder_path, filename))]
    spam_labels = [1] * len(spam_emails)

    # Laden und begrenzen der Ham-E-Mails
    ham_emails = [read_email(os.path.join(ham_folder_path, filename)) for filename in os.listdir(ham_folder_path) if
                  os.path.isfile(os.path.join(ham_folder_path, filename))]
    ham_labels = [0] * len(ham_emails)

    # Begrenze die Anzahl der E-Mails, falls ein Limit gesetzt ist
    if limit:
        half_limit = limit // 2
        spam_emails = spam_emails[:half_limit]
        spam_labels = spam_labels[:half_limit]
        ham_emails = ham_emails[:half_limit]
        ham_labels = ham_labels[:half_limit]

    # Kombinieren der Ham- und Spam-E-Mails
    emails = spam_emails + ham_emails
    labels = spam_labels + ham_labels

    # Zufälliges Mischen der Daten, wenn ein Seed-Wert angegeben ist
    if seed is not None:
        random.seed(seed)
        combined = list(zip(emails, labels))
        random.shuffle(combined)
        emails[:], labels[:] = zip(*combined)

    return emails, labels


# Konvertiere Labels in numerische Werte
def encode_labels(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)


# Modifizierte create_model Funktion, um Parameter aus Optuna zu übernehmen
def create_model(input_dim, n_layers, n_units, optimizer='adam'):
    print("input_dim")
    print(input_dim)
    model = Sequential()
    model.add(Dense(n_units, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    for _ in range(n_layers - 1):
        model.add(Dense(n_units, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))  # 2 Klassen: Spam und Ham
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Optuna-Objektivfunktion
def objective(trial, folder_name, vectorizer, seed=12345, total_data_size=0):
    # Hyperparameter
    n_layers = trial.suggest_int('n_layers', 1, 10)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_units = trial.suggest_categorical('n_units', [32, 64, 128, 256, 512, 1024])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam'])

    # Modell erstellen
    input_dim = len(vectorizer.get_feature_names_out())
    model = create_model(input_dim, n_layers, n_units, optimizer=optimizer_name)

    # Generatoren für das Training und die Validierung erstellen
    train_gen, val_gen = create_email_generators(folder_name, vectorizer, batch_size=batch_size, validation_split=0.2, seed=seed)

    steps_per_epoch = int(total_data_size * 0.8 // batch_size)
    validation_steps = int(total_data_size * 0.2 // batch_size)

    # Training
    history = model.fit(train_gen,
                        epochs=100,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        verbose=0,
                        validation_data=val_gen,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])


    # Evaluation
    _, accuracy = model.evaluate(val_gen, steps=validation_steps, verbose=0)
    return accuracy


# Trainiere oder lade das neuronale Netzwerkmodell
def train_or_load_nn_model(model_name, folder_name, vectorizer, tensorboard_log_dir, optuna_db_path, seed=12345, total_data_size=0):
    model_path = model_name + '.keras'
    if os.path.exists(model_path):
        print("Lade vorhandenes neuronales Netzwerkmodell...")
        return load_model(model_path)

    print("Optimiere Hyperparameter...")

    # Pruner-Konfiguration
    pruner = MedianPruner()
    # Erstellen der Studie mit Pruner
    study = optuna.create_study(study_name=optuna_db_path, storage=optuna_db_path, load_if_exists=True, direction='maximize', pruner=pruner)

    def objective_wrapper(trial):
        return objective(trial, folder_name, vectorizer, seed, total_data_size)

    # Callback für den Fortschritt
    def progress_callback(study, trial):
        print(f"Trial {trial.number} abgeschlossen: {trial.value}")

    # Studie mit Callback starten
    study.optimize(objective_wrapper, n_trials=50, timeout=60*60*24*3, callbacks=[progress_callback])

    best_params = study.best_trial.params
    print(f"Beste Parameter: {best_params}")

    # Modell mit den besten Hyperparametern trainieren
    print("Trainiere neues neuronales Netzwerkmodell mit besten Hyperparametern...")
    model = create_model(len(vectorizer.get_feature_names_out()), best_params['n_layers'], best_params['n_units'], best_params['optimizer'])

    # Trainings- und Validierungsgeneratoren vorbereiten
    train_gen, val_gen = create_email_generators(folder_name, vectorizer, batch_size=best_params['batch_size'], validation_split=0.2, seed=seed)

    # TensorBoard Callback konfigurieren
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    steps_per_epoch = int(total_data_size * 0.8 // best_params['batch_size'])
    validation_steps = int(total_data_size * 0.2 // best_params['batch_size'])

    # Modell mit dem Generator trainieren
    model.fit(train_gen, epochs=100, steps_per_epoch=steps_per_epoch,
              validation_data=val_gen, validation_steps=validation_steps,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), tensorboard_callback])

    model.save(model_path)

    return model


def main(check_single_email=False, email_path=None, model_name_='defaultname', vectorizer_name_='defaultvektorizername'):
    seed = 12345
    active_upsampling = False
    limits = [0]  # Diese Zeile kann angepasst werden, um verschiedene Limits zu testen

    optuna_db_path = f"sqlite:///Model_NN_batched_Ergebnisse/_optuna_studies/{model_name_}.db"

    folder_name = '__mails_cleaned_new/mails/'  # Pfad zum Ordner mit den E-Mail-Daten

    for limit in limits:
        model_name = f'nn_spam_classifier_{model_name_}_{limit}'
        vectorizer_name = f'Model_NN_batched_Ergebnisse/nn_vectorizer_{vectorizer_name_}_{limit}.joblib'
        tensorboard_log_dir = f"Model_NN_batched_Ergebnisse/tensorboard_logs/{model_name}"
        results_file_path = f"Model_NN_batched_Ergebnisse/test_results_notCleaned_{model_name}.txt"
        model_path = f'Model_NN_batched_Ergebnisse/{model_name}.keras'


        os.makedirs("Model_NN_batched_Ergebnisse/_optuna_studies", exist_ok=True)
        os.makedirs("Model_NN_batched_Ergebnisse/tensorboard_logs", exist_ok=True)
        os.makedirs("Model_NN_batched_Ergebnisse", exist_ok=True)

        emails, labels = load_data(folder_name, seed, None,
                                   active_upsampling)

        # Vektorisierer laden oder trainieren
        if os.path.exists(vectorizer_name):
            vectorizer = load(vectorizer_name)
        else:
            vectorizer = TfidfVectorizer()
            vectorizer.fit(emails)
            dump(vectorizer, vectorizer_name)

        total_data_size = len(emails)

        # Modell laden oder trainieren
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            # Training und Evaluierung
            model = train_or_load_nn_model(model_name, folder_name, vectorizer, tensorboard_log_dir, optuna_db_path, seed, total_data_size)

        # Testmodell evaluieren,
        test_model(model, vectorizer, results_file_path, tensorboard_log_dir)

        # Klassifiziere eine einzelne E-Mail, falls gewünscht
        if check_single_email and email_path:
            result = classify_single_email(email_path, model, vectorizer)
            print("Klassifizierung der E-Mail:", result)



# Funktion zum Laden des Modells und des Vektorisierers
def load_model_and_vectorizer(model_name, vectorizer_name):
    model = load_model(model_name + '.keras')
    vectorizer = load(vectorizer_name)
    return model, vectorizer


# Funktion zur Klassifizierung einer einzelnen E-Mail
def classify_single_email(email_path, model, vectorizer):
    email_text = read_email(email_path)
    email_vectorized = vectorizer.transform([email_text]).toarray()
    prediction = model.predict(email_vectorized)
    class_label = 'Ham' if prediction[0][0] > prediction[0][1] else 'Spam'
    return class_label


# Funktion zum Testen des Modells
def test_model(model, vectorizer, results_file_path, tensorboard_log_dir):
    seed = 12345
    limit = 0  # Kein Limit für Testdaten
    active_upsampling = False

    test_emails, test_labels = load_data('__mails_cleaned_new_testdata/mails/', seed, limit, active_upsampling)
    test_emails_vectorized = vectorizer.transform(test_emails).toarray()

    predictions = model.predict(test_emails_vectorized)
    predicted_labels = [np.argmax(prediction) for prediction in predictions]

    accuracy = accuracy_score(test_labels, predicted_labels)
    print("Testdaten Genauigkeit:", accuracy)

    # Konfusionsmatrix berechnen
    conf_matrix = confusion_matrix(test_labels, predicted_labels)

    # TensorBoard File Writer
    test_log_dir = os.path.join(tensorboard_log_dir, 'testdatensatz_notCleaned')
    test_writer = tf.summary.create_file_writer(test_log_dir)

    # Loggt die Konfusionsmatrix in TensorBoard
    log_confusion_matrix_and_metrics(conf_matrix, test_labels, predicted_labels, test_writer)

    # Ergebnisse in eine Datei schreiben
    with open(results_file_path, "a") as file:
        file.write(f"Testdaten Genauigkeit bei Limit {limit}: {accuracy}\n")
        file.write("Konfusionsmatrix:\n")
        np.savetxt(file, conf_matrix, fmt="%d")



if __name__ == "__main__":
    model_name = f'BatchedEmails'
    vectorizer_name = f'BatchedEmails.joblib'
    main(model_name_=model_name, vectorizer_name_=vectorizer_name)
