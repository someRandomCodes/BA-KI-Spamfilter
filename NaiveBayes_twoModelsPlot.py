import os
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import random
from sklearn.utils import resample
import numpy as np
import email
import email.policy


# Funktion zum Einlesen einer einzelnen E-Mail
import email
import email.policy

def read_email(file_path, filtered=False):
    # Versuche zunächst, die Datei mit einer der Kodierungen zu öffnen
    for encoding in ['utf-8', 'iso-8859-1']:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                if filtered:
                    email_message = email.message_from_file(file, policy=email.policy.default)
                    body = ""
                    if email_message.is_multipart():
                        for part in email_message.walk():
                            ctype = part.get_content_type()
                            cdispo = str(part.get('Content-Disposition'))
                            if ctype in ('text/plain', 'text/html') and 'attachment' not in cdispo:
                                body += part.get_payload(decode=True).decode(encoding)
                    else:
                        body = email_message.get_payload(decode=True).decode(encoding)
                    return body
                else:
                    return file.read()
        except UnicodeDecodeError:
            continue

    print(f"Fehler beim Lesen der E-Mail {file_path}: Keine Kodierung erfolgreich.")
    return None

# Funktion zum Laden der Daten aus Textdateien in den Ordnern Spam und Ham mit Upsampling
def load_data(folder_name, seed=12345, active_upsampling=False):

    emails, labels = [], []


    spam_folder_path = os.path.join(folder_name, 'spam')
    ham_folder_path = os.path.join(folder_name, 'ham')

    # Laden der Spam-E-Mails
    for filename in os.listdir(spam_folder_path):
        email_path = os.path.join(spam_folder_path, filename)
        if os.path.isfile(email_path):
            email_content = read_email(email_path, True)
            if email_content:
                emails.append(email_content)
                labels.append(1)  # Spam-E-Mails erhalten das Label 1

    # Laden der Ham-E-Mails
    for filename in os.listdir(ham_folder_path):
        email_path = os.path.join(ham_folder_path, filename)
        if os.path.isfile(email_path):
            email_content = read_email(email_path, True)
            if email_content:
                emails.append(email_content)
                labels.append(0)  # Ham-E-Mails erhalten das Label 0


    if seed is not None:
        random.seed(seed)

    if active_upsampling:
        # Upsampling der unterrepräsentierten Klasse (Ham-Mails)
        ham_emails = [email for email, label in zip(emails, labels) if label == 0]
        spam_emails = [email for email, label in zip(emails, labels) if label == 1]

        # Anzahl der Ham-Mails und Spam-Mails im Trainingsdatensatz
        num_ham = len(ham_emails)
        num_spam = len(spam_emails)

        # Upsampling der Ham-Mails, um die Anzahl an Spam-Mails zu erreichen
        ham_emails_upsampled = resample(ham_emails, replace=True, n_samples=num_spam, random_state=seed)

        # Kombinieren der upgesampleten Ham-Mails mit den Spam-Mails
        emails = spam_emails + ham_emails_upsampled
        labels = [1] * num_spam + [0] * num_spam

    # Zufälliges Mischen der Daten
    combined = list(zip(emails, labels))
    random.shuffle(combined)
    emails[:], labels[:] = zip(*combined)


    print(f"Anzahl der E-Mails mit Label 0 (Ham): {labels.count(0)}")
    print(f"Anzahl der E-Mails mit Label 1 (Spam): {labels.count(1)}")

    return emails, labels


def plot_roc_curve(X_test, y_test, model):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig('learning_curve_roc.png')



# Funktion zum Trainieren des Modells und Plotten der Lernkurve
def train_model_and_plot_learning_curve(model_name, save_plot=False, path='__mails_new/mails/'):
    model_path_tfidf = model_name + '_tfidf.joblib'
    vectorizer_path_tfidf = model_name + '_tfidf_vectorizer.joblib'

    model_path_binary = model_name + '_binary.joblib'
    vectorizer_path_binary = model_name + '_binary_vectorizer.joblib'

    if os.path.exists(model_path_tfidf) and os.path.exists(vectorizer_path_tfidf) and os.path.exists(model_path_binary) and os.path.exists(vectorizer_path_binary):
        print("Lade vorhandene Modelle und Vektorisierer...")
        model_tfidf = load(model_path_tfidf)
        vectorizer_tfidf = load(vectorizer_path_tfidf)
        model_binary = load(model_path_binary)
        vectorizer_binary = load(vectorizer_path_binary)
    else:
        print("Trainiere neue Modelle...")

        emails, labels = load_data(path)

        # TF-IDF-Ansatz
        vectorizer_tfidf = TfidfVectorizer()
        X_vectorized_tfidf = vectorizer_tfidf.fit_transform(emails)
        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_vectorized_tfidf, labels, test_size=0.2, random_state=42)

        model_tfidf = MultinomialNB()
        train_accuracy_tfidf = []
        test_accuracy_tfidf = []

        for i in range(1, 101):
            subset_size = int(X_train_tfidf.shape[0] * i / 100)
            X_subset, y_subset = X_train_tfidf[:subset_size], y_train[:subset_size]
            model_tfidf.fit(X_subset, y_subset)

            train_accuracy_tfidf.append(accuracy_score(y_subset, model_tfidf.predict(X_subset)))
            test_accuracy_tfidf.append(accuracy_score(y_test, model_tfidf.predict(X_test_tfidf)))

        # Binärer Ansatz
        vectorizer_binary = CountVectorizer(binary=True)
        X_vectorized_binary = vectorizer_binary.fit_transform(emails)
        X_train_binary, X_test_binary, y_train, y_test = train_test_split(X_vectorized_binary, labels, test_size=0.2, random_state=42)

        model_binary = MultinomialNB()
        train_accuracy_binary = []
        test_accuracy_binary = []

        for i in range(1, 101):
            subset_size = int(X_train_binary.shape[0] * i / 100)
            X_subset, y_subset = X_train_binary[:subset_size], y_train[:subset_size]
            model_binary.fit(X_subset, y_subset)

            train_accuracy_binary.append(accuracy_score(y_subset, model_binary.predict(X_subset)))
            test_accuracy_binary.append(accuracy_score(y_test, model_binary.predict(X_test_binary)))

        # Plot der Lernkurven
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, 101), train_accuracy_tfidf, label='Trainingsdaten (TF-IDF)')
        plt.plot(range(1, 101), test_accuracy_tfidf, label='Testdaten (TF-IDF)')
        plt.xlabel('Prozentsatz des Trainingsdatensatzes')
        plt.ylabel('Genauigkeit')
        plt.title('Lernkurve (TF-IDF)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, 101), train_accuracy_binary, label='Trainingsdaten (Binär)')
        plt.plot(range(1, 101), test_accuracy_binary, label='Testdaten (Binär)')
        plt.xlabel('Prozentsatz des Trainingsdatensatzes')
        plt.ylabel('Genauigkeit')
        plt.title('Lernkurve (Binär)')
        plt.legend()
        plt.grid(True)

        dump(model_tfidf, model_path_tfidf)
        dump(vectorizer_tfidf, vectorizer_path_tfidf)
        dump(model_binary, model_path_binary)
        dump(vectorizer_binary, vectorizer_path_binary)

        print("MultinomialNB (TF-IDF):")
        print(classification_report(y_test, model_tfidf.predict(X_test_tfidf)))
        print("Genauigkeit auf Validierungsdaten (TF-IDF):", accuracy_score(y_test, model_tfidf.predict(X_test_tfidf)))

        print("MultinomialNB (Binär):")
        print(classification_report(y_test, model_binary.predict(X_test_binary)))
        print("Genauigkeit auf Validierungsdaten (Binär):", accuracy_score(y_test, model_binary.predict(X_test_binary)))

        if save_plot:
            plt.savefig('learning_curve_comparison.png')

        plot_roc_curve(X_test_tfidf, y_test, model_tfidf)

    return model_tfidf, vectorizer_tfidf, model_binary, vectorizer_binary


# Funktion zum Klassifizieren neuer E-Mails
def classify_email(model, vectorizer, email_text):
    email_vectorized = vectorizer.transform([email_text])
    return model.predict(email_vectorized)



# Modellname festlegen und trainieren, falls das Model nicht gefunden wird
model_name = 'spam_classifier_nb'
path = '__mails_new/mails/'

model_tfidf, vectorizer_tfidf, model_binary, vectorizer_binary = train_model_and_plot_learning_curve(model_name, save_plot=True, path=path)

# Klassifizierung einer neuen E-Mail
# email_path = os.path.join('eml/second.eml')
# email_text = read_email(email_path)

# print("Klassifizierung:", classify_email(model, vectorizer, email_text))


# Lade Testdaten
test_emails, test_labels = load_data('__mails_new_testdata/mails/')

# Überprüfe die Anzahl der geladenen E-Mails
num_ham = np.sum(np.array(test_labels) == 0)
num_spam = np.sum(np.array(test_labels) == 1)

print("Anzahl geladener Ham-E-Mails: ", num_ham)
print("Anzahl geladener Spam-E-Mails: ", num_spam)

# Verhältnis von Ham zu Spam im Testset anzeigen
print(f"Verhältnis von Ham zu Spam im Testset: {num_ham / (num_ham + num_spam):.2f}/{num_spam / (num_ham + num_spam):.2f}")

# Verteilung der Labels im Testset
unique, counts = np.unique(test_labels, return_counts=True)
print("Verteilung der Labels im Testset:", dict(zip(unique, counts)))

# Transformiere die Testdaten und mache Vorhersagen
X_test_tfidf = vectorizer_tfidf.transform(test_emails)
X_test_binary = vectorizer_binary.transform(test_emails)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
y_pred_binary = model_binary.predict(X_test_binary)

# Konfusionsmatrix erstellen
confusion_tfidf = confusion_matrix(test_labels, y_pred_tfidf, labels=[0, 1])
confusion_binary = confusion_matrix(test_labels, y_pred_binary, labels=[0, 1])

# Ergebnisse ausgeben
print("Konfusionsmatrix TF-IDF:")
print(confusion_tfidf)

print("Konfusionsmatrix Binary:")
print(confusion_binary)

# Berechne die Genauigkeit
accuracy_tfidf = accuracy_score(test_labels, y_pred_tfidf)
accuracy_binary = accuracy_score(test_labels, y_pred_binary)

print("Genauigkeit der Vorhersage TF-IDF:", accuracy_tfidf)
print("Genauigkeit der Vorhersage Binary:", accuracy_binary)

# Speichere die Ergebnisse in einer Datei
output_file = "model_evaluation_nb.txt"
with open(output_file, 'w') as f:
    f.write(f"Genauigkeit TF-IDF: {accuracy_tfidf}\n")
    f.write("Konfusionsmatrix TF-IDF:\n")
    f.write(np.array2string(confusion_tfidf))

    f.write(f"\nGenauigkeit Binary: {accuracy_binary}\n")
    f.write("Konfusionsmatrix Binary:\n")
    f.write(np.array2string(confusion_binary))
