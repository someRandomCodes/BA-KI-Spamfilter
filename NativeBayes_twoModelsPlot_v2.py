import random
import email.policy
import os
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.utils import resample
import numpy as np

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
    # Liste für E-Mails und Labels
    emails, labels = [], []

    # Pfade zu den Ordnern Spam und Ham
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

    # Wenn ein Seed-Wert angegeben ist, setze den Seed für den Zufallsgenerator
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



# Funktion zum Trainieren des Modells und Plotten der Lernkurve mit Cross-Validation und Hyperparameter-Tuning
def train_model_and_plot_learning_curve(model_name, save_plot=False, path='__mails_cleaned_new/mails/'):
    model_path_tfidf = model_name + '_cv_cleaned_tfidf.joblib'
    vectorizer_path_tfidf = model_name + '_cv_cleaned_tfidf_vectorizer.joblib'
    model_path_binary = model_name + '_cv_cleaned_binary.joblib'
    vectorizer_path_binary = model_name + '_cv_cleaned_binary_vectorizer.joblib'

    emails, labels = load_data(path)

    X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer()
    binary_vectorizer = CountVectorizer(binary=True)

    best_models = {}
    vectorizers = {
        'tfidf': tfidf_vectorizer,
        'binary': binary_vectorizer
    }

    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}

    for key in vectorizers.keys():
        print(f"Trainiere das {key.upper()} Modell...")
        X_train_vec = vectorizers[key].fit_transform(X_train)
        X_test_vec = vectorizers[key].transform(X_test)

        grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
        grid_search.fit(X_train_vec, y_train)

        best_model = grid_search.best_estimator_
        print(f"Bestes Modell für {key}: ", best_model)
        accuracy = best_model.score(X_test_vec, y_test)
        print(f"Genauigkeit des besten {key.upper()} Modells: {accuracy}")

        dump(best_model, model_path_tfidf if key == 'tfidf' else model_path_binary)
        dump(vectorizers[key], vectorizer_path_tfidf if key == 'tfidf' else vectorizer_path_binary)

        best_models[key] = best_model

    # Lernkurven plotten
    colors = {'tfidf': 'r', 'binary': 'g'}
    line_styles = {'train': '-', 'test': '--'}
    plt.figure(figsize=(10, 5))
    train_sizes = np.linspace(0.1, 1.0, 5)
    for model_key in best_models:
        train_sizes, train_scores, test_scores = learning_curve(
            best_models[model_key],
            vectorizers[model_key].transform(emails),
            labels,
            train_sizes=train_sizes,
            cv=5
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        # Trainings-Scores plotten
        plt.plot(train_sizes, train_scores_mean, 'o-', color=colors[model_key], linestyle=line_styles['train'],
                 label=f"Training ({model_key})")
        # Test-Scores plotten
        plt.plot(train_sizes, test_scores_mean, 'o-', color=colors[model_key], linestyle=line_styles['test'],
                 label=f"Cross-validation ({model_key})")

    plt.title("Lernkurven für TF-IDF und Binärmodelle")
    plt.xlabel("Trainingsbeispiele")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()

    if save_plot:
        plt.savefig('learning_curve_comparison_nb_cv.png')
    plt.show()

    return best_models['tfidf'], vectorizers['tfidf'], best_models['binary'], vectorizers['binary']



# Funktion zum Klassifizieren neuer E-Mails
def classify_email(model, vectorizer, email_text):
    email_vectorized = vectorizer.transform([email_text])
    return model.predict(email_vectorized)


# Modellname festlegen und trainieren, falls das Model nicht gefunden wird

model_name = 'spam_classifier_nb'
model_tfidf, vectorizer_tfidf, model_binary, vectorizer_binary = train_model_and_plot_learning_curve(model_name, save_plot=True)

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
