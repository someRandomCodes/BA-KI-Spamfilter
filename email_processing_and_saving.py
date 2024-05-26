import os
import re
from collections import Counter
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv
import sqlite3
from langdetect import detect, LangDetectException
from email import message_from_string
from email.message import EmailMessage


def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


def create_database(database_path='./__mails_new_db/_emails.db'):
    database_dir = os.path.dirname(database_path)
    if not os.path.exists(database_dir):
        os.makedirs(database_dir)

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY,
            email TEXT,
            label TEXT,
            is_test INTEGER,
            cleaned INTEGER
        )
    ''')

    connection.commit()
    connection.close()


def save_emails_to_database(emails, labels, cleaned=False, num_emails=27604,
                            database_path='./__mails_new_db/_emails.db'):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # Bestimme die Grenzen für die Aufteilung in Trainings- und Testdaten
    num_ham = labels.count(0)
    num_spam = labels.count(1)
    ham_boundary = num_ham - num_emails
    spam_boundary = num_spam - num_emails

    for idx, (email, label) in enumerate(zip(emails, labels)):
        # Bestimme, ob die E-Mail zu den Testdaten gehört
        is_test = 0
        if label == 0 and idx >= ham_boundary:  # Ham
            is_test = 1
        elif label == 1 and idx >= spam_boundary:  # Spam
            is_test = 1

        is_cleaned = 0
        if cleaned:
            is_cleaned = 1

        # Füge die E-Mail in die Datenbank ein
        cursor.execute('''
            INSERT INTO emails (email, label, is_test, cleaned) VALUES (?, ?, ?, ?)
        ''', (email, 'ham' if label == 0 else 'spam', is_test, is_cleaned))

    connection.commit()
    connection.close()


def save_emails_to_different_folders(emails, labels, num_emails=8810, cleaned=False):
    base_dir = '__mails_new'
    test_dir = '__mails_new_testdata'

    categories = {0: 'ham', 1: 'spam'}
    filename = 'email_'

    if cleaned:
        base_dir = '__mails_cleaned_new'
        test_dir = '__mails_cleaned_new_testdata'
        filename = 'email_cleaned_'

    # Trenne E-Mails nach Labels
    spam_emails = [email for email, label in zip(emails, labels) if label == 1]
    ham_emails = [email for email, label in zip(emails, labels) if label == 0]

    # Bestimme die Grenzen für die Aufteilung
    spam_boundary = len(spam_emails) - num_emails
    ham_boundary = len(ham_emails) - num_emails

    # Ordnerstrukturen erstellen
    for category in categories.values():
        os.makedirs(os.path.join(base_dir, 'mails', category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'mails', category), exist_ok=True)

    # Hilfsfunktion zum Speichern der E-Mails
    def save_email(email, idx, category, is_test):
        folder = test_dir if is_test else base_dir
        file_path = os.path.join(folder, 'mails', category, f'{filename}{category}_{idx}.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(email)

    for idx, email in enumerate(spam_emails):
        save_email(email, idx, 'spam', idx >= spam_boundary)
    for idx, email in enumerate(ham_emails):
        save_email(email, idx, 'ham', idx >= ham_boundary)


def safe_decode(payload):
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
    for enc in encodings:
        try:
            return payload.decode(enc)
        except UnicodeDecodeError:
            continue
    return payload.decode('utf-8', errors='replace')  # Als letzten Ausweg, ersetze unbekannte Zeichen


def clean_email(email_text):
    """Parsen und Bereinigen einer E-Mail, Betreff beibehalten und HTML aus dem Körpertext entfernen."""
    # Parse die E-Mail aus dem String
    email_msg = message_from_string(email_text)

    # Extrahiere das Betreff
    subject = email_msg['Subject'] if email_msg['Subject'] else ""

    # Ermittle den Körpertext
    if email_msg.is_multipart():
        for part in email_msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain' or content_type == 'text/html':
                body = part.get_payload(decode=True)
                body = safe_decode(body)
                break
    else:
        body = email_msg.get_payload(decode=True)
        body = safe_decode(body)

    if '<html>' in body:
        body = BeautifulSoup(body, "html.parser").get_text()

    body = re.sub('\s+', ' ', body).strip()

    cleaned_email = f"Subject: {subject}\n\n{body}"

    return cleaned_email


def read_email(file_path):
    """Read the content of an email file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def load_data(folder_name, clean_html=False):
    """Load email data from a dataset folder with an option to clean HTML."""
    index_path = os.path.join('_mails', 'trec07p', folder_name, 'index')
    emails, labels = [], []

    try:
        with open(index_path, 'r') as file:
            lines = file.readlines()
    except IOError as e:
        print(f"Error opening index file {index_path}: {e}")
        return emails, labels

    for line in lines:
        label, path = line.strip().split(' ')
        email_path = os.path.join('_mails', 'trec07p', 'full', path)
        email_content = read_email(email_path)

        if clean_html and email_content:
            email_content = clean_email(email_content)

        if email_content:
            emails.append(email_content)
            labels.append(0 if label.lower() == 'ham' else 1)

    # Additional loading of spam emails from untroubled_spam archive
    spam_folder_path = os.path.join('_mails', 'untroubled_spam')
    for year_folder in os.listdir(spam_folder_path):
        year_folder_path = os.path.join(spam_folder_path, year_folder)
        if os.path.isdir(year_folder_path):
            for month_folder in os.listdir(year_folder_path):
                month_folder_path = os.path.join(year_folder_path, month_folder)
                if os.path.isdir(month_folder_path):
                    for email_filename in os.listdir(month_folder_path):
                        if email_filename.endswith('.txt'):
                            email_path = os.path.join(month_folder_path, email_filename)
                            email_content = read_email(email_path)

                            if clean_html and email_content:
                                email_content = clean_email(email_content)

                            if email_content:
                                emails.append(email_content)
                                labels.append(1)

    # Zusätzliche Verarbeitung für SpamAssassin-Daten
    spamassassin_folder_path = os.path.join('_mails', 'spamassassin')
    for root, dirs, files in os.walk(spamassassin_folder_path):
        for file in files:

            email_path = os.path.join(root, file)
            email_content = read_email(email_path)

            if clean_html and email_content:
                email_content = clean_email(email_content)

            if email_content:
                last_folder_name = os.path.basename(os.path.normpath(root))
                if 'ham' in last_folder_name:
                    label = 0
                elif 'spam' in last_folder_name:
                    label = 1
                else:
                    continue

                emails.append(email_content)
                labels.append(label)

    return emails, labels


def tokenize(text):
    """Tokenize the text into words."""
    return re.findall(r'\b\w+\b', text.lower())


def find_most_common_words(emails, labels):
    """Find the 50 most common words in spam and non-spam (ham) emails."""
    spam_words = Counter()
    ham_words = Counter()

    for email, label in zip(emails, labels):
        tokens = tokenize(email)
        if label == 1:
            spam_words.update(tokens)
        else:
            ham_words.update(tokens)

    most_common_spam = spam_words.most_common(50)
    most_common_ham = ham_words.most_common(50)

    with open("most_common_words.txt", "w") as file:
        file.write("50 most common words in spam emails:\n")
        for word, count in most_common_spam:
            file.write(f"{word}: {count}\n")
        file.write("\n50 most common words in ham emails:\n")
        for word, count in most_common_ham:
            file.write(f"{word}: {count}\n")

    return most_common_spam, most_common_ham


def find_top_tfidf_words(emails, labels, num_features=50):
    """Find the most significant words in spam and ham emails using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=num_features, stop_words='english')
    X_tfidf = vectorizer.fit_transform(emails)

    spam_indices = np.array(labels) == 1
    ham_indices = np.array(labels) == 0

    spam_tfidf_sum = np.sum(X_tfidf[spam_indices], axis=0)
    ham_tfidf_sum = np.sum(X_tfidf[ham_indices], axis=0)

    feature_names = np.array(vectorizer.get_feature_names_out())

    sorted_spam_indices = spam_tfidf_sum.argsort()[0][::-1]
    sorted_ham_indices = ham_tfidf_sum.argsort()[0][::-1]

    top_spam_words = feature_names[sorted_spam_indices[:num_features]]
    top_ham_words = feature_names[sorted_ham_indices[:num_features]]

    top_spam_tfidf = spam_tfidf_sum[0, sorted_spam_indices[:num_features]]
    top_ham_tfidf = ham_tfidf_sum[0, sorted_ham_indices[:num_features]]

    return list(zip(top_spam_words, top_spam_tfidf)), list(zip(top_ham_words, top_ham_tfidf))


def save_words_to_csv(filename, words, header):
    """Save words and their scores to a CSV file."""
    directory = './_wordcounts'
    filename = os.path.join(directory, filename)

    # Erstelle den Ordner, falls er nicht existiert
    os.makedirs(directory, exist_ok=True)

    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for word, score in words:
            writer.writerow([word, score])


# Main execution
if __name__ == "__main__":
    create_database()

    # Headers for CSV files
    headers = ["Word", "TF-IDF Score"]
    headersMostCommon = ["Word", "Count"]

    # Process with HTML cleaning
    print("Processing with HTML cleaning...")
    cleaned_emails, labels = load_data('full', clean_html=True)

    top_tfidf_spam_cleaned, top_tfidf_ham_cleaned = find_top_tfidf_words(cleaned_emails, labels)
    save_words_to_csv('top_tfidf_spam_words.csv', top_tfidf_spam_cleaned, headers)
    save_words_to_csv('top_tfidf_ham_words.csv', top_tfidf_ham_cleaned, headers)

    most_common_spam, most_common_ham = find_most_common_words(cleaned_emails, labels)
    save_words_to_csv('most_common_spam_words.csv', most_common_spam, headersMostCommon)
    save_words_to_csv('most_common_ham_words.csv', most_common_ham, headersMostCommon)

    save_emails_to_different_folders(cleaned_emails, labels, 4906, True)
    save_emails_to_database(cleaned_emails, labels, cleaned=True)

    # print("E-Mails wurden erfolgreich in den angegebenen Verzeichnissen gespeichert.")

    spam_count = sum(1 for label in labels if label == 1)
    ham_count = sum(1 for label in labels if label == 0)
    print("\n spam_count:  ")
    print(spam_count)
    print("\n  ")
    print("\n ham_count:  ")
    print(ham_count)
    print("\n  ")
    print("\n  ")
    # Process without HTML cleaning
    print("\nProcessing without HTML cleaning...")

    emails, labels = load_data('full', clean_html=False)
    top_tfidf_spam, top_tfidf_ham = find_top_tfidf_words(emails, labels)
    save_words_to_csv('top_tfidf_spam_words_raw.csv', top_tfidf_spam, headers)
    save_words_to_csv('top_tfidf_ham_words_raw.csv', top_tfidf_ham, headers)

    most_common_spam, most_common_ham = find_most_common_words(emails, labels)
    save_words_to_csv('most_common_spam_words.csv', most_common_spam, headersMostCommon)
    save_words_to_csv('most_common_ham_words.csv', most_common_ham, headersMostCommon)

    print("\nResults have been saved to CSV files.")

    save_emails_to_different_folders(emails, labels, 4906, False)
    save_emails_to_database(emails, labels, cleaned=False)
    print("E-Mails wurden erfolgreich in den angegebenen Verzeichnissen gespeichert.")
