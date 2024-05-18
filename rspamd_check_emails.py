import subprocess
import os

def evaluate_email(filepath):
    """Sendet den E-Mail-Inhalt an Rspamd über rspamc und gibt das Ergebnis zurück."""
    try:
        result = subprocess.run(
            ['docker', 'exec', 'rspamd', 'rspamc', filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None

def is_spam(output):
    """Analysiert die Ausgabe von Rspamd, um festzustellen, ob die E-Mail als Spam eingestuft wurde."""
    print(output)
    return "Spam: true" in output

def load_emails_from_directory(directory, expected_spam):
    """Lädt alle E-Mail-Dateien aus einem Verzeichnis und wertet sie aus."""
    results = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            print(f"Verarbeite Datei: {filename}")
            result = evaluate_email(filepath)
            is_spam_mail = is_spam(result)
            results.append((filename, is_spam_mail, expected_spam))
            print(f"{filename}: {'Spam' if is_spam_mail else 'Ham'}")
    return results

def create_confusion_matrix(results):
    """Erstellt eine Konfusionsmatrix aus den Ergebnissen."""
    tp = sum(1 for _, is_spam, expected_spam in results if is_spam and expected_spam)
    fp = sum(1 for _, is_spam, expected_spam in results if is_spam and not expected_spam)
    tn = sum(1 for _, is_spam, expected_spam in results if not is_spam and not expected_spam)
    fn = sum(1 for _, is_spam, expected_spam in results if not is_spam and expected_spam)
    return tp, fp, tn, fn

if __name__ == '__main__':
    spam_dir = "__mails_new_testdata/mails/spam"
    ham_dir = "__mails_new_testdata/mails/ham"

    spam_results = load_emails_from_directory(spam_dir, True)
    ham_results = load_emails_from_directory(ham_dir, False)

    all_results = spam_results + ham_results
    tp, fp, tn, fn = create_confusion_matrix(all_results)
    print(f"Konfusionsmatrix:\nTP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
