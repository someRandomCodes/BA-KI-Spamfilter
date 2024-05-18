import subprocess
import os

def train_email(filepath, learn_type='spam'):
    """Trainiert rspamd, indem es eine E-Mail als Spam oder Ham markiert."""
    command = ['docker', 'exec', 'rspamd', 'rspamc', f'learn_{learn_type}', filepath]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode('utf-8')
        if result.returncode == 0:
            print(f"{filepath} erfolgreich als {learn_type} trainiert.")
        else:
            print(f"Fehler beim Training von {filepath}: {output}")
    except subprocess.CalledProcessError as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

def train_directory(directory, learn_type='spam'):
    """Trainiert rspamd für alle E-Mails in einem Verzeichnis."""
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            print(f"Trainiere Datei: {filename} als {learn_type}")
            train_email(filepath, learn_type)

if __name__ == '__main__':
    spam_dir = "__mails_new/mails/spam"
    ham_dir = "__mails_new/mails/ham"

    # Trainiere rspamd mit Spam-E-Mails
    train_directory(spam_dir, 'spam')

    # Trainiere rspamd mit Ham-E-Mails
    train_directory(ham_dir, 'ham')
