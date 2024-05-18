def count_spam_in_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().lower()  # Konvertierung in Kleinbuchstaben für eine case-insensitive Suche
            return content.count('spam')  # Zähle, wie oft das Wort "spam" vorkommt
    except FileNotFoundError:
        print("Das angegebene Dokument wurde nicht gefunden.")
        return 0
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return 0

# Pfad zur Datei (aktualisieren Sie diesen Pfad entsprechend Ihrer Dateistruktur)
file_path = 'index'

# Zählen Sie, wie oft "spam" vorkommt
spam_count = count_spam_in_document(file_path)
print(f"Das Wort 'spam' kommt im Dokument {spam_count} Mal vor.")
