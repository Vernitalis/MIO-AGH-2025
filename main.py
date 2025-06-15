# Importowanie bibliotek
import pandas as pd
import os
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import shap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Pobranie słownika VADER
nltk.download('vader_lexicon')
nltk.download('punkt')

# Sprawdzenie dostępności CUDA i ustawienie urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

# Wczytanie danych
try:
    df = pd.read_csv('trumptweets.csv')
except FileNotFoundError:
    print("Plik 'trumptweets.csv' nie znaleziony. Upewnij się, że plik znajduje się w odpowiednim folderze.")
    exit()

# Wyświetlenie pierwszych kilku wierszy
print(df.head())


# 1. Funkcja do czyszczenia tekstu
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Usuwanie URLi
    text = re.sub(r'@\w+', '', text)  # Usuwanie wzmianek
    text = re.sub(r'#', '', text)  # Usuwanie symbolu #
    text = re.sub(r'RT\s+', '', text)  # Usuwanie 'RT'
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Usuwanie znaków specjalnych i cyfr
    text = text.lower()  # Zamiana na małe litery
    return text


# Zastosowanie czyszczenia do kolumny 'content'
df['cleaned_content'] = df['content'].apply(clean_text)

# 2. Generowanie etykiet sentymentu za pomocą VADER
analyzer = SentimentIntensityAnalyzer()


def get_sentiment_label(text):
    score = analyzer.polarity_scores(text)
    # jeśli 'compound' > 0.05, sentyment jest pozytywny (1), w przeciwnym razie negatywny (0)
    return 1 if score['compound'] > 0.05 else 0


df['sentiment'] = df['cleaned_content'].apply(get_sentiment_label)

print("\nLiczba tweetów w każdej kategorii sentymentu:")
print(df['sentiment'].value_counts())

# 3. Wektoryzacja tekstu za pomocą TF-IDF
X = df['cleaned_content'].values
y = df['sentiment'].values

# Ograniczamy liczbę cech (słów) do 2000 najczęstszych, aby model był prostszy
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Konwersja macierzy rzadkiej do gęstej dla PyTorch
X_dense = X_tfidf.toarray()

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_dense, y, test_size=0.2, random_state=42, stratify=y)

# Konwersja danych do tensorów PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

print(f"\nRozmiar zbioru treningowego: {X_train_tensor.shape}")
print(f"Rozmiar zbioru testowego: {X_test_tensor.shape}")


# Definicja modelu sieci neuronowej
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x


# Inicjalizacja modelu, funkcji straty i optymalizatora
input_dim = X_train_tensor.shape[1]
model = SentimentClassifier(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss()  # Lepsza stabilność numeryczna niż Sigmoid + BCELoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Przygotowanie DataLoaderów
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Pętla treningowa
epochs = 10
print("\nRozpoczynanie treningu...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoka {epoch + 1}/{epochs}, Strata: {total_loss / len(train_loader):.4f}")

# Ewaluacja modelu
model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_tensor)
    y_pred = torch.sigmoid(y_pred_logits) > 0.5
    accuracy = (y_pred.squeeze() == y_test_tensor.squeeze()).float().mean()
    print(f"\nDokładność na zbiorze testowym: {accuracy.item():.4f}")

    # Przygotowanie danych do analizy SHAP
    background_data = X_train[:100]  # Tło do obliczeń SHAP (reprezentatywna próbka)
    test_sample = X_test[:50]  # Próbka do wyjaśnienia


    def predict_pytorch(data):
        if data.ndim == 1:
            data = data.reshape(1, -1)

        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            output = torch.sigmoid(model(data_tensor))

        return output.cpu().numpy().flatten()


    explainer = shap.KernelExplainer(predict_pytorch, background_data)

    print("\nObliczanie wartości SHAP (może to potrwać kilka minut)...")
    shap_values = explainer.shap_values(test_sample)
    print("Obliczanie zakończone.")

    # Zmiana nazwy cech z indeksów na faktyczne słowa
    feature_names = vectorizer.get_feature_names_out()

    shap.initjs()

    if not os.path.exists("imgs"):
        os.makedirs("imgs")

    # Wizualizacja 1: Summary Plot
    print("\nGenerowanie wykresu podsumowującego SHAP (Summary Plot)...")
    shap.summary_plot(shap_values, features=test_sample, feature_names=feature_names, show=False)
    plt.title("Wykres podsumowujący SHAP")
    plt.savefig("imgs/SHAP.png")
    plt.close()

    sample_index = 0

    expected_value = explainer.expected_value
    shap_values_sample = shap_values[sample_index, :]
    features_sample = test_sample[sample_index, :]

    # Wizualizacja 2: Force Plot
    print("\nRysowanie wykresu force_plot...")
    shap.force_plot(expected_value, shap_values_sample, features_sample, feature_names=feature_names)
    print("\nGenerowanie wykresu sił SHAP (Force Plot) dla pierwszej próbki...")
    force_plot_obj = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_index, :],
        test_sample[sample_index, :],
        feature_names=feature_names,
        show=False
    )
    shap.save_html("imgs/force_plot_output.html", force_plot_obj)
