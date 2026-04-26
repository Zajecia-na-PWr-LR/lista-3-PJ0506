import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
# THE PILE IS GROWING ALL HAIL THE PILE 
 
#---------------------------------------------- ZAD 1 a) ----------------------------------------

df = pd.read_csv('Heart_disease_cleveland_new.csv')

print("--- Pierwsze 5 wierszy ---")
print(df.head())

print("\n--- Informacje o strukturze danych ---")
df.info()

print("\n--- Podstawowe statystyki opisowe ---")
print(df.describe())

print("\n--- Liczba braków danych ---")
print(df.isnull().sum())

#---------------------------------------------- ZAD 1 b) ----------------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df['age'], kde=True, bins=20, color='skyblue')
plt.title('Rozkład wieku pacjentów')
plt.xlabel('Wiek')
plt.ylabel('Liczba pacjentów')
plt.savefig("Rozkład wieku.png")

plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='pastel')
plt.title('Balans klas (0 = brak choroby, 1 = choroba)')
plt.savefig("Balans.png")

#---------------------------------------------- ZAD 1 c) ----------------------------------------
df = df.dropna()

X = df.drop('target', axis=1)

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

print("\n--- Wyniki podziału ---")
print(f"Rozmiar zbioru treningowego (X_train): {X_train.shape}")
print(f"Rozmiar zbioru testowego (X_test): {X_test.shape}")
print(f"Rozmiar etykiet treningowych (y_train): {y_train.shape}")
print(f"Rozmiar etykiet testowych (y_test): {y_test.shape}")

#---------------------------------------------- ZAD 2 ----------------------------------------

model = LogisticRegression(max_iter=1500, random_state=42)

model.fit(X_train, y_train)
print("DONE")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nDokładność (Accuracy): {accuracy * 100:.2f}%")

print("\nRaport:")
print(classification_report(y_test, y_pred))
print("\n#######################################################################################")

# czulosc op reszta eeee? precyzja ig z braku innych sensownych opcji
# f1 to komunizm malo wazny gdy zycie support to nie miara

#---------------------------------------------- ZAD 3 ----------------------------------------

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Zdrowy (0)', 'Chory (1)'])

plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', ax=plt.gca(), colorbar=False)
plt.title('Macierz Pomyłek')
plt.grid(False)
plt.savefig("Macierz pomyłek.png")

y_pred_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Odsetek fałszywych alarmów (FPR)')
plt.ylabel('Czułość (TPR)')
plt.title('Krzywa ROC (Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.savefig("ROC.png")

#---------------------------------------------- ZAD 4 a) ----------------------------------------

std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train_std, y_train)
print("DONE")

y_pred = model.predict(X_test_std)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nDokładność (Accuracy): {accuracy * 100:.2f}%")

print("\nRaport standaryzacji:")
print(classification_report(y_test, y_pred))
print("\n#######################################################################################")

#---------------------------------------------- ZAD 4 b) ----------------------------------------

minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
# mniejszy max iter w obu przypadkach jest spoko 
model.fit(X_train_minmax, y_train)
print("DONE")

y_pred = model.predict(X_test_minmax)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nDokładność (Accuracy): {accuracy * 100:.2f}%")

print("\nRaport normalizacji:")
print(classification_report(y_test, y_pred))
print("\n#######################################################################################")

#---------------------------------------------- ZAD 5 ----------------------------------------

C_values = [0.01, 0.1, 1.0, 10.0]

l1_ratios = [0.0, 0.5, 1.0]

wyniki = []

print("trwa duzo testowanie czas zaczac\n")
# nie dużo po standaryzacji więc olewamy

for C in C_values:
    for l1_ratio in l1_ratios:
        model = LogisticRegression(
            penalty='elasticnet',
            C=C,
            l1_ratio=l1_ratio,
            solver='saga',
            max_iter=1000,
            random_state=42
        )
        
        model.fit(X_train_std, y_train)
        
        y_pred = model.predict(X_test_std)
        
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        zuzyte_iteracje = model.n_iter_[0]
        
        wyniki.append({
            'C': C,
            'l1_ratio': l1_ratio,
            'Dokładność (%)': round(acc * 100, 2),
            'Czułość (%)': round(recall * 100, 2),
            'Zużyte iteracje': zuzyte_iteracje
        })

tabela_wynikow = pd.DataFrame(wyniki)
print(tabela_wynikow.to_string(index=False))

#---------------------------------------------- ZAD 5 ----------------------------------------

model = make_pipeline(
    StandardScaler(), 
    LogisticRegression(max_iter=1000, random_state=42)
)

model = make_pipeline(
    StandardScaler(), 
    LogisticRegression(max_iter=1000, random_state=42)
)

folds = [2, 5, 10]

print("\n SO IT BEGINS")
print("\n#############################################################################")
for stratify in [False, True]:
    for k in folds:
        if stratify:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=k, shuffle=True, random_state=42)

        scores = cross_validate(model, X, y, cv=cv, scoring=['accuracy', 'recall'])
        
        acc_mean = scores['test_accuracy'].mean() * 100
        acc_std = scores['test_accuracy'].std() * 100
        rec_mean = scores['test_recall'].mean() * 100
        rec_std = scores['test_recall'].std() * 100
        
        print(f"{k:>2}-fold | Dokładność: {acc_mean:>5.2f}% (±{acc_std:>4.2f}%) | Czułość: {rec_mean:>5.2f}% (±{rec_std:>4.2f}%)")