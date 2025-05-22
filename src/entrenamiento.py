import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# === Paso 1: Cargar archivo excel ===
df = pd.read_excel("datos/boletas_detalladas.xlsx")
print(df.head())

# Crear matriz de categorías por boleta
df_categorias = (
    df.groupby(["Boleta", "Categoría"])
    .size()
    .unstack(fill_value=0)
)
df_categorias = (df_categorias > 0).astype(int)
print(df_categorias.head())

# === Paso 2: Mostrar categorías disponibles ===
categorias = sorted(df["Categoría"].unique().tolist())
print("Selecciona la categoría que deseas precedir: \n")
for i, cat in enumerate(categorias):
    print(f"{i + 1}. {cat}")

# Leer opción seleccionada
try:
    opcion = int(input("\nIngresa el número de la categoría: "))
    if opcion < 1 or opcion > len(categorias):
        raise ValueError
    categoria_objetivo = categorias[opcion - 1]
except ValueError:
    print("Opción inválida. Debes ingresar un número entre 1 y", len(categorias))
    exit()

print(f"\nCategoría seleccionada: {categoria_objetivo}\n")

# === Paso 3: Separar las variables ===
X = df_categorias.drop(columns=[categoria_objetivo])
y = df_categorias[categoria_objetivo]

# === Paso 4: Separar grupo de entrenamiento y prueba ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Paso 5: Entrenar el modelo ===
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# === Paso 6: Evaluar modelo ===
y_pred = modelo.predict(X_test)
reporte = classification_report(y_test, y_pred)
