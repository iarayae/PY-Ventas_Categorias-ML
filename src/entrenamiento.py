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