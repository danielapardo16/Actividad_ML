import pandas as pd
import random 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generar el dataset

n = 500
edades, generos, origenes, promedios_academicos =[], [], [] , []
puntaje_admision, promedios_semestrales, niveles_socioeconmicos, becas, aceptaciones, rechazos = [], [], [], [], [], []

for i in range(n):
    edad = random.randint(16, 48)
    edades.append(edad)

    genero = random.randint(0, 1)
    generos.append("M" if genero == 0 else "F")

    origen = random.randint(1, 3)
    if origen == 1:
        origenes.append("Nacional")
    elif origen == 2:
        origenes.append("Regional")
    else:
        origenes.append("Internacional")

    promedio  = round(random.uniform(40, 100), 1)
    promedios_academicos.append(promedio)

    puntaje = round(random.uniform(30, 100), 1)
    puntaje_admision.append(puntaje)

    promedio_semestral = round(random.uniform(0, 5), 2)
    promedios_semestrales.append(promedio_semestral)

    nivel_socioeconmico = random.randint(1, 4)
    niveles_socioeconmicos.append(nivel_socioeconmico)

    beca = random.randint(0, 1)
    becas.append("Si" if beca == 0 else "No")

    aceptacion = random.randint(1, 3)
    if aceptacion == 1:
        aceptaciones.append("ninguna")
    elif aceptacion == 2:
        aceptaciones.append("prestamo")
    else:
        aceptaciones.append("subsidio")

    # Generar rechazos con reglas simples

    probabilidades_rechazo = 0.2  # Probabilidad base de rechazo 
    if promedio_semestral < 2.5:
        probabilidades_rechazo += 0.3
    if becas[-1] == "No":
            probabilidades_rechazo +=0.2 # Aumentar probabilidad si no tiene beca 
    if nivel_socioeconmico == 1: 
                probabilidades_rechazo += 0.1 # Aumentar probabilidad si es de nivel socioeconmico bajo

    r = random.random()
    rechazos.append("Si" if r < probabilidades_rechazo else "No")

df = pd.DataFrame({
    "Edad": edades, 
    "Genero": generos,
    "Origen": origenes,
    "Promedio_Academico": promedios_academicos,
    "Puntaje_Admision": puntaje_admision,
    "Promedio_Semestral": promedios_semestrales,
    "Nivel_Socioeconomico": niveles_socioeconmicos,
    "Beca": becas,
    "Aceptacion": aceptaciones,
    "Rechazo": rechazos
})

# Introduccion de valores nulos y outliers
for _ in range(10):
    idx = random.randint(0, n-1)
    df.loc[idx, "Promedio_Academico"] = None 
for _ in range(5):
    idx = random.randint(0, n-1)
    df.loc[idx, "Edad"] = 60 # Outlier en edad 

# Guardar el dataset en un archivo CSV

df.to_csv("dataset_estudiantes.csv", index=False, encoding="utf-8")
print("Dataset generado y guardado en 'dataset_estudiantes.csv'")

# Preprcesamiento simple 

# Rellenar valores nulos con la media
df["Promedio_Academico"].fillna(df["Promedio_Academico"].mean(), inplace=True)

# Codificacion de variables categoricas 
Label_encoder = LabelEncoder()
for column in ["Genero", "Origen", "Beca", "Aceptacion", "Rechazo"]:
    df[column] = Label_encoder.fit_transform(df[column])

#  Division en train y test
X = df.drop("Rechazo", axis=1)
y = df["Rechazo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento y evaluacion de modelos

# Modelo de Regresion Logistica
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Modelo de Arbol de desicion 
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Evaluacion de modelos

def evaluar_modelo(y_true, y_pred, modelo_nombre):
    print(f"Evaluacion del modelo: {modelo_nombre}")
    print(f"Accuary", round(accuracy_score(y_true, y_pred), 2))
    print(f"Precision", round(precision_score(y_true, y_pred), 2))
    print(f"Recall", round(precision_score(y_true, y_pred, average="binary"), 2))
    print(f"F1-Score", round(f1_score(y_true, y_pred), 2))

evaluar_modelo(y_test, y_pred_log_reg, "Regresion Logistica")
evaluar_modelo(y_test, y_pred_tree, "Arbol de Decision")



