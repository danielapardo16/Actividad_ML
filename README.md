# Proyecto: Clasificacion de Rechazo de Estudiantes 

Este proyecto fue realizado como parte de la actividad de **Aprendizaje Supervisado vs No Supervisado**.
Se genera un dataset sintetico de estudiantes y se aplican dos modelos de Machine Learning para predecir el rechazo.

# Arhivos principales
- 'codigo_dataset.py' -> Contiene el codigo en Python que genera el dataset, lo preprocesa, entrena y evalua los modelos.
- 'dataset_estudiantes.csv' -> Archivo CSV generado automaticamente con los datos sinteticos.
- 'README.md' -> Este documento.

# Tecnologias usadas 
- Python 3.11
- Pandas 
- Scikit-learn 

# Descripcion del dataset 
Cada fila representa un estudiante con estas variables:

-**Edad**
-**Genero** (M/F)
-**Origen** (Nacional, Regional, Internacional)
-**Promedio_Academico**
-**Puntaje_Academico**
-**Promedio_Semestral**
-**Nivel_Socioeconomico** (1 a 4)
-**Beca** (Si/No)
-**Aceptacion** (ninguna, prestamo, subsidio)
-**Rechazo** (Si/No) -> variable objetivo

# Modelos implementados
1. **Regresion Logistica**
2. **Arbol de Decision**


# Ejecucion 

Para ejecutar el codigo:

''bash 
python codigo_dataset.py

Esto generara el dataset y mostrara en consola las metricas de cada modelo

# Resultados esperados

El programa imprimira en la terminal los resultados de ambos modelos.

*PROYECTO REALIZADO POR*
*DANIELA ISABEL PARDO*
*JUAN DONADO*
*CURSO : INGENERIA DE SISTEMAS - 6TO SEMESTRE*

