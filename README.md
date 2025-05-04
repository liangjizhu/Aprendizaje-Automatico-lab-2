# Aprendizaje Automático – Práctica 2: Clustering de Semillas

**Repositorio:** https://github.com/liangjizhu/Aprendizaje-Automatico-lab-2

## Descripción
Este proyecto realiza un análisis de clustering no supervisado sobre un conjunto de datos de semillas.  
Incluye:
1. Comparación de scalers (MinMax, Robust, Standard) con PCA.
2. Reducción a 2D (MinMax + PCA).
3. Clustering con K-Means, jerárquico (ward) y DBSCAN.
4. Evaluación interna (silhouette) y externa (ARI, Davies–Bouldin).
5. Tablas de contingencia y boxplots para interpretación.

## Entorno
Se recomienda usar un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
