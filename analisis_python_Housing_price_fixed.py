# Análisis de Precios de Viviendas - Versión Corregida
# Instalar librerías necesarias: pip install pandas numpy matplotlib seaborn scipy scikit-learn missingno

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.preprocessing import label_binarize
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficas
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Cargar datos
df = pd.read_csv('Housing_price_prediction.csv')

print("="*60)
print("ANÁLISIS EXPLORATORIO DE DATOS - PRECIOS DE VIVIENDAS")
print("="*60)

# 1. Descripción del conjunto de datos
print("\n1. DESCRIPCIÓN DEL CONJUNTO DE DATOS:")
print(f"Dimensiones: {df.shape[0]} observaciones, {df.shape[1]} variables")
print(f"\nTipos de variables:")
print(df.dtypes)
print(f"\nPrimeras 5 filas:")
print(df.head())

# Verificar datos faltantes
print(f"\nDatos faltantes por columna:")
print(df.isnull().sum())

# 2. Análisis de la variable objetivo (price)
plt.figure(figsize=(15, 5))

# Histograma de precios
plt.subplot(1, 3, 1)
plt.hist(df['price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribución de Precios', fontsize=12, fontweight='bold')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

# Boxplot de precios
plt.subplot(1, 3, 2)
plt.boxplot(df['price'])
plt.title('Boxplot de Precios', fontsize=12, fontweight='bold')
plt.ylabel('Precio')
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Q-Q Plot
plt.subplot(1, 3, 3)
stats.probplot(df['price'], dist="norm", plot=plt)
plt.title('Q-Q Plot - Normalidad de Precios', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('analisis_precios.png', dpi=300, bbox_inches='tight')
plt.show()

# Test de normalidad
stat, p_value = stats.shapiro(df['price'][:5000])  # Shapiro-Wilk para muestra
print(f"\nTest de Shapiro-Wilk (muestra de 5000): estadístico={stat:.4f}, p-value={p_value:.2e}")

# 3. Análisis de variables numéricas
numeric_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
print(f"\n2. ESTADÍSTICAS DESCRIPTIVAS - VARIABLES NUMÉRICAS:")
print(df[numeric_cols].describe())

# Matriz de correlación
plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
            center=0, square=True, fmt='.3f', cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación - Variables Numéricas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Análisis de variables categóricas
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                   'airconditioning', 'prefarea', 'furnishingstatus']

plt.figure(figsize=(18, 12))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 3, i)
    counts = df[col].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    bars = plt.bar(counts.index, counts.values, color=colors, alpha=0.8, edgecolor='black')
    plt.title(f'Distribución: {col.replace("_", " ").title()}', fontweight='bold')
    plt.xlabel(col.replace("_", " ").title())
    plt.ylabel('Frecuencia')
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('variables_categoricas.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Relación entre precio y variables categóricas clave
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Precio vs Estado de amueblado
sns.boxplot(data=df, x='furnishingstatus', y='price', ax=axes[0,0])
axes[0,0].set_title('Precio por Estado de Amueblado', fontweight='bold')
axes[0,0].tick_params(axis='y', labelsize=8)
axes[0,0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Precio vs Aire acondicionado
sns.boxplot(data=df, x='airconditioning', y='price', ax=axes[0,1])
axes[0,1].set_title('Precio por Aire Acondicionado', fontweight='bold')
axes[0,1].tick_params(axis='y', labelsize=8)
axes[0,1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Precio vs Área preferencial
sns.boxplot(data=df, x='prefarea', y='price', ax=axes[1,0])
axes[1,0].set_title('Precio por Área Preferencial', fontweight='bold')
axes[1,0].tick_params(axis='y', labelsize=8)
axes[1,0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Precio vs Carretera principal
sns.boxplot(data=df, x='mainroad', y='price', ax=axes[1,1])
axes[1,1].set_title('Precio por Acceso a Carretera Principal', fontweight='bold')
axes[1,1].tick_params(axis='y', labelsize=8)
axes[1,1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig('precio_vs_categoricas.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Análisis de relaciones entre variables numéricas
plt.figure(figsize=(15, 10))

# Precio vs Área
plt.subplot(2, 3, 1)
plt.scatter(df['area'], df['price'], alpha=0.6, color='coral')
plt.xlabel('Área (sq ft)')
plt.ylabel('Precio')
plt.title('Precio vs Área', fontweight='bold')
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Precio vs Habitaciones
plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='bedrooms', y='price')
plt.title('Precio vs Número de Habitaciones', fontweight='bold')
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Precio vs Baños
plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='bathrooms', y='price')
plt.title('Precio vs Número de Baños', fontweight='bold')
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Precio vs Pisos
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='stories', y='price')
plt.title('Precio vs Número de Pisos', fontweight='bold')
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Precio vs Estacionamientos
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='parking', y='price')
plt.title('Precio vs Estacionamientos', fontweight='bold')
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Distribución de área
plt.subplot(2, 3, 6)
plt.hist(df['area'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('Área (sq ft)')
plt.ylabel('Frecuencia')
plt.title('Distribución del Área', fontweight='bold')

plt.tight_layout()
plt.savefig('relaciones_numericas.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Análisis de Componentes Principales (PCA)
print(f"\n3. ANÁLISIS DE COMPONENTES PRINCIPALES:")

# Preparar datos para PCA
le = LabelEncoder()
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df[col])

# Estandarizar datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

# Aplicar PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Gráficas de PCA
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Varianza explicada
axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_.cumsum(), marker='o', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Componentes')
axes[0].set_ylabel('Varianza Explicada Acumulada')
axes[0].set_title('Varianza Explicada por Componentes Principales', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Biplot - Contribución de variables
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
axes[1].scatter(loadings[:, 0], loadings[:, 1], alpha=0.7, s=100)
for i, col in enumerate(df_encoded.columns):
    axes[1].annotate(col, (loadings[i, 0], loadings[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
axes[1].set_title('Biplot - Contribución de Variables', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_pca.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Varianza explicada por los primeros 3 componentes: {pca.explained_variance_ratio_[:3].sum():.3f}")

# 8. Modelos de Clasificación
print(f"\n4. MODELOS DE CLASIFICACIÓN:")

# Preparar datos
X = df_encoded.drop('furnishingstatus', axis=1)
y = df_encoded['furnishingstatus']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Modelo 1: Todas las variables
model1 = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

# Modelo 2: Selección de características (RFE)
selector = RFE(LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000), 
               n_features_to_select=6)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

model2 = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model2.fit(X_train_selected, y_train)
y_pred2 = model2.predict(X_test_selected)

# Métricas de evaluación
metrics_data = {
    'Modelo': ['Todas las variables', 'Selección de características'],
    'Accuracy': [accuracy_score(y_test, y_pred1), accuracy_score(y_test, y_pred2)],
    'Precision': [precision_score(y_test, y_pred1, average='weighted'), 
                  precision_score(y_test, y_pred2, average='weighted')],
    'Recall': [recall_score(y_test, y_pred1, average='weighted'), 
               recall_score(y_test, y_pred2, average='weighted')],
    'F1-Score': [f1_score(y_test, y_pred1, average='weighted'), 
                 f1_score(y_test, y_pred2, average='weighted')]
}

metrics_df = pd.DataFrame(metrics_data)
print("\nComparación de Modelos:")
print(metrics_df.round(4))

# Visualización de resultados de modelos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Matrices de confusión
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)

sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
axes[0,0].set_title('Matriz de Confusión - Modelo 1\n(Todas las variables)', fontweight='bold')
axes[0,0].set_xlabel('Predicción')
axes[0,0].set_ylabel('Valor Real')

sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[0,1])
axes[0,1].set_title('Matriz de Confusión - Modelo 2\n(Selección de características)', fontweight='bold')
axes[0,1].set_xlabel('Predicción')
axes[0,1].set_ylabel('Valor Real')

# Comparación de métricas
metrics_plot = metrics_df.set_index('Modelo')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
metrics_plot.plot(kind='bar', ax=axes[1,0], width=0.8)
axes[1,0].set_title('Comparación de Métricas de Rendimiento', fontweight='bold')
axes[1,0].set_ylabel('Puntuación')
axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1,0].tick_params(axis='x', rotation=45)

# Importancia de características (Modelo 2)
feature_names = X.columns[selector.support_]
feature_importance = np.abs(model2.coef_).mean(axis=0)
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=True)

axes[1,1].barh(importance_df['Feature'], importance_df['Importance'], color='orange', alpha=0.7)
axes[1,1].set_title('Importancia de Características\n(Modelo Seleccionado)', fontweight='bold')
axes[1,1].set_xlabel('Importancia')

plt.tight_layout()
plt.savefig('resultados_modelos.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nCaracterísticas seleccionadas por RFE:")
for i, feature in enumerate(feature_names):
    print(f"{i+1}. {feature}")

print(f"\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)