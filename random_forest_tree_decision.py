import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predicción de Fuga de Clientes",
    page_icon="🛡️",
    layout="wide"
)

# --- Simulación de Datos ---
# Usamos st.cache_data para que los datos se generen solo una vez.
@st.cache_data
def load_data():
    """Genera un conjunto de datos simulado para la fuga de clientes de seguros."""
    np.random.seed(42)
    num_customers = 1000
    data = {
        'age': np.random.randint(22, 65, num_customers),
        'tenure_years': np.random.randint(0, 25, num_customers),
        'annual_premium': np.random.normal(1200, 400, num_customers).round(2),
        'policy_type': np.random.choice(['Auto', 'Home', 'Health'], num_customers, p=[0.5, 0.3, 0.2]),
        'customer_service_calls': np.random.randint(0, 6, num_customers),
        'premium_increase_last_year': np.random.choice([0, 5, 10, 15, 20], num_customers, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    }
    df = pd.DataFrame(data)

    # Lógica de negocio simulada para la probabilidad de fuga
    churn_probability = (
        (df['premium_increase_last_year'] * 0.02) +
        (df['customer_service_calls'] * 0.03) -
        (df['tenure_years'] * 0.005) +
        (df['age'] / 1000)
    )
    df['churn'] = (churn_probability + np.random.normal(0, 0.1, num_customers)) > 0.3
    df['churn'] = df['churn'].astype(int)

    # Convertir variables categóricas a dummies
    df = pd.get_dummies(df, columns=['policy_type'], drop_first=True)
    return df

# --- Entrenamiento del Modelo ---
# Usamos st.cache_resource para que el modelo se entrene solo una vez por tipo.
@st.cache_resource
def train_model(model_name, X_train, y_train):
    """Entrena un modelo de clasificación específico."""
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    
    model.fit(X_train, y_train)
    return model

# --- Función para la visualización de SHAP ---
# Esta función se usa para renderizar los gráficos SHAP en Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- Aplicación Principal ---
st.title("🛡️ Predicción e Interpretación de Fuga de Clientes de Seguros")

# Carga y preparación de datos
df = load_data()
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- Barra Lateral de Navegación ---
st.sidebar.title("Controles y Navegación")
app_mode = st.sidebar.selectbox(
    "Elige una sección",
    ["Introducción", "Entrenamiento y Evaluación del Modelo", "Interpretación Global del Modelo", "Predicción Individual"]
)

# --- Lógica de las Secciones ---

if app_mode == "Introducción":
    st.header("El Problema de Negocio: Retención de Clientes")
    st.write("""
    La retención de clientes es crucial para cualquier compañía de seguros. Perder un cliente no solo significa una pérdida de ingresos, 
    sino también costos asociados a la adquisición de nuevos clientes. Esta aplicación utiliza modelos de Machine Learning para predecir 
    qué clientes tienen una alta probabilidad de abandonar la compañía (churn).
    
    Además de predecir, la aplicación se enfoca en **interpretar** por qué el modelo toma ciertas decisiones, permitiendo a la empresa
    tomar acciones proactivas y personalizadas para retener a sus clientes valiosos.
    """)
    st.subheader("Datos Simulados de Clientes")
    st.dataframe(df.head())

elif app_mode == "Entrenamiento y Evaluación del Modelo":
    st.header("Entrenamiento y Evaluación del Modelo")
    
    model_choice = st.selectbox("Elige un modelo para entrenar:", ["Decision Tree", "Random Forest"], key="model_selector")
    
    # Entrenar el modelo seleccionado
    model = train_model(model_choice, X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.subheader(f"Rendimiento de {model_choice}")
    col1, col2 = st.columns(2)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("AUC Score", f"{auc:.2f}")

    st.text("Reporte de Clasificación:")
    st.text_area("classification_report", classification_report(y_test, y_pred), height=200)

    # CORRECCIÓN: Guardar el modelo elegido en el estado de la sesión para usarlo en otras pestañas
    st.session_state['selected_model'] = model
    st.session_state['model_name'] = model_choice

    if model_choice == "Decision Tree":
        st.subheader("Visualización del Árbol de Decisión")
        fig, ax = plt.subplots(figsize=(25, 15))
        plot_tree(model, feature_names=X.columns.tolist(), class_names=['No Fuga', 'Fuga'], 
                  filled=True, rounded=True, fontsize=10)
        st.pyplot(fig)

elif app_mode == "Interpretación Global del Modelo":
    st.header("Interpretación Global del Modelo")

    # CORRECCIÓN: Usar el modelo seleccionado en la pestaña anterior.
    if 'selected_model' not in st.session_state:
        st.warning("Por favor, primero entrena un modelo en la pestaña 'Entrenamiento y Evaluación del Modelo'.")
        st.stop()
        
    model = st.session_state['selected_model']
    model_name = st.session_state['model_name']
    st.info(f"Mostrando interpretaciones para el modelo: **{model_name}**")

    st.subheader("1. Importancia de las Características (Feature Importance)")
    st.write("Este gráfico muestra qué características tienen el mayor impacto general en las predicciones del modelo.")
    
    # La importancia de características es nativa de los modelos de árbol.
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=True)
        
        fig, ax = plt.subplots()
        ax.barh(importances['feature'], importances['importance'])
        ax.set_xlabel("Importancia")
        ax.set_title("Importancia de Características Global")
        st.pyplot(fig)
    else:
        st.write("Este modelo no soporta la importancia de características de forma nativa.")

    st.subheader("2. Gráfico de Resumen SHAP (Beeswarm)")
    st.write("Este gráfico muestra no solo qué características son importantes, sino también cómo impactan en la predicción (si aumentan o disminuyen la probabilidad de fuga).")
    
    # Usar un subconjunto de datos para que SHAP sea más rápido
    X_sample = X_test.sample(100, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # CORRECCIÓN: Seleccionar explícitamente los valores SHAP para la clase positiva (Churn=1)
    # El objeto shap_values tiene dimensiones (muestras, características, clases)
    # Para el gráfico de resumen, nos enfocamos en el impacto hacia la clase "Fuga" (índice 1)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[:,:,1], X_sample, show=False)
    st.pyplot(fig)
    

elif app_mode == "Predicción Individual":
    st.header("Predicción y Explicación para un Cliente Individual")

    # CORRECCIÓN: Usar el modelo seleccionado previamente.
    if 'selected_model' not in st.session_state:
        st.warning("Por favor, primero entrena un modelo en la pestaña 'Entrenamiento y Evaluación del Modelo'.")
        st.stop()

    model = st.session_state['selected_model']
    model_name = st.session_state['model_name']
    st.info(f"Realizando predicción con el modelo: **{model_name}**")

    st.sidebar.subheader("Perfil del Cliente:")
    age = st.sidebar.slider("Edad", 20, 70, 45)
    tenure_years = st.sidebar.slider("Antigüedad (Años)", 0, 30, 5)
    annual_premium = st.sidebar.slider("Prima Anual ($)", 500, 3000, 1100)
    customer_service_calls = st.sidebar.slider("Llamadas a Soporte (Último Año)", 0, 10, 2)
    premium_increase_last_year = st.sidebar.selectbox("Aumento de Prima (%)", [0, 5, 10, 15, 20])
    policy_type = st.sidebar.selectbox("Tipo de Póliza", ['Auto', 'Home', 'Health'])

    # CORRECCIÓN: Crear el DataFrame de entrada de forma robusta
    input_data = {
        'age': age, 
        'tenure_years': tenure_years, 
        'annual_premium': annual_premium,
        'customer_service_calls': customer_service_calls, 
        'premium_increase_last_year': premium_increase_last_year,
        'policy_type_Home': policy_type == 'Home', 
        'policy_type_Health': policy_type == 'Health'
    }
    # Asegurar que el orden de las columnas sea el mismo que en el entrenamiento
    input_df = pd.DataFrame([input_data])[X_train.columns]

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("Resultado de la Predicción")
    if prediction == 1:
        st.error(f"**Predicción: FUGA (CHURN)** (Probabilidad: {prediction_proba[1]:.2%})")
    else:
        st.success(f"**Predicción: RETENER** (Probabilidad de Fuga: {prediction_proba[1]:.2%})")
        
    st.subheader("¿Por qué se hizo esta predicción? (Explicación SHAP)")
    
    st.write("""
    Este gráfico de 'cascada' (waterfall) muestra cómo cada característica empuja la predicción desde un valor base hasta el resultado final.
    - Las características en **rojo** empujan la predicción hacia arriba (hacia 'Fuga').
    - Las características en **azul** empujan la predicción hacia abajo (hacia 'Retener').
    """)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)

    # Para el gráfico de cascada, también nos enfocamos en la clase "Fuga" (índice 1)
    # y la primera (y única) muestra (índice 0).
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[0,:,1], show=False)
    st.pyplot(fig)
