
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Configuración inicial
df_path = "Vienna_Final.csv"
st.set_page_config(page_title="Airbnb en Viena", layout="wide")

# Título y presentación
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Airbnb en Viena 📈")
    st.markdown("""
        <div style='text-align: justify'>
        Viena, la capital de Austria, es conocida por su rica historia cultural, su arquitectura imperial y su vibrante escena musical. 
        La ciudad ha sido el hogar de grandes compositores como Mozart, Beethoven y Schubert. Además, Viena es famosa por sus palacios, 
        museos y su estilo de vida elegante.

        En este dashboard exploraremos diversos aspectos de Viena utilizando datos disponibles, y cómo estos se pueden visualizar y analizar.
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("<div style='padding-top: 80px;'></div>", unsafe_allow_html=True)
    st.image("Vienna.JPG", width=400)

# Estilos
st.markdown("""
<style>
[data-testid="stHeader"] { background-color: #023E8A; }
[data-testid="stSidebar"] { background-color: #CAF0F8; }
button { background-color: #0096C7 !important; color: white !important; }
.stApp h1, .stApp h2, .stApp h3 { color: #03045E; }
.stApp { color: #03045E; background-color: #F5F6FA; }
</style>
""", unsafe_allow_html=True)

# Carga de datos
@st.cache_resource
def cargar_datos():
    data = pd.read_csv(df_path, index_col="property_type")
    columnas_a_eliminar = ["Unnamed: 0", "listing_url", "id", "description", "picture_url", "host_url", "host_picture_url", "host_thumbnail_url"]
    columnas_existentes = [col for col in columnas_a_eliminar if col in data.columns]
    data.drop(columns=columnas_existentes, inplace=True)
    data.replace('', np.nan, inplace=True)
    columnas_numericas = data.select_dtypes(include=['float', 'int']).columns
    data[columnas_numericas] = data[columnas_numericas].replace(0, np.nan)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    columnas_numericas = data.select_dtypes(include=['float', 'int']).columns
    columnas_texto = data.select_dtypes(include=['object']).columns
    opciones_superhost = data['has_availability'].unique()
    return data, columnas_numericas, columnas_texto, opciones_superhost, data[columnas_numericas]

data, columnas_numericas, columnas_texto, opciones_superhost, data_numerica = cargar_datos()

# Sidebar
st.sidebar.title("🔍 Opciones de Visualización")
vista = st.sidebar.selectbox("Selecciona una vista", [
    "Análisis Univariado", "Comportamiento de Variables", "Dispersión", "Gráfico circular", "Barras", "Modelado Predictivo"])

# Vista: Análisis Univariado
if vista == "Análisis Univariado":
    st.header("🔎 Análisis Univariado de Variables Categóricas")
    cat_var = st.sidebar.selectbox("Variable categórica:", options=columnas_texto)
    conteo = data[cat_var].value_counts().reset_index()
    conteo.columns = [cat_var, "Frecuencia"]
    fig = px.bar(conteo, x=cat_var, y="Frecuencia", title=f"Frecuencia de {cat_var}")
    st.plotly_chart(fig)
    st.dataframe(conteo)

# Vista: Comportamiento de Variables
elif vista == "Comportamiento de Variables":
    st.header("Tendencias por Categoría")
    mostrar_df = st.sidebar.checkbox("Mostrar dataset")
    variables_graficar = st.sidebar.multiselect("Variables numéricas:", options=columnas_numericas)
    filtro_superhost = st.sidebar.selectbox("¿Está disponible?", options=opciones_superhost)

    if mostrar_df:
        st.dataframe(data, use_container_width=True)
        st.write("Resumen estadístico:", data.describe())

    if variables_graficar:
        df_filtrado = data[data['has_availability'] == filtro_superhost]
        top_tipos = df_filtrado.index.value_counts().nlargest(4).index
        df_filtrado_top = df_filtrado.loc[top_tipos]

        # Si no se eligen suficientes variables, mostrar un mensaje
        if len(variables_graficar) < 2:
            st.warning("Selecciona al menos dos variables numéricas para realizar la regresión.")

        else:
            # Simulación: Y es la primera variable seleccionada
            y_var = variables_graficar[0]
            x_vars = variables_graficar[1:]

            # Datos base para gráfico Plotly
            long_data = pd.DataFrame()
            for var in x_vars:
                tmp = pd.DataFrame({
                    "index": df_filtrado_top.index,
                    "valor": df_filtrado_top[var],
                    "variable": "X - " + var
                })
                long_data = pd.concat([long_data, tmp], ignore_index=True)

            # Y real
            y_real = pd.DataFrame({
                "index": df_filtrado_top.index,
                "valor": df_filtrado_top[y_var],
                "variable": "Y - " + y_var
            })
            long_data = pd.concat([long_data, y_real], ignore_index=True)

            # Predicción (regresión múltiple real)
            df_modelo = df_filtrado_top.dropna(subset=x_vars + [y_var])
            if df_modelo.empty:
                st.error("No hay suficientes datos para realizar la regresión.")
            else:
                modelo = LinearRegression().fit(df_modelo[x_vars], df_modelo[y_var])
                predicciones = modelo.predict(df_modelo[x_vars])

                y_pred_df = pd.DataFrame({
                    "index": df_modelo.index,
                    "valor": predicciones,
                    "variable": "🔴 Predicción"
                })
                long_data = pd.concat([long_data, y_pred_df], ignore_index=True)

                # Gráfico interactivo con Plotly
                fig = px.line(long_data, x="index", y="valor", color="variable",
                              title="Tendencias con Regresión Múltiple",
                              width=1200, height=500)
                st.plotly_chart(fig)

                # Gráfico de dispersión: Valores Reales vs Predichos
                fig2, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(range(len(df_modelo)), df_modelo[y_var], color='blue', label='Valores Reales (Y)', alpha=0.6)
                ax.scatter(range(len(predicciones)), predicciones, color='red', label='Predicciones (Ŷ)', alpha=0.6)

                ax.set_title("Regresión Múltiple: Valores Reales vs Predichos")
                ax.set_xlabel("Índice de Observación")
                ax.set_ylabel("Valor")
                ax.legend()
                st.pyplot(fig2)


# Vista: Dispersión
elif vista == "Dispersión":
    st.header("Relación de Variables 📊")
    eje_x = st.sidebar.selectbox("Eje X", options=columnas_numericas)
    eje_y = st.sidebar.selectbox("Eje Y", options=columnas_numericas)

    x_vals = data_numerica[eje_x]
    y_vals = data_numerica[eje_y]

    correlacion = np.corrcoef(x_vals, y_vals)[0, 1]
    pendiente, intercepto = np.polyfit(x_vals, y_vals, 1)
    linea_regresion = pendiente * x_vals + intercepto

    figura_dispersion = go.Figure()
    figura_dispersion.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Datos'))
    figura_dispersion.add_trace(go.Scatter(x=x_vals, y=linea_regresion, mode='lines', name='Regresión', line=dict(color='red')))
    figura_dispersion.update_layout(title='Dispersión con Línea de Tendencia',
                                    xaxis_title=eje_x, yaxis_title=eje_y)
    st.plotly_chart(figura_dispersion)
    st.subheader("Coeficiente de Correlación")
    st.write(f"Correlación entre *{eje_x}* y *{eje_y}*: **{correlacion:.4f}**")

# Vista: Circular
elif vista == "Gráfico circular":
    st.header("Distribución Categórica")
    cat_col = st.sidebar.selectbox("Categoría", options=columnas_texto)
    num_col = st.sidebar.selectbox("Valor Numérico", options=columnas_numericas)
    graf_pie = px.pie(data_frame=data, names=cat_col, values=num_col, title=f"Distribución de {cat_col}")
    st.plotly_chart(graf_pie)

# Vista: Barras
elif vista == "Barras":
    st.header("Comparación de Categorías 📊")

    # Selección de variables
    categoria = st.sidebar.selectbox("Categoría", options=columnas_texto)
    valor = st.sidebar.selectbox("Valor Numérico", options=columnas_numericas)

    # Gráfica de barras general
    graf_bar = px.bar(data_frame=data, x=categoria, y=valor,
                      title=f"Comparativa de {valor} por {categoria}")
    st.plotly_chart(graf_bar)

    # Top 5 categorías más frecuentes
    st.subheader("Top 5 categorías más frecuentes")
    top5 = data[categoria].value_counts().head(5)

    # Mostrar tabla de frecuencias
    st.write(top5)

    # Gráfica del top 5
    fig_top5 = px.bar(x=top5.index, y=top5.values,
                      labels={'x': categoria, 'y': 'Frecuencia'},
                      title=f"Top 5 valores más frecuentes de {categoria}")
    st.plotly_chart(fig_top5)


# Vista: Modelado Predictivo
elif vista == "Modelado Predictivo":
    st.header("🔢 Modelado Predictivo")
    tipo_modelo = st.selectbox("Selecciona el tipo de modelo", ["Regresión Lineal Simple", "Regresión Lineal Múltiple", "Regresión Logística"])

    if tipo_modelo == "Regresión Lineal Simple":
        x = st.selectbox("Variable independiente (X):", options=columnas_numericas)
        y = st.selectbox("Variable dependiente (Y):", options=columnas_numericas)
        if st.button("Ejecutar Modelo"):
            df_modelo = data[[x, y]].dropna()
            X = df_modelo[[x]]
            Y = df_modelo[y]
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
            modelo = LinearRegression().fit(X_train, Y_train)
            predicciones = modelo.predict(X_test)
            st.markdown("---")
            st.subheader("📈 Resultados")
            st.write("Coeficiente R^2:", r2_score(Y_test, predicciones))
            #st.write("Error cuadrático medio:", mean_squared_error(Y_test, predicciones))

    elif tipo_modelo == "Regresión Lineal Múltiple":
        vars_x = st.multiselect("Variables independientes:", options=columnas_numericas)
        y = st.selectbox("Variable dependiente:", options=columnas_numericas)
        if vars_x and st.button("Ejecutar Modelo"):
            df_modelo = data[vars_x + [y]].dropna()
            X = df_modelo[vars_x]
            Y = df_modelo[y]
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
            modelo = LinearRegression().fit(X_train, Y_train)
            predicciones = modelo.predict(X_test)
            st.markdown("---")
            st.subheader("📈 Resultados")
            st.write("Coeficiente R^2:", r2_score(Y_test, predicciones))
           # st.write("Error cuadrático medio:", mean_squared_error(Y_test, predicciones))

    elif tipo_modelo == "Regresión Logística":
    # 1. Identificar variables binarias (con solo 2 valores únicos)
        columnas_binarias = [col for col in columnas_texto if data[col].nunique() == 2]
    

        if not columnas_binarias:
            st.warning("No se encontraron variables binarias adecuadas.")
        else:
            # 2. Elegir variable objetivo binaria
            var_objetivo = st.selectbox("Variable binaria objetivo (Y):", columnas_binarias)

            # 3. Elegir variables predictoras numéricas
            x_vars = st.multiselect("Variables predictoras (X)", columnas_numericas)

            if x_vars and st.button("Ejecutar Modelo"):
                data_filtrada = data.dropna(subset=x_vars + [var_objetivo])
                X = data_filtrada[x_vars]

                # Convertimos la variable objetivo a binaria (0 y 1)
                y_binaria = data_filtrada[var_objetivo].astype(str).apply(
                    lambda x: 1 if x == data_filtrada[var_objetivo].unique()[0] else 0
                )

                # Entrenamiento
                X_train, X_test, y_train, y_test = train_test_split(X, y_binaria, test_size=0.3)
                modelo = LogisticRegression(max_iter=500).fit(X_train, y_train)
                predicciones = modelo.predict(X_test)

                st.markdown("---")
                st.subheader("📊 Precisión del modelo")
                st.write(f"Precisión: **{accuracy_score(y_test, predicciones):.4f}**")

                # 4. Matriz de confusión
                st.subheader("📌 Matriz de Confusión")
                cm = confusion_matrix(y_test, predicciones)
                etiquetas = ["Real f", "Real t"]
                cm_df = pd.DataFrame(cm, index=etiquetas, columns=["Predicho f", "Predicho t"])
                st.dataframe(cm_df)

                # 5. Reporte de métricas personalizado
                st.subheader("📋 Métricas de Rendimiento")

                # Obtener TN, FP, FN, TP de la matriz de confusión
                tn, fp, fn, tp = confusion_matrix(y_test, predicciones).ravel()

                # Calcular métricas
                exactitud = accuracy_score(y_test, predicciones)
                precision = tp / (tp + fp) if (tp + fp) != 0 else 0
                sensibilidad = tp / (tp + fn) if (tp + fn) != 0 else 0  # también llamada recall
                especificidad = tn / (tn + fp) if (tn + fp) != 0 else 0

                # Crear tabla con métricas
                metricas = pd.DataFrame({
                    "Métrica": ["Exactitud", "Precisión", "Sensibilidad", "Especificidad"],
                    "Valor": [f"{exactitud:.4f}", f"{precision:.4f}", f"{sensibilidad:.4f}", f"{especificidad:.4f}"]
                })

                st.dataframe(metricas.style.set_properties(**{
                    'text-align': 'center',
                    'font-weight': 'bold'
                }).set_table_styles([
                    {"selector": "th", "props": [("text-align", "center"), ("background-color", "#f0f2f6")]}
                ]))


                # 5. Visualización de predicciones
                graf = pd.DataFrame({"Real": y_test, "Predicción": predicciones}).reset_index(drop=True)
                st.bar_chart(graf)
