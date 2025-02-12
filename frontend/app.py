import streamlit as st
import requests

# Define the API URL (replace with your actual Cloud Run URL)
API_URL = "https://toxic-comments-api-678895434688.us-central1.run.app/predict"


import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Função para enviar texto para a API Flask e obter previsões
def prever_toxicidade(texto):
    """Envia um texto para a API e retorna as previsões de toxicidade."""
    # api_url = "http://localhost:5000/predict"  # Altere para URL de produção se necessário
    resposta = requests.post(api_url, json={"text": texto})

    if resposta.status_code == 200:
        return resposta.json()
    else:
        return {"erro": "Falha ao obter previsão"}

# Configuração da Interface no Streamlit
st.title("🔍 Classificação de Comentários Tóxicos")
st.markdown("Digite um comentário para analisar seu nível de toxicidade.")

# Entrada do usuário
entrada_usuario = st.text_area("📝 Comentário:", "")

# Botão para análise
if st.button("📊 Analisar"):
    if entrada_usuario:
        resultado = prever_toxicidade(entrada_usuario)

        if "erro" in resultado:
            st.error(resultado["erro"])
        else:
            # Exibir textos original e traduzido
            st.subheader("📄 Resultados da Análise")
            st.markdown(f"**🗣️ Texto Original:** {resultado['original_text']}")
            st.markdown(f"**🌎 Tradução:** {resultado['translated_text']}")

            # Obter previsões e arredondar valores
            previsoes = resultado["prediction"]
            previsoes_arredondadas = {k: round(v, 2) for k, v in previsoes.items()}

            # Exibir tabela de previsões
            st.markdown("### 🔢 Níveis de Toxicidade")
            df = pd.DataFrame.from_dict(previsoes_arredondadas, orient="index", columns=["Probabilidade"])
            df.reset_index(inplace=True)
            df.columns = ["Categoria", "Probabilidade"]
            st.dataframe(df)

            # Criar gráfico de barras
            st.markdown("### 📊 Visualização Gráfica")
            fig, ax = plt.subplots()
            ax.barh(df["Categoria"], df["Probabilidade"], color="crimson")
            ax.set_xlabel("Probabilidade de Toxicidade")
            ax.set_title("Classificação de Toxicidade do Comentário")

            # Exibir gráfico no Streamlit
            st.pyplot(fig)
    else:
        st.warning("⚠️ Por favor, insira um comentário antes de analisar.")


if __name__ == "__main__":
    # Ensure Streamlit is bound to the correct port
    import os
    os.system("streamlit run app.py --server.port=8501 --server.address=0.0.0.0")
