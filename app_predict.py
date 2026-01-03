import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from io import BytesIO

# Configuração da página
st.set_page_config(
    page_title="Predição de Evasão Escolar",
    page_icon="🎓",
    layout="wide"
)

# Título da aplicação
st.title("🎓 Predição de Evasão Escolar")
st.markdown("---")

# Categorias disponíveis (baseadas no dataset original)
CATEGORIAS_ESCOLARIDADE = [
    'Ensino médio completo',
    'Ensino fundamental incompleto',
    'Ensino médio incompleto',
    'Ensino fundamental completo',
    'Ensino superior completo',
    'Ensino superior incompleto',
    'Alfabetizado',
    'Não estudou',
    'Pós graduação completo',
    'Pós graduação incompleto',
    'Não conhece'
]

CATEGORIAS_RACA = [
    'Parda', 'Branca', 'Preta', 'Amarela', 'Não declarado', 'Indígena'
]

CATEGORIAS_SEXO = ['F', 'M']

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo do GitHub"""
    try:
        # URL do modelo no GitHub (raw)
        url = "https://raw.githubusercontent.com/tmedeirosb/tsi-ad-2025/main/modelo_arvore_decisao.pkl"
        response = requests.get(url)
        response.raise_for_status()
        modelo = pickle.load(BytesIO(response.content))
        return modelo
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Carregar o modelo
modelo = carregar_modelo()

if modelo is not None:
    st.success("✅ Modelo carregado com sucesso!")
    
    # Sidebar para escolher o modo de entrada
    st.sidebar.header("📊 Modo de Entrada")
    modo = st.sidebar.radio(
        "Escolha o modo de entrada de dados:",
        ["Entrada Manual", "Gerar Dados Fake"]
    )
    
    st.markdown("---")
    
    if modo == "Entrada Manual":
        st.header("📝 Entrada Manual de Dados")
        st.markdown("Preencha os campos abaixo com os dados do aluno:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lingua_portuguesa = st.number_input(
                "Língua Portuguesa e Literatura I (90H)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=0.1,
                help="Nota de 0 a 100"
            )
            
            matematica = st.number_input(
                "Matemática I (120H)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=0.1,
                help="Nota de 0 a 100"
            )
            
            idade = st.number_input(
                "Idade",
                min_value=14,
                max_value=80,
                value=18,
                step=1
            )
        
        with col2:
            escolaridade = st.selectbox(
                "Escolaridade do Responsável",
                options=CATEGORIAS_ESCOLARIDADE
            )
            
            raca = st.selectbox(
                "Raça/Cor",
                options=CATEGORIAS_RACA
            )
            
            sexo = st.selectbox(
                "Sexo",
                options=CATEGORIAS_SEXO,
                format_func=lambda x: "Feminino" if x == "F" else "Masculino"
            )
        
        # Botão para fazer predição
        if st.button("🔮 Fazer Predição", type="primary"):
            # Criar DataFrame com os dados
            dados = pd.DataFrame({
                'LnguaPortuguesaeLiteraturaI90H': [lingua_portuguesa],
                'MatemticaI120H': [matematica],
                'descricao_responsavel_escolaridade': [escolaridade],
                'idade': [idade],
                'descricao_raca': [raca],
                'pessoa_fisica__sexo': [sexo]
            })
            
            # Fazer predição
            predicao = modelo.predict(dados)
            probabilidades = modelo.predict_proba(dados)
            
            st.markdown("---")
            st.header("📊 Resultado da Predição")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if predicao[0] == 1:
                    st.error("### ⚠️ RISCO DE EVASÃO")
                else:
                    st.success("### ✅ PERMANÊNCIA")
            
            with col2:
                st.metric(
                    "Probabilidade de Permanência",
                    f"{probabilidades[0][0]*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Probabilidade de Evasão",
                    f"{probabilidades[0][1]*100:.1f}%"
                )
            
            # Mostrar dados de entrada
            st.subheader("Dados de Entrada:")
            st.dataframe(dados, use_container_width=True)
    
    else:  # Gerar Dados Fake
        st.header("🎲 Geração de Dados Fake")
        st.markdown("Gere automaticamente dados aleatórios para teste do modelo:")
        
        n_amostras = st.slider(
            "Quantidade de amostras a gerar:",
            min_value=1,
            max_value=50,
            value=5
        )
        
        if st.button("🎲 Gerar Dados e Fazer Predições", type="primary"):
            # Gerar dados fake
            np.random.seed(None)  # Para diferentes resultados a cada execução
            
            dados_fake = pd.DataFrame({
                'LnguaPortuguesaeLiteraturaI90H': np.random.uniform(0, 100, n_amostras),
                'MatemticaI120H': np.random.uniform(0, 100, n_amostras),
                'descricao_responsavel_escolaridade': np.random.choice(CATEGORIAS_ESCOLARIDADE, n_amostras),
                'idade': np.random.randint(14, 65, n_amostras),
                'descricao_raca': np.random.choice(CATEGORIAS_RACA, n_amostras),
                'pessoa_fisica__sexo': np.random.choice(CATEGORIAS_SEXO, n_amostras)
            })
            
            # Fazer predições
            predicoes = modelo.predict(dados_fake)
            probabilidades = modelo.predict_proba(dados_fake)
            
            # Adicionar resultados ao DataFrame
            dados_fake['Predição'] = ['EVASÃO' if p == 1 else 'PERMANÊNCIA' for p in predicoes]
            dados_fake['Prob. Permanência (%)'] = (probabilidades[:, 0] * 100).round(1)
            dados_fake['Prob. Evasão (%)'] = (probabilidades[:, 1] * 100).round(1)
            
            st.markdown("---")
            st.header("📊 Resultados das Predições")
            
            # Estatísticas gerais
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_evasao = sum(predicoes == 1)
                st.metric("Total Evasão", total_evasao)
            
            with col2:
                total_permanencia = sum(predicoes == 0)
                st.metric("Total Permanência", total_permanencia)
            
            with col3:
                taxa_evasao = (total_evasao / n_amostras) * 100
                st.metric("Taxa de Evasão", f"{taxa_evasao:.1f}%")
            
            # Tabela com resultados
            st.subheader("Dados Gerados e Predições:")
            
            # Estilizar a tabela
            def highlight_predicao(val):
                if val == 'EVASÃO':
                    return 'background-color: #ffcccc'
                elif val == 'PERMANÊNCIA':
                    return 'background-color: #ccffcc'
                return ''
            
            styled_df = dados_fake.style.applymap(
                highlight_predicao, 
                subset=['Predição']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Gráfico de barras
            st.subheader("Distribuição das Predições:")
            
            import plotly.express as px
            
            fig = px.bar(
                x=['Permanência', 'Evasão'],
                y=[total_permanencia, total_evasao],
                color=['Permanência', 'Evasão'],
                color_discrete_map={'Permanência': '#2ecc71', 'Evasão': '#e74c3c'},
                labels={'x': 'Classe', 'y': 'Quantidade'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download dos dados
            csv = dados_fake.to_csv(index=False)
            st.download_button(
                label="📥 Download dos Resultados (CSV)",
                data=csv,
                file_name="predicoes_evasao.csv",
                mime="text/csv"
            )

else:
    st.error("❌ Não foi possível carregar o modelo. Verifique se o arquivo está disponível no GitHub.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Desenvolvido para o curso de Análise de Dados - TSI 2025</p>
    <p>Modelo: Árvore de Decisão com Pipeline sklearn</p>
</div>
""", unsafe_allow_html=True)
