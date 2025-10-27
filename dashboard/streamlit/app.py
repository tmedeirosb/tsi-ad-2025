"""
Aplicação Streamlit para Análise de Dados - Módulos 01 a 04
============================================================

Esta aplicação demonstra conceitos fundamentais de análise de dados,
seguindo os princípios de storytelling with data de Cole Nussbaumer Knaflic:
- Contexto é fundamental
- Escolher visualizações apropriadas
- Eliminar ruído visual
- Focar a atenção do espectador
- Pensar como um designer
- Contar uma história

Autor: Disciplina de Análise de Dados
Data: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from numpy import sqrt

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Análise de Dados - TSI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILO PERSONALIZADO
# Seguindo princípios de design: simplicidade e clareza
# ============================================================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÃO PARA CARREGAR DADOS COM CACHE
# Cache evita recarregamento desnecessário dos dados
# ============================================================
@st.cache_data
def load_data():
    """
    Carrega os dados do repositório GitHub ou arquivo local.
    Tenta primeiro o GitHub; se falhar (ex: erro 429), usa arquivo local.
    O uso de @st.cache_data garante que os dados sejam carregados apenas uma vez,
    melhorando significativamente a performance da aplicação.
    
    Returns:
        pd.DataFrame: DataFrame com os dados carregados
    """
    try:
        # Tentar carregar do GitHub primeiro
        url = "https://raw.githubusercontent.com/tmedeirosb/tsi-ad-2025/refs/heads/main/dados/dados_workflow_ivan.csv"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        # Se falhar, carregar do arquivo local
        try:
            # Caminho relativo ao arquivo local
            local_path = "../../dados/dados_workflow_ivan.csv"
            df = pd.read_csv(local_path)
            st.warning(f"⚠️ Dados carregados do arquivo local devido a erro no GitHub: {str(e)}")
            return df
        except Exception as e2:
            # Se ambos falharem, lançar erro
            raise Exception(f"Não foi possível carregar os dados. GitHub: {str(e)} | Local: {str(e2)}")

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def calculate_cramers_v(contingency_table):
    """
    Calcula o V de Cramer para medir associação entre variáveis qualitativas.
    
    O V de Cramer varia de 0 (sem associação) a 1 (associação perfeita).
    
    Args:
        contingency_table: Tabela de contingência (pd.crosstab)
    
    Returns:
        float: Valor do V de Cramer
    """
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    
    # Evitar divisão por zero
    if min_dim == 0 or n == 0:
        return 0
    
    return sqrt(chi2 / (n * min_dim))

def create_clean_plot():
    """
    Cria uma figura matplotlib com estilo limpo.
    
    Seguindo os princípios de Cole Nussbaumer Knaflic:
    - Remove elementos desnecessários (ruído visual)
    - Usa cores sutis e profissionais
    - Mantém o foco nos dados
    """
    # Configurar estilo seaborn para gráficos mais limpos
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    # Configurar paleta de cores profissional e acessível
    sns.set_palette("husl")

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

# Título principal
st.title("📊 Análise Exploratória de Dados")
st.markdown("### Técnicas de Análise de Dados - Módulos 01 a 04")

# Informação sobre os dados
st.info("""
**Sobre esta aplicação**: Esta ferramenta interativa demonstra conceitos fundamentais 
de análise exploratória de dados, desde a importação até análises multivariadas.
Use o menu lateral para navegar entre os diferentes módulos.
""")

# ============================================================
# SIDEBAR - NAVEGAÇÃO
# ============================================================
st.sidebar.title("📚 Navegação")
st.sidebar.markdown("---")

modulo = st.sidebar.radio(
    "Escolha o módulo:",
    [
        "🏠 Início",
        "📥 Módulo 01 - Importação de Dados",
        "🔍 Módulo 02 - Manipulação de Dados",
        "📈 Módulo 03 - Estatística Descritiva",
        "🔗 Módulo 04 - Análise Multivariada"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Dica**: Explore cada módulo sequencialmente 
para melhor compreensão dos conceitos.
""")

# ============================================================
# CARREGAR DADOS
# ============================================================
try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    data_loaded = False

# ============================================================
# MÓDULO 0 - INÍCIO
# ============================================================
if modulo == "🏠 Início":
    st.header("Bem-vindo à Aplicação de Análise de Dados!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Objetivos de Aprendizagem
        
        Esta aplicação cobre os seguintes tópicos:
        
        1. **Importação de Dados**
           - Carregar dados de diferentes fontes
           - Visualizar estrutura inicial dos dados
        
        2. **Manipulação de Dados**
           - Seleção de colunas
           - Filtragem de dados
           - Operações de transformação
        
        3. **Estatística Descritiva**
           - Medidas de tendência central
           - Medidas de dispersão
           - Visualizações univariadas
        
        4. **Análise Multivariada**
           - Correlações entre variáveis
           - Análise bivariada
           - Visualizações avançadas
        """)
    
    with col2:
        st.markdown("""
        ### 📖 Princípios de Visualização
        
        Esta aplicação segue os princípios de 
        **Storytelling with Data** de Cole Nussbaumer Knaflic:
        
        - ✅ **Contexto**: Cada gráfico tem um propósito claro
        - ✅ **Simplicidade**: Removemos elementos desnecessários
        - ✅ **Destaque**: Focamos a atenção no que importa
        - ✅ **Clareza**: Rótulos e títulos informativos
        - ✅ **Cores**: Paleta profissional e acessível
        
        ### 🚀 Como Usar
        
        1. Use o menu lateral para navegar entre módulos
        2. Explore os controles interativos em cada seção
        3. Observe os comentários explicativos
        4. Experimente diferentes configurações
        """)
    
    if data_loaded:
        st.success(f"✅ Dados carregados com sucesso! ({len(df)} registros, {len(df.columns)} colunas)")
        
        # Estatísticas rápidas
        st.markdown("### 📊 Visão Geral dos Dados")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Registros", f"{len(df):,}")
        with col2:
            st.metric("Total de Colunas", len(df.columns))
        with col3:
            st.metric("Variáveis Numéricas", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Variáveis Categóricas", len(df.select_dtypes(include=['object']).columns))

# ============================================================
# MÓDULO 01 - IMPORTAÇÃO DE DADOS
# ============================================================
elif modulo == "📥 Módulo 01 - Importação de Dados":
    st.header("📥 Módulo 01: Importação e Visualização Inicial")
    
    st.markdown("""
    ### Objetivo
    Aprender a importar dados de diferentes fontes e realizar uma primeira inspeção.
    """)
    
    if not data_loaded:
        st.error("Dados não carregados. Verifique a conexão.")
    else:
        # Tabs para organizar o conteúdo
        tab1, tab2, tab3 = st.tabs(["📋 Primeiras Linhas", "ℹ️ Informações", "📊 Tipos de Dados"])
        
        with tab1:
            st.subheader("Primeiras Linhas do Dataset")
            st.markdown("""
            **Por que visualizar as primeiras linhas?**
            - Verificar se os dados foram carregados corretamente
            - Entender a estrutura básica dos dados
            - Identificar os tipos de variáveis presentes
            """)
            
            # Controle para número de linhas
            n_rows = st.slider("Número de linhas para visualizar:", 5, 50, 10)
            st.dataframe(df.head(n_rows), use_container_width=True)
        
        with tab2:
            st.subheader("Informações sobre o Dataset")
            st.markdown("""
            **Informações importantes:**
            - Número total de entradas (registros)
            - Tipos de dados de cada coluna
            - Presença de valores ausentes (null)
            - Uso de memória
            """)
            
            # Criar DataFrame com informações
            info_data = {
                'Coluna': df.columns,
                'Tipo': df.dtypes.values,
                'Não-Nulos': df.count().values,
                'Nulos': df.isnull().sum().values,
                '% Nulos': (df.isnull().sum().values / len(df) * 100).round(2)
            }
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True)
            
            # Estatísticas resumidas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Registros", f"{len(df):,}")
            with col2:
                st.metric("Total de Colunas", len(df.columns))
            with col3:
                total_nulls = df.isnull().sum().sum()
                st.metric("Total de Valores Nulos", f"{total_nulls:,}")
        
        with tab3:
            st.subheader("Distribuição dos Tipos de Dados")
            st.markdown("""
            **Entendendo os tipos de dados:**
            - **int64**: Números inteiros (ex: idade, quantidade)
            - **float64**: Números decimais (ex: notas, salários)
            - **object**: Texto ou categórico (ex: nome, status)
            """)
            
            # Contar tipos de dados
            type_counts = df.dtypes.value_counts()
            
            # Criar gráfico de barras simples e limpo
            fig, ax = plt.subplots(figsize=(10, 6))
            type_counts.plot(kind='barh', ax=ax, color='#1f77b4')
            ax.set_xlabel('Número de Colunas', fontsize=12)
            ax.set_ylabel('Tipo de Dado', fontsize=12)
            ax.set_title('Distribuição dos Tipos de Dados', fontsize=14, pad=20)
            
            # Adicionar valores nas barras
            for i, v in enumerate(type_counts.values):
                ax.text(v + 0.5, i, str(v), va='center', fontsize=11)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ============================================================
# MÓDULO 02 - MANIPULAÇÃO DE DADOS
# ============================================================
elif modulo == "🔍 Módulo 02 - Manipulação de Dados":
    st.header("🔍 Módulo 02: Manipulação de Dados com Pandas")
    
    st.markdown("""
    ### Objetivo
    Aprender técnicas essenciais de manipulação de dados usando Pandas.
    """)
    
    if not data_loaded:
        st.error("Dados não carregados. Verifique a conexão.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Seleção de Colunas", 
            "🔎 Filtros", 
            "📊 Agrupamento",
            "🔄 Ordenação"
        ])
        
        with tab1:
            st.subheader("Seleção de Colunas")
            st.markdown("""
            **Por que selecionar colunas?**
            - Reduzir a complexidade dos dados
            - Focar nas variáveis relevantes para a análise
            - Melhorar a performance do processamento
            """)
            
            # Multiselect para escolher colunas
            selected_cols = st.multiselect(
                "Selecione as colunas que deseja visualizar:",
                df.columns.tolist(),
                default=df.columns.tolist()[:5]
            )
            
            if selected_cols:
                st.dataframe(df[selected_cols].head(20), use_container_width=True)
                
                # Mostrar o código equivalente
                with st.expander("💻 Ver código Python equivalente"):
                    st.code(f"""
# Selecionando colunas específicas
colunas_selecionadas = {selected_cols}
df_filtrado = df[colunas_selecionadas]
display(df_filtrado.head())
                    """, language="python")
        
        with tab2:
            st.subheader("Filtros (Seleção Booleana)")
            st.markdown("""
            **Filtrar dados permite:**
            - Focar em subconjuntos específicos
            - Comparar grupos diferentes
            - Identificar padrões em categorias específicas
            """)
            
            # Escolher coluna para filtrar
            filter_col = st.selectbox(
                "Escolha uma coluna para filtrar:",
                ['idade', 'descricao', 'qnt_salarios'] + 
                [col for col in df.select_dtypes(include=['object', 'int64', 'float64']).columns 
                 if col not in ['idade', 'descricao', 'qnt_salarios']][:5]
            )
            
            # Filtros dinâmicos baseados no tipo de coluna
            if df[filter_col].dtype in ['int64', 'float64']:
                # Filtro numérico
                col1, col2 = st.columns(2)
                with col1:
                    min_val = st.number_input(
                        f"Valor mínimo de {filter_col}:", 
                        value=float(df[filter_col].min())
                    )
                with col2:
                    max_val = st.number_input(
                        f"Valor máximo de {filter_col}:", 
                        value=float(df[filter_col].max())
                    )
                
                df_filtered = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]
            else:
                # Filtro categórico
                unique_values = df[filter_col].dropna().unique()
                selected_values = st.multiselect(
                    f"Selecione valores de {filter_col}:",
                    unique_values,
                    default=unique_values[:3] if len(unique_values) >= 3 else unique_values
                )
                
                if selected_values:
                    df_filtered = df[df[filter_col].isin(selected_values)]
                else:
                    df_filtered = df
            
            # Mostrar resultados
            st.info(f"📊 Registros após filtro: {len(df_filtered)} de {len(df)} ({len(df_filtered)/len(df)*100:.1f}%)")
            st.dataframe(df_filtered.head(20), use_container_width=True)
        
        with tab3:
            st.subheader("Agrupamento e Agregação")
            st.markdown("""
            **Agrupamento permite:**
            - Calcular estatísticas por categoria
            - Comparar grupos diferentes
            - Resumir grandes volumes de dados
            """)
            
            # Selecionar coluna para agrupar
            group_col = st.selectbox(
                "Agrupar por:",
                df.select_dtypes(include=['object']).columns
            )
            
            # Selecionar coluna numérica para agregação
            agg_col = st.selectbox(
                "Coluna para calcular estatísticas:",
                df.select_dtypes(include=[np.number]).columns
            )
            
            # Selecionar função de agregação
            agg_func = st.selectbox(
                "Função de agregação:",
                ['mean', 'sum', 'count', 'min', 'max', 'median']
            )
            
            # Calcular agregação
            if group_col and agg_col:
                result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
                result.columns = [group_col, f'{agg_func}_{agg_col}']
                result = result.sort_values(f'{agg_func}_{agg_col}', ascending=False)
                
                # Visualizar resultado
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(result, use_container_width=True)
                
                with col2:
                    # Gráfico de barras limpo
                    fig, ax = plt.subplots(figsize=(10, 6))
                    result_plot = result.head(10)  # Top 10
                    ax.barh(result_plot[group_col], result_plot[f'{agg_func}_{agg_col}'], 
                           color='#1f77b4')
                    ax.set_xlabel(f'{agg_func.title()} de {agg_col}', fontsize=12)
                    ax.set_ylabel(group_col, fontsize=12)
                    ax.set_title(f'{agg_func.title()} de {agg_col} por {group_col} (Top 10)', 
                                fontsize=14, pad=20)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        
        with tab4:
            st.subheader("Ordenação de Dados")
            st.markdown("""
            **Ordenar dados ajuda a:**
            - Identificar valores extremos (maiores/menores)
            - Visualizar padrões de forma mais clara
            - Preparar dados para análises específicas
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                sort_col = st.selectbox(
                    "Ordenar por:",
                    df.columns
                )
            
            with col2:
                sort_order = st.radio(
                    "Ordem:",
                    ['Crescente', 'Decrescente']
                )
            
            ascending = True if sort_order == 'Crescente' else False
            df_sorted = df.sort_values(by=sort_col, ascending=ascending)
            
            st.dataframe(df_sorted.head(20), use_container_width=True)

# ============================================================
# MÓDULO 03 - ESTATÍSTICA DESCRITIVA
# ============================================================
elif modulo == "📈 Módulo 03 - Estatística Descritiva":
    st.header("📈 Módulo 03: Estatística Descritiva e Visualização")
    
    st.markdown("""
    ### Objetivo
    Compreender distribuições, medidas de tendência central e dispersão.
    """)
    
    if not data_loaded:
        st.error("Dados não carregados. Verifique a conexão.")
    else:
        tab1, tab2, tab3 = st.tabs([
            "📊 Variáveis Qualitativas",
            "📈 Variáveis Quantitativas",
            "📦 Medidas de Dispersão"
        ])
        
        with tab1:
            st.subheader("Análise de Variáveis Qualitativas")
            st.markdown("""
            **Variáveis qualitativas** representam categorias ou atributos.
            A análise se concentra em frequências e proporções.
            """)
            
            # Selecionar variável qualitativa
            qual_var = st.selectbox(
                "Selecione uma variável qualitativa:",
                df.select_dtypes(include=['object']).columns
            )
            
            if qual_var:
                # Calcular frequências
                freq_table = df[qual_var].value_counts().reset_index()
                freq_table.columns = [qual_var, 'Contagem']
                freq_table['Percentual (%)'] = (freq_table['Contagem'] / len(df) * 100).round(2)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Tabela de Frequências:**")
                    st.dataframe(freq_table, use_container_width=True)
                
                with col2:
                    # Gráfico de barras horizontal limpo
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Limitar a 10 categorias para clareza
                    plot_data = freq_table.head(10)
                    
                    ax.barh(plot_data[qual_var], plot_data['Contagem'], color='#1f77b4')
                    ax.set_xlabel('Contagem', fontsize=12, fontweight='bold')
                    ax.set_ylabel(qual_var, fontsize=12, fontweight='bold')
                    ax.set_title(f'Distribuição de {qual_var} (Top 10)', 
                                fontsize=14, pad=20, fontweight='bold')
                    
                    # Adicionar valores nas barras
                    for i, v in enumerate(plot_data['Contagem']):
                        ax.text(v, i, f' {v}', va='center', fontsize=10)
                    
                    # Remover spines superiores e direita (princípio de Cole: menos ruído)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        
        with tab2:
            st.subheader("Análise de Variáveis Quantitativas")
            st.markdown("""
            **Variáveis quantitativas** representam valores numéricos.
            Analisamos distribuição, tendência central e dispersão.
            """)
            
            # Selecionar variável quantitativa
            quant_var = st.selectbox(
                "Selecione uma variável quantitativa:",
                df.select_dtypes(include=[np.number]).columns
            )
            
            if quant_var:
                # Estatísticas descritivas
                stats = df[quant_var].describe()
                
                # Medidas de tendência central
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Média", f"{stats['mean']:.2f}")
                with col2:
                    st.metric("Mediana", f"{stats['50%']:.2f}")
                with col3:
                    st.metric("Mínimo", f"{stats['min']:.2f}")
                with col4:
                    st.metric("Máximo", f"{stats['max']:.2f}")
                
                # Visualizações
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df[quant_var].dropna(), bins=30, color='#1f77b4', 
                           edgecolor='white', alpha=0.7)
                    ax.axvline(stats['mean'], color='red', linestyle='--', 
                              linewidth=2, label=f'Média: {stats["mean"]:.2f}')
                    ax.axvline(stats['50%'], color='green', linestyle='--', 
                              linewidth=2, label=f'Mediana: {stats["50%"]:.2f}')
                    ax.set_xlabel(quant_var, fontsize=12, fontweight='bold')
                    ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
                    ax.set_title(f'Distribuição de {quant_var}', 
                                fontsize=14, pad=20, fontweight='bold')
                    ax.legend()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Box Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bp = ax.boxplot(df[quant_var].dropna(), vert=False, patch_artist=True)
                    
                    # Colorir a caixa
                    for patch in bp['boxes']:
                        patch.set_facecolor('#1f77b4')
                        patch.set_alpha(0.7)
                    
                    ax.set_xlabel(quant_var, fontsize=12, fontweight='bold')
                    ax.set_title(f'Box Plot de {quant_var}', 
                                fontsize=14, pad=20, fontweight='bold')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set_yticks([])
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Interpretação
                st.markdown("""
                **Interpretação:**
                - **Histograma**: Mostra a distribuição dos valores e a forma da distribuição
                - **Box Plot**: Identifica valores extremos (outliers) e quartis
                - **Linha vermelha**: Média (sensível a valores extremos)
                - **Linha verde**: Mediana (mais robusta a outliers)
                """)
        
        with tab3:
            st.subheader("Medidas de Dispersão")
            st.markdown("""
            **Dispersão** indica o quão espalhados os dados estão.
            Importante para entender a variabilidade dos dados.
            """)
            
            quant_var_disp = st.selectbox(
                "Selecione uma variável quantitativa:",
                df.select_dtypes(include=[np.number]).columns,
                key="disp_var"
            )
            
            if quant_var_disp:
                stats = df[quant_var_disp].describe()
                
                # Calcular medidas adicionais
                variance = df[quant_var_disp].var()
                std = stats['std']
                iqr = stats['75%'] - stats['25%']
                amplitude = stats['max'] - stats['min']
                
                # Exibir medidas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Desvio Padrão", f"{std:.2f}")
                    st.caption("Medida de variabilidade mais comum")
                
                with col2:
                    st.metric("Variância", f"{variance:.2f}")
                    st.caption("Quadrado do desvio padrão")
                
                with col3:
                    st.metric("IQR", f"{iqr:.2f}")
                    st.caption("Intervalo Interquartil (Q3-Q1)")
                
                with col4:
                    st.metric("Amplitude", f"{amplitude:.2f}")
                    st.caption("Diferença entre máx e mín")
                
                # Violin plot para visualizar dispersão
                fig, ax = plt.subplots(figsize=(12, 6))
                parts = ax.violinplot([df[quant_var_disp].dropna()], 
                                     vert=False, showmeans=True, showmedians=True)
                
                # Colorir
                for pc in parts['bodies']:
                    pc.set_facecolor('#1f77b4')
                    pc.set_alpha(0.7)
                
                ax.set_xlabel(quant_var_disp, fontsize=12, fontweight='bold')
                ax.set_title(f'Violin Plot de {quant_var_disp} - Visualizando Dispersão', 
                            fontsize=14, pad=20, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ============================================================
# MÓDULO 04 - ANÁLISE MULTIVARIADA
# ============================================================
elif modulo == "🔗 Módulo 04 - Análise Multivariada":
    st.header("🔗 Módulo 04: Análise Multivariada")
    
    st.markdown("""
    ### Objetivo
    Explorar relações entre múltiplas variáveis simultaneamente.
    """)
    
    if not data_loaded:
        st.error("Dados não carregados. Verifique a conexão.")
    else:
        tab1, tab2, tab3 = st.tabs([
            "📊 Correlação entre Variáveis Numéricas",
            "🔗 Associação entre Variáveis Categóricas",
            "🎯 Análise Multivariada"
        ])
        
        with tab1:
            st.subheader("Correlação entre Variáveis Quantitativas")
            st.markdown("""
            **Correlação** mede a força e direção da relação linear entre duas variáveis.
            - Varia de -1 (correlação negativa perfeita) a +1 (correlação positiva perfeita)
            - Valores próximos a 0 indicam correlação fraca
            """)
            
            # Selecionar variáveis numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    var1 = st.selectbox("Variável 1:", numeric_cols, key="corr_var1")
                
                with col2:
                    var2 = st.selectbox("Variável 2:", 
                                       [c for c in numeric_cols if c != var1], 
                                       key="corr_var2")
                
                if var1 and var2:
                    # Calcular correlação
                    correlation = df[[var1, var2]].corr().iloc[0, 1]
                    
                    # Interpretar correlação
                    if abs(correlation) < 0.3:
                        strength = "fraca"
                        color = "blue"
                    elif abs(correlation) < 0.7:
                        strength = "moderada"
                        color = "orange"
                    else:
                        strength = "forte"
                        color = "red"
                    
                    direction = "positiva" if correlation > 0 else "negativa"
                    
                    st.info(f"""
                    **Correlação de Pearson**: {correlation:.3f}
                    
                    Interpretação: Correlação **{strength} {direction}**
                    """)
                    
                    # Scatter plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Amostrar dados se houver muitos pontos
                    if len(df) > 1000:
                        df_sample = df.sample(n=1000, random_state=42)
                    else:
                        df_sample = df
                    
                    # Remover valores NaN para o scatter plot e linha de tendência
                    df_clean = df_sample[[var1, var2]].dropna()
                    
                    if len(df_clean) > 0:
                        ax.scatter(df_clean[var1], df_clean[var2], 
                                  alpha=0.5, color='#1f77b4', edgecolors='white', linewidth=0.5)
                        
                        # Adicionar linha de tendência apenas se houver dados suficientes
                        if len(df_clean) > 1:
                            z = np.polyfit(df_clean[var1], df_clean[var2], 1)
                            p = np.poly1d(z)
                            x_sorted = df_clean[var1].sort_values()
                            ax.plot(x_sorted, p(x_sorted), 
                                   "r--", alpha=0.8, linewidth=2, 
                                   label=f'Linha de Tendência (r={correlation:.3f})')
                    
                        ax.set_xlabel(var1, fontsize=12, fontweight='bold')
                        ax.set_ylabel(var2, fontsize=12, fontweight='bold')
                        ax.set_title(f'Relação entre {var1} e {var2}', 
                                    fontsize=14, pad=20, fontweight='bold')
                        ax.legend()
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Não há dados válidos para plotar.")
                    
                    plt.close()
                
                # Matriz de correlação
                st.markdown("---")
                st.subheader("Matriz de Correlação")
                
                # Selecionar variáveis para a matriz
                selected_vars = st.multiselect(
                    "Selecione variáveis para incluir na matriz de correlação:",
                    numeric_cols,
                    default=numeric_cols[:min(8, len(numeric_cols))]
                )
                
                if selected_vars and len(selected_vars) >= 2:
                    corr_matrix = df[selected_vars].corr()
                    
                    # Heatmap
                    fig, ax = plt.subplots(figsize=(12, 10))
                    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', 
                                  vmin=-1, vmax=1)
                    
                    # Configurar ticks
                    ax.set_xticks(np.arange(len(selected_vars)))
                    ax.set_yticks(np.arange(len(selected_vars)))
                    ax.set_xticklabels(selected_vars, rotation=45, ha='right')
                    ax.set_yticklabels(selected_vars)
                    
                    # Adicionar valores nas células
                    for i in range(len(selected_vars)):
                        for j in range(len(selected_vars)):
                            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                         ha="center", va="center", color="black", 
                                         fontsize=9)
                    
                    ax.set_title('Matriz de Correlação de Pearson', 
                                fontsize=14, pad=20, fontweight='bold')
                    
                    # Colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Correlação', rotation=270, labelpad=20)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            else:
                st.warning("Necessário pelo menos 2 variáveis numéricas para análise de correlação.")
        
        with tab2:
            st.subheader("Associação entre Variáveis Qualitativas")
            st.markdown("""
            **V de Cramer** mede a associação entre variáveis categóricas.
            - Varia de 0 (sem associação) a 1 (associação perfeita)
            - Baseado no teste Qui-Quadrado
            """)
            
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if len(categorical_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    cat_var1 = st.selectbox("Variável Categórica 1:", 
                                           categorical_cols, key="cat_var1")
                
                with col2:
                    cat_var2 = st.selectbox("Variável Categórica 2:", 
                                           [c for c in categorical_cols if c != cat_var1], 
                                           key="cat_var2")
                
                if cat_var1 and cat_var2:
                    # Criar tabela de contingência
                    contingency_table = pd.crosstab(df[cat_var1], df[cat_var2])
                    
                    # Calcular V de Cramer
                    v_cramer = calculate_cramers_v(contingency_table)
                    
                    # Interpretar
                    if v_cramer < 0.1:
                        strength = "muito fraca"
                    elif v_cramer < 0.3:
                        strength = "fraca"
                    elif v_cramer < 0.5:
                        strength = "moderada"
                    else:
                        strength = "forte"
                    
                    st.info(f"""
                    **V de Cramer**: {v_cramer:.3f}
                    
                    Interpretação: Associação **{strength}** entre as variáveis
                    """)
                    
                    # Tabela de contingência
                    st.markdown("**Tabela de Contingência:**")
                    st.dataframe(contingency_table, use_container_width=True)
                    
                    # Heatmap da tabela de contingência
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Limitar categorias se houver muitas
                    if contingency_table.shape[0] > 10 or contingency_table.shape[1] > 10:
                        st.warning("⚠️ Muitas categorias detectadas. Mostrando apenas as 10 principais de cada variável.")
                        top_rows = contingency_table.sum(axis=1).nlargest(10).index
                        top_cols = contingency_table.sum(axis=0).nlargest(10).index
                        contingency_table = contingency_table.loc[top_rows, top_cols]
                    
                    im = ax.imshow(contingency_table, cmap='Blues', aspect='auto')
                    
                    # Configurar ticks
                    ax.set_xticks(np.arange(len(contingency_table.columns)))
                    ax.set_yticks(np.arange(len(contingency_table.index)))
                    ax.set_xticklabels(contingency_table.columns, rotation=45, ha='right')
                    ax.set_yticklabels(contingency_table.index)
                    
                    # Adicionar valores
                    for i in range(len(contingency_table.index)):
                        for j in range(len(contingency_table.columns)):
                            text = ax.text(j, i, contingency_table.iloc[i, j],
                                         ha="center", va="center", color="black", fontsize=9)
                    
                    ax.set_xlabel(cat_var2, fontsize=12, fontweight='bold')
                    ax.set_ylabel(cat_var1, fontsize=12, fontweight='bold')
                    ax.set_title(f'Tabela de Contingência: {cat_var1} vs {cat_var2}', 
                                fontsize=14, pad=20, fontweight='bold')
                    
                    plt.colorbar(im, ax=ax, label='Frequência')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            else:
                st.warning("Necessário pelo menos 2 variáveis categóricas para análise de associação.")
        
        with tab3:
            st.subheader("Análise Multivariada")
            st.markdown("""
            **Análise N-variada** explora relações entre três ou mais variáveis.
            Permite descobrir padrões complexos e interações entre variáveis.
            """)
            
            # Análise com 3 variáveis: 2 numéricas + 1 categórica
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    num_var1 = st.selectbox("Variável Numérica (Eixo X):", 
                                           numeric_cols, key="multi_num1")
                
                with col2:
                    num_var2 = st.selectbox("Variável Numérica (Eixo Y):", 
                                           [c for c in numeric_cols if c != num_var1], 
                                           key="multi_num2")
                
                with col3:
                    cat_var = st.selectbox("Variável Categórica (Cor):", 
                                          categorical_cols, key="multi_cat")
                
                if num_var1 and num_var2 and cat_var:
                    # Limitar categorias
                    top_categories = df[cat_var].value_counts().head(5).index
                    df_filtered = df[df[cat_var].isin(top_categories)]
                    
                    # Scatter plot por categoria
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Cores distintas para cada categoria
                    colors = plt.cm.Set3(np.linspace(0, 1, len(top_categories)))
                    
                    for idx, category in enumerate(top_categories):
                        mask = df_filtered[cat_var] == category
                        ax.scatter(df_filtered[mask][num_var1], 
                                  df_filtered[mask][num_var2],
                                  alpha=0.6, label=category, 
                                  color=colors[idx], edgecolors='white', 
                                  linewidth=0.5, s=50)
                    
                    ax.set_xlabel(num_var1, fontsize=12, fontweight='bold')
                    ax.set_ylabel(num_var2, fontsize=12, fontweight='bold')
                    ax.set_title(f'{num_var1} vs {num_var2} por {cat_var}', 
                                fontsize=14, pad=20, fontweight='bold')
                    ax.legend(title=cat_var, bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Estatísticas por grupo
                    st.markdown("---")
                    st.subheader("Estatísticas por Grupo")
                    
                    grouped_stats = df_filtered.groupby(cat_var)[[num_var1, num_var2]].agg([
                        'mean', 'median', 'std', 'min', 'max'
                    ]).round(2)
                    
                    st.dataframe(grouped_stats, use_container_width=True)
            else:
                st.warning("Necessário pelo menos 2 variáveis numéricas e 1 categórica para análise multivariada.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Aplicação Educacional de Análise de Dados</strong></p>
    <p>Desenvolvida para a disciplina de Técnicas de Análise de Dados - TSI</p>
    <p>Baseada nos princípios de <em>Storytelling with Data</em> de Cole Nussbaumer Knaflic</p>
</div>
""", unsafe_allow_html=True)
