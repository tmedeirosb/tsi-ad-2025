# 📊 Simulação de Dados Interativa

Aplicação Streamlit para desenhar dados interativamente e visualizar estatísticas descritivas por classe.

## 🎯 Funcionalidades

- **Desenho Interativo**: Use DrawData para criar datasets visualmente
- **Múltiplas Classes**: Suporte para várias classes de dados
- **Estatísticas Descritivas**: Média, mediana, desvio padrão e variância
- **Visualizações**: Histogramas com KDE para cada classe
- **Comparação**: Visualize todas as classes juntas
- **Export**: Baixe os dados em formato CSV

## 🚀 Instalação

### Usando UV (recomendado)

```bash
# Instalar uv se ainda não tiver
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar dependências
uv pip install streamlit drawdata pandas numpy matplotlib seaborn
```

### Usando pip

```bash
pip install streamlit drawdata pandas numpy matplotlib seaborn
```

## ▶️ Como Executar

```bash
# Navegue até a pasta
cd simulacao_modulo_03

# Execute a aplicação
streamlit run app.py
```

## 📖 Como Usar

1. **Desenhe os Dados**:
   - Use o ScatterWidget do DrawData na interface
   - Clique no gráfico para adicionar pontos
   - Use os botões coloridos para mudar de classe
   - Clique em "Copy CSV" no widget

2. **Cole e Analise**:
   - Cole o CSV no campo de texto
   - Clique em "Analisar Dados"

3. **Visualize as Estatísticas**:
   - Cada classe tem sua própria aba
   - Veja histogramas com KDE
   - Analise média, mediana, desvio padrão e variância

4. **Compare Classes**:
   - Visualize todas as classes em um único gráfico
   - Compare estatísticas na tabela resumo

5. **Exporte os Dados**:
   - Baixe seus dados em formato CSV
   - Use em outras análises

## 📦 Dependências

- streamlit
- drawdata
- pandas
- numpy
- matplotlib
- seaborn

## 🎨 Estatísticas Calculadas

Para cada classe e coordenada (X e Y):
- **Média**: Valor médio dos dados
- **Mediana**: Valor central
- **Desvio Padrão**: Dispersão dos dados
- **Variância**: Variabilidade dos dados
- **Mínimo**: Menor valor
- **Máximo**: Maior valor

## 💡 Dicas

- Desenhe pelo menos 10 pontos por classe para análises mais robustas
- Use diferentes classes para comparar distribuições
- Experimente padrões diferentes para ver como as estatísticas mudam

## 📝 Notas

- A aplicação calcula estatísticas em tempo real
- O KDE (Kernel Density Estimation) mostra a distribuição suavizada dos dados
- Linhas verticais nos histogramas indicam média (vermelho) e mediana (verde)
