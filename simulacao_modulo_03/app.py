import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import json

# Configuração da página
st.set_page_config(page_title="Simulação de Dados", layout="wide")

# Inicializar dados no session_state
if 'canvas_data' not in st.session_state:
    st.session_state.canvas_data = None
if 'analysis_triggered' not in st.session_state:
    st.session_state.analysis_triggered = False

# Título
st.title("📊 Simulação de Dados Interativa com DrawData")
st.markdown("Desenhe seus dados diretamente aqui e gere estatísticas!")

# Sidebar
with st.sidebar:
    st.header("📝 Como Usar")
    st.markdown("""
    1. 🎨 Desenhe pontos no gráfico abaixo
    2. 🎨 Use os botões de cor para mudar de classe
    3. 📊 Clique em **"Gerar Análise"**
    """)
    st.info("💡 Desenhe pelo menos 10 pontos por classe")
    
    if st.button("🗑️ Limpar Dados", use_container_width=True):
        st.session_state.canvas_data = None
        st.session_state.analysis_triggered = False
        st.rerun()

# Renderizar o DrawData Widget
st.header("🎨 Desenhe seus Dados")

# Criar HTML personalizado com DrawData integrado que envia dados ao Streamlit
drawdata_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
        #app { width: 100%; height: 550px; }
    </style>
</head>
<body>
    <div id="app"></div>
    
    <script type="module">
        import { createApp } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js'
        
        const app = createApp({
            data() {
                return {
                    points: [],
                    currentClass: 0,
                    classes: [
                        { id: 0, color: '#FF6B6B', name: 'Classe 0' },
                        { id: 1, color: '#4ECDC4', name: 'Classe 1' },
                        { id: 2, color: '#45B7D1', name: 'Classe 2' },
                        { id: 3, color: '#FFA07A', name: 'Classe 3' }
                    ]
                }
            },
            methods: {
                addPoint(event) {
                    const rect = event.target.getBoundingClientRect();
                    const x = (event.clientX - rect.left) / rect.width;
                    const y = 1 - (event.clientY - rect.top) / rect.height;
                    
                    this.points.push({
                        x: x,
                        y: y,
                        label: this.currentClass
                    });
                    
                    this.drawCanvas();
                    this.saveData();
                    this.sendToStreamlit();
                },
                drawCanvas() {
                    const canvas = document.getElementById('canvas');
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // Grid
                    ctx.strokeStyle = '#e0e0e0';
                    ctx.lineWidth = 1;
                    for(let i = 0; i <= 10; i++) {
                        ctx.beginPath();
                        ctx.moveTo(i * canvas.width / 10, 0);
                        ctx.lineTo(i * canvas.width / 10, canvas.height);
                        ctx.stroke();
                        
                        ctx.beginPath();
                        ctx.moveTo(0, i * canvas.height / 10);
                        ctx.lineTo(canvas.width, i * canvas.height / 10);
                        ctx.stroke();
                    }
                    
                    // Points
                    this.points.forEach(point => {
                        const classInfo = this.classes.find(c => c.id === point.label);
                        ctx.fillStyle = classInfo.color;
                        ctx.beginPath();
                        ctx.arc(
                            point.x * canvas.width,
                            (1 - point.y) * canvas.height,
                            8,
                            0,
                            2 * Math.PI
                        );
                        ctx.fill();
                        ctx.strokeStyle = '#000';
                        ctx.lineWidth = 2;
                        ctx.stroke();
                    });
                },
                saveData() {
                    localStorage.setItem('drawdata_points', JSON.stringify(this.points));
                },
                sendToStreamlit() {
                    // Enviar dados para o Streamlit via window.parent
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        data: this.points
                    }, '*');
                },
                loadData() {
                    const saved = localStorage.getItem('drawdata_points');
                    if(saved) {
                        this.points = JSON.parse(saved);
                        this.drawCanvas();
                        this.sendToStreamlit();
                    }
                },
                clearData() {
                    this.points = [];
                    localStorage.removeItem('drawdata_points');
                    this.drawCanvas();
                    this.sendToStreamlit();
                },
                exportCSV() {
                    let csv = 'x,y,label\\n';
                    this.points.forEach(p => {
                        csv += `${p.x.toFixed(3)},${p.y.toFixed(3)},${p.label}\\n`;
                    });
                    
                    const blob = new Blob([csv], { type: 'text/csv' });
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'data.csv';
                    a.click();
                }
            },
            mounted() {
                this.loadData();
                this.drawCanvas();
                
                // Carregar dados quando a janela ganhar foco
                window.addEventListener('focus', () => {
                    this.loadData();
                });
            },
            template: `
                <div>
                    <div style="margin-bottom: 15px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                        <button 
                            v-for="cls in classes" 
                            :key="cls.id"
                            @click="currentClass = cls.id"
                            :style="{
                                padding: '10px 20px',
                                border: currentClass === cls.id ? '3px solid #000' : '2px solid #ccc',
                                borderRadius: '5px',
                                backgroundColor: cls.color,
                                color: '#fff',
                                fontWeight: 'bold',
                                cursor: 'pointer'
                            }">
                            {{ cls.name }}
                        </button>
                        <button 
                            @click="clearData"
                            style="padding: 10px 20px; background: #ff4444; color: white; border: none; border-radius: 5px; cursor: pointer; margin-left: auto;">
                            🗑️ Limpar
                        </button>
                    </div>
                    <canvas 
                        id="canvas" 
                        width="700" 
                        height="450"
                        @click="addPoint"
                        style="border: 2px solid #333; border-radius: 8px; cursor: crosshair; background: white;">
                    </canvas>
                    <div style="margin-top: 10px; color: #666; font-weight: bold;">
                        📍 Pontos desenhados: {{ points.length }}
                    </div>
                </div>
            `
        })
        
        app.mount('#app')
    </script>
</body>
</html>
"""

canvas_data = components.html(drawdata_html, height=650, scrolling=False)

# Armazenar dados no session_state
if canvas_data:
    st.session_state.canvas_data = canvas_data

st.markdown("---")

# Mostrar informações sobre os dados
if st.session_state.canvas_data:
    try:
        df_preview = pd.DataFrame(st.session_state.canvas_data)
        if len(df_preview) > 0:
            st.info(f"✅ {len(df_preview)} pontos detectados! Clique no botão abaixo para gerar a análise.")
    except:
        pass

# Botão para gerar análise
if st.button("📊 Gerar Análise Completa", type="primary", use_container_width=True):
    try:
        df = None
        
        # Tentar obter dados do canvas
        if st.session_state.canvas_data:
            df = pd.DataFrame(st.session_state.canvas_data)
        
        if df is not None and len(df) > 0:
            # Verificar colunas
            if 'x' in df.columns and 'y' in df.columns and 'label' in df.columns:
                st.success(f"✅ {len(df)} pontos em {df['label'].nunique()} classe(s)!")
                
                # Mostrar dados brutos
                with st.expander("🔍 Ver Dados Brutos"):
                    st.dataframe(df, use_container_width=True)
                
                st.markdown("---")
                
                # Análise por classe
                st.header("📈 Análise Estatística por Classe")
                
                classes = sorted(df['label'].unique())
                tabs = st.tabs([f"Classe {cls}" for cls in classes])
                
                for idx, classe in enumerate(classes):
                    with tabs[idx]:
                        df_classe = df[df['label'] == classe]
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader(f"📊 Estatísticas - Classe {classe}")
                            
                            # Estatísticas X
                            st.markdown("**Coordenada X:**")
                            stats_x = {
                                "Média": df_classe['x'].mean(),
                                "Mediana": df_classe['x'].median(),
                                "Desvio Padrão": df_classe['x'].std(),
                                "Variância": df_classe['x'].var(),
                                "Mínimo": df_classe['x'].min(),
                                "Máximo": df_classe['x'].max(),
                                "N° Pontos": len(df_classe)
                            }
                            
                            cols = st.columns(3)
                            for i, (key, value) in enumerate(stats_x.items()):
                                with cols[i % 3]:
                                    if key == "N° Pontos":
                                        st.metric(key, f"{value}")
                                    else:
                                        st.metric(key, f"{value:.4f}")
                            
                            st.markdown("---")
                            
                            # Estatísticas Y
                            st.markdown("**Coordenada Y:**")
                            stats_y = {
                                "Média": df_classe['y'].mean(),
                                "Mediana": df_classe['y'].median(),
                                "Desvio Padrão": df_classe['y'].std(),
                                "Variância": df_classe['y'].var(),
                                "Mínimo": df_classe['y'].min(),
                                "Máximo": df_classe['y'].max()
                            }
                            
                            cols = st.columns(3)
                            for i, (key, value) in enumerate(stats_y.items()):
                                with cols[i % 3]:
                                    st.metric(key, f"{value:.4f}")
                        
                        with col2:
                            st.subheader("📉 Histogramas com KDE")
                            
                            # Criar figura
                            fig, axes = plt.subplots(2, 1, figsize=(10, 10))
                            
                            # Histograma X com KDE
                            axes[0].hist(df_classe['x'], bins=20, alpha=0.6, 
                                        color='skyblue', edgecolor='black', 
                                        density=True, label='Histograma')
                            
                            if len(df_classe) > 1 and df_classe['x'].std() > 0:
                                df_classe['x'].plot(kind='kde', ax=axes[0], 
                                                   color='darkblue', linewidth=2, label='KDE')
                            
                            axes[0].axvline(stats_x['Média'], color='red', 
                                          linestyle='--', linewidth=2, 
                                          label=f"Média: {stats_x['Média']:.2f}")
                            axes[0].axvline(stats_x['Mediana'], color='green', 
                                          linestyle='--', linewidth=2, 
                                          label=f"Mediana: {stats_x['Mediana']:.2f}")
                            
                            axes[0].set_xlabel('X', fontsize=12)
                            axes[0].set_ylabel('Densidade', fontsize=12)
                            axes[0].set_title(f'Distribuição de X - Classe {classe}', 
                                            fontsize=14, fontweight='bold')
                            axes[0].legend()
                            axes[0].grid(True, alpha=0.3)
                            
                            # Histograma Y com KDE
                            axes[1].hist(df_classe['y'], bins=20, alpha=0.6, 
                                        color='lightcoral', edgecolor='black', 
                                        density=True, label='Histograma')
                            
                            if len(df_classe) > 1 and df_classe['y'].std() > 0:
                                df_classe['y'].plot(kind='kde', ax=axes[1], 
                                                   color='darkred', linewidth=2, label='KDE')
                            
                            axes[1].axvline(stats_y['Média'], color='red', 
                                          linestyle='--', linewidth=2, 
                                          label=f"Média: {stats_y['Média']:.2f}")
                            axes[1].axvline(stats_y['Mediana'], color='green', 
                                          linestyle='--', linewidth=2, 
                                          label=f"Mediana: {stats_y['Mediana']:.2f}")
                            
                            axes[1].set_xlabel('Y', fontsize=12)
                            axes[1].set_ylabel('Densidade', fontsize=12)
                            axes[1].set_title(f'Distribuição de Y - Classe {classe}', 
                                            fontsize=14, fontweight='bold')
                            axes[1].legend()
                            axes[1].grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                
                # Comparação entre classes
                st.markdown("---")
                st.header("🔄 Comparação entre Classes")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
                    
                    for idx, classe in enumerate(classes):
                        df_classe = df[df['label'] == classe]
                        ax.scatter(df_classe['x'], df_classe['y'], 
                                  label=f'Classe {classe}', alpha=0.6, s=100, 
                                  color=colors[idx], edgecolors='black', linewidth=0.5)
                    
                    ax.set_xlabel('X', fontsize=12)
                    ax.set_ylabel('Y', fontsize=12)
                    ax.set_title('Todas as Classes', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.subheader("📋 Resumo Estatístico")
                    
                    summary_data = []
                    for classe in classes:
                        df_classe = df[df['label'] == classe]
                        summary_data.append({
                            'Classe': classe,
                            'N° Pontos': len(df_classe),
                            'Média X': df_classe['x'].mean(),
                            'Média Y': df_classe['y'].mean(),
                            'Std X': df_classe['x'].std(),
                            'Std Y': df_classe['y'].std(),
                            'Var X': df_classe['x'].var(),
                            'Var Y': df_classe['y'].var()
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df.style.format({
                        'Média X': '{:.4f}',
                        'Média Y': '{:.4f}',
                        'Std X': '{:.4f}',
                        'Std Y': '{:.4f}',
                        'Var X': '{:.4f}',
                        'Var Y': '{:.4f}'
                    }), use_container_width=True)
                
                # Download
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.subheader("💾 Exportar")
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Baixar CSV",
                        data=csv,
                        file_name="dados_simulados.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.error("⚠️ Dados inválidos. Colunas necessárias: x, y, label")
        else:
            st.warning("⚠️ Nenhum dado encontrado! Desenhe alguns pontos no gráfico acima.")
    
    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")
        st.info("💡 Desenhe alguns pontos no gráfico acima e tente novamente")
        with st.expander("Ver detalhes do erro"):
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><em>Desenvolvido com ❤️ usando Streamlit e DrawData</em></p>
</div>
""", unsafe_allow_html=True)
