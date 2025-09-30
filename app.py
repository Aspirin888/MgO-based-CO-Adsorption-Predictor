import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# 页面配置
st.set_page_config(
    page_title="MgO-based CO₂ Adsorption Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载模型
@st.cache_resource
def load_model():
    with open('CatBoost.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# 特征标签映射
feature_labels = {
    'SBET (m2/g)': 'S$_{BET}$ (m²/g)',
    'Vpore (cm3/g)': 'V$_{pore}$ (cm³/g)',
    'dpore (nm)': 'd$_{pore}$ (nm)',
    'Particle size (nm)': 'Particle size (nm)',
    'Dopant or modifier': 'Doped-modified',
    'Pressure (Mpa)': 'Pressure (MPa)',
    'temputure ': 'Temperature (°C)',
    'time (min)': 'Time (min)',
    'Morphology': 'Morphology'
}

# 形态选项映射到原始特征名
morphology_mapping = {
    'Block': 'Morphology_Block',
    'Flake': 'Morphology_Flake',
    'Flower-like': 'Morphology_Flower-like',
    'Granular': 'Morphology_Granular',
    'Rod-shaped': 'Morphology_Rod-shaped',
    'Cube': 'Morphology_cube',
    'Nest-like': 'Morphology_nestlike',
    'Sphere': 'Morphology_sphere'
}

# 应用标题
st.title("🌿 MgO-based CO₂ Adsorption Capacity Predictor")
st.markdown("Predict the CO₂ adsorption capacity of MgO-based adsorbents using machine learning")

# 侧边栏
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app predicts the **CO₂ uptake capacity** of magnesium oxide-based adsorbents 
    using a trained CatBoost regression model.
    
    Adjust the parameters in the main panel and get instant predictions!
    """)
    
    st.header("Model Info")
    st.markdown("""
    - **Algorithm**: CatBoost Regressor
    - **Target**: CO₂ adsorption capacity (mmol/g)
    - **Features**: Material properties and experimental conditions
    """)
    
    st.header("Morphology Options")
    st.markdown("""
    Available morphology types:
    - Block
    - Flake
    - Flower-like
    - Granular
    - Rod-shaped
    - Cube
    - Nest-like
    - Sphere
    """)

# 主内容区域 - 分为两列
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔬 Material Parameters")
    
    # 创建三列用于输入字段
    subcol1, subcol2, subcol3 = st.columns(3)
    
    with subcol1:
        st.subheader("Physical Properties")
        sbet = st.number_input(feature_labels['SBET (m2/g)'], min_value=0.0, max_value=1000.0, value=150.0, step=10.0, 
                              help="Specific surface area in m²/g")
        vpore = st.number_input(feature_labels['Vpore (cm3/g)'], min_value=0.0, max_value=5.0, value=0.5, step=0.1,
                               help="Pore volume in cm³/g")
        dpore = st.number_input(feature_labels['dpore (nm)'], min_value=0.0, max_value=50.0, value=10.0, step=1.0,
                               help="Pore diameter in nm")
        particle_size = st.number_input(feature_labels['Particle size (nm)'], min_value=0.0, max_value=500.0, value=50.0, step=10.0,
                                       help="Particle size in nm")
    
    with subcol2:
        st.subheader("Experimental Conditions")
        pressure = st.number_input(feature_labels['Pressure (Mpa)'], min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                  help="Pressure in MPa")
        temperature = st.number_input(feature_labels['temputure '], min_value=0.0, max_value=500.0, value=25.0, step=5.0,
                                     help="Temperature in °C")
        time_min = st.number_input(feature_labels['time (min)'], min_value=0.0, max_value=300.0, value=60.0, step=10.0,
                                  help="Time in minutes")
        
        # 掺杂/修饰选择
        dopant_options = {'No': 0, 'Yes': 1}
        dopant = st.selectbox(feature_labels['Dopant or modifier'], options=list(dopant_options.keys()),
                             help="Whether the material is doped or modified")
        dopant_value = dopant_options[dopant]
    
    with subcol3:
        st.subheader("Morphology")
        
        # 形态选择 - 合并为一个特征
        morphology_options = list(morphology_mapping.keys())
        selected_morphology = st.selectbox(
            feature_labels['Morphology'], 
            options=morphology_options,
            help="Select one morphology type from: " + ", ".join(morphology_options)
        )
        
        # 显示形态提示
        with st.expander("ℹ️ Morphology Information"):
            st.markdown("""
            **Available Morphology Types:**
            - **Block**: Regular block-shaped particles
            - **Flake**: Thin, flat plate-like structures
            - **Flower-like**: Complex flower-like hierarchical structures
            - **Granular**: Irregular granular particles
            - **Rod-shaped**: Elongated rod-like structures
            - **Cube**: Cubic-shaped particles
            - **Nest-like**: Nest-like porous structures
            - **Sphere**: Spherical particles
            """)

with col2:
    st.header("📊 Prediction Results")
    
    # 准备形态特征 - 所有形态特征初始化为0，选中的设为1
    morphology_features = {morphology_mapping[morph]: 0 for morph in morphology_options}
    morphology_features[morphology_mapping[selected_morphology]] = 1
    
    # 创建特征字典
    input_features = {
        'SBET (m2/g)': sbet,
        'Vpore (cm3/g)': vpore,
        'dpore (nm)': dpore,
        'Particle size (nm)': particle_size,
        'Dopant or modifier': dopant_value,
        'Pressure (Mpa)': pressure,
        'temputure ': temperature,
        'time (min)': time_min,
        **morphology_features
    }
    
    # 转换为DataFrame
    feature_order = [
        'SBET (m2/g)', 'Vpore (cm3/g)', 'dpore (nm)', 'Particle size (nm)',
        'Dopant or modifier', 'Pressure (Mpa)', 'temputure ', 'time (min)',
        'Morphology_Block', 'Morphology_Flake', 'Morphology_Flower-like',
        'Morphology_Granular', 'Morphology_Rod-shaped', 'Morphology_cube',
        'Morphology_nestlike', 'Morphology_sphere'
    ]
    
    input_df = pd.DataFrame([input_features])[feature_order]
    
    # 预测按钮
    if st.button("Predict CO₂ Adsorption Capacity", type="primary", use_container_width=True):
        try:
            model = load_model()
            prediction = model.predict(input_df)[0]
            
            # 显示结果
            st.metric(
                label="Predicted CO₂ Adsorption Capacity",
                value=f"{prediction:.2f} mmol/g",
                delta=None
            )
            
            # 结果解释
            st.success(f"The predicted CO₂ adsorption capacity is **{prediction:.2f} mmol/g**")
            
            # 可视化条
            max_capacity = 10.0  # 可根据您的数据范围调整
            percentage = min((prediction / max_capacity) * 100, 100)
            
            st.progress(int(percentage)/100, text=f"Capacity: {prediction:.2f} mmol/g")
            
            # 显示输入特征值（用于调试或确认）
            with st.expander("View Input Features"):
                st.dataframe(input_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.info("Please ensure the CatBoost.pkl model file is in the same directory as this app.")
    
    # 输入特征摘要
    st.header("📝 Input Summary")
    summary_data = {
        'Parameter': [
            feature_labels['SBET (m2/g)'],
            feature_labels['Vpore (cm3/g)'],
            feature_labels['dpore (nm)'],
            feature_labels['Particle size (nm)'],
            feature_labels['Dopant or modifier'],
            feature_labels['Pressure (Mpa)'],
            feature_labels['temputure '],
            feature_labels['time (min)'],
            feature_labels['Morphology']
        ],
        'Value': [
            f"{sbet} m²/g",
            f"{vpore} cm³/g", 
            f"{dpore} nm",
            f"{particle_size} nm",
            dopant,
            f"{pressure} MPa",
            f"{temperature} °C",
            f"{time_min} min",
            selected_morphology
        ]
    }
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# 底部信息
st.markdown("---")
st.markdown(
    "**Note**: This prediction is based on a machine learning model trained on experimental data. "
    "Results should be validated with laboratory experiments."
)

# 响应式设计
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stExpander {
        border: 1px solid #e6e6e6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)