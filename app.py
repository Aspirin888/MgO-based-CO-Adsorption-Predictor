import streamlit as st
import pandas as pd
import numpy as np
import pickle

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
    'Dopant or modifier': 'Dopant or modifier',
    'Pressure (Mpa)': 'Pressure (MPa)',
    'temputure ': 'Temperature (°C)',
    'time (min)': 'Time (min)',
    'Morphology': 'Morphology'
}

# 根据您提供的映射表，创建编码到原始值的映射
dopant_encoding_mapping = {
    0: 'No dopant',
    1: '0.005K',
    2: '0.005Li',
    3: '0.005Na',
    4: '0.01MC',
    5: '0.02Fe-AMS',
    6: '0.05Mn',
    7: '0.09(0.18Li0.3Na0.52K)NO3',
    8: '0.09(0.18Li0.52Na0.3K)NO3',
    9: '0.09(0.3Li0.18Na0.52K)NO3',
    10: '0.09(0.3Li0.52Na0.18K)NO3',
    11: '0.09(0.52Li0.18Na0.3K)NO3',
    12: '0.09(0.52Li0.3Na0.18K)NO3',
    13: '0.11MC',
    14: '0.14g SDS',
    15: '0.15Mn',
    16: '0.15Na-k',
    17: '0.18g CTAB',
    18: '0.1Fe-AMS',
    19: '0.1MC',
    20: '0.1Mn',
    21: '0.2Fe-AMS',
    22: '0.33Al2O3',
    23: '0.33CeO2',
    24: '0.4Fe-AMS',
    25: '0.5(0.375Li0.075Na0.55K)-0.15(NaK)',
    26: '0.66g Tween-60',
    27: '0.6Fe-AMS',
    28: '0.8Fe-AMS',
    29: '10.21C',
    30: '10.58C',
    31: '9.66C',
    32: 'AMS',
    33: 'APTES',
    34: 'CTAB',
    35: 'DETA',
    36: 'P-123',
    37: 'PEI',
    38: 'PVP',
    39: 'SDS'
}

# 为了用户选择方便，也创建原始值到编码的映射
dopant_original_to_encoding = {v: k for k, v in dopant_encoding_mapping.items()}

# 形态选项映射到原始特征名
morphology_mapping = {
    'Block': 'Morphology_Block',
    'Flake': 'Morphology_Flake',
    'Flower-like': 'Morphology_Flower-like',
    'Granular': 'Morphology_Granular',
    'Rod-shaped': 'Morphology_Rod-shaped',
    'Cube': 'Morphology_cube',
    'Nest-like': 'Morphology_nestlike',
    'Sphere': 'Morphology_sphere',
    'Other (Morphology_0)': 'Morphology_0'
}

# 模型需要的特征顺序
feature_order = [
    'SBET (m2/g)', 'Vpore (cm3/g)', 'dpore (nm)', 'Particle size (nm)',
    'Dopant or modifier', 'Pressure (Mpa)', 'temputure ', 'time (min)',
    'Morphology_0', 'Morphology_Block', 'Morphology_Flake', 'Morphology_Flower-like',
    'Morphology_Granular', 'Morphology_Rod-shaped', 'Morphology_cube',
    'Morphology_nestlike', 'Morphology_sphere'
]

# 应用标题
st.title("🌿 MgO-based CO₂ Adsorption Capacity Predictor")
st.markdown("Predict the CO₂ adsorption capacity of MgO-based adsorbents using machine learning")

# 侧边栏
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app predicts the **CO₂ uptake capacity** of magnesium oxide-based adsorbents 
    using a trained ETFM model.
    
    Adjust the parameters in the main panel and get instant predictions!
    """)
    
    st.header("Model Info")
    st.markdown("""
    - **Algorithm**: ETFM model
    - **Target**: CO₂ adsorption capacity (mmol/g)
    - **Features**: Material properties and experimental conditions
  
    """)
    
    st.header("Dopant/Modifier Info")
    st.markdown("""
    The **Dopant or modifier** field shows the original chemical notation.
    """)
    
    st.header("Morphology Options")
    st.markdown("""
    Available morphology types:
    - Block, Flake, Flower-like, Granular
    - Rod-shaped, Cube, Nest-like, Sphere
    - Other: For unspecified morphologies
    """)

# 主内容区域 - 分为两列
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔬 Material Parameters")
    
    # 创建三列用于输入字段
    subcol1, subcol2, subcol3 = st.columns(3)
    
    with subcol1:
        st.subheader("Physical Properties")
        sbet = st.number_input(feature_labels['SBET (m2/g)'], min_value=50.0, max_value=500.0, value=100.0, step=10.0, 
                              help="Specific surface area in m²/g")
        vpore = st.number_input(feature_labels['Vpore (cm3/g)'], min_value=0.1, max_value=1.0, value=0.2, step=0.1,
                               help="Pore volume in cm³/g")
        dpore = st.number_input(feature_labels['dpore (nm)'], min_value=5.0, max_value=60.0, value=10.0, step=1.0,
                               help="Pore diameter in nm")
        particle_size = st.number_input(feature_labels['Particle size (nm)'], min_value=10.0, max_value=8000.0, value=50.0, step=10.0,
                                       help="Particle size in nm")
    
    with subcol2:
        st.subheader("Experimental Conditions")
        pressure = st.number_input(feature_labels['Pressure (Mpa)'], min_value=0.05, max_value=1.20, value=0.1, step=0.02,
                                  help="Pressure in MPa")
        temperature = st.number_input(feature_labels['temputure '], min_value=25.0, max_value=500.0, value=25.0, step=5.0,
                                     help="Temperature in °C")
        time_min = st.number_input(feature_labels['time (min)'], min_value=0.0, max_value=350.0, value=300.0, step=10.0,
                                  help="Time in minutes")
        
        # 掺杂/修饰选择
        st.subheader("Dopant/Modifier")
        dopant_options = list(dopant_encoding_mapping.values())
        selected_dopant = st.selectbox(
            feature_labels['Dopant or modifier'], 
            options=dopant_options,
            index=0,
            help="Select the dopant/modifier (original chemical notation)"
        )
        dopant_encoded_value = dopant_original_to_encoding[selected_dopant]
    
    with subcol3:
        st.subheader("Morphology")
        
        # 形态选择
        morphology_options = list(morphology_mapping.keys())
        selected_morphology = st.selectbox(
            feature_labels['Morphology'], 
            options=morphology_options,
            index=1,
            help="Select one morphology type"
        )
        
        # 显示形态提示
        with st.expander("ℹ️ Morphology Information"):
            st.markdown("""
            **Morphology Types (One-hot encoded):**
            - **Block**: Regular block-shaped particles
            - **Flake**: Thin, flat plate-like structures  
            - **Flower-like**: Complex hierarchical structures
            - **Granular**: Irregular granular particles
            - **Rod-shaped**: Elongated rod-like structures
            - **Cube**: Cubic-shaped particles
            - **Nest-like**: Nest-like porous structures
            - **Sphere**: Spherical particles
            - **Other**: Unspecified morphology
            """)

with col2:
    st.header("📊 Prediction Results")
    
    # 准备形态特征
    morphology_features = {morphology_mapping[morph]: 0 for morph in morphology_options}
    morphology_features[morphology_mapping[selected_morphology]] = 1
    
    # 创建特征字典
    input_features = {
        'SBET (m2/g)': sbet,
        'Vpore (cm3/g)': vpore,
        'dpore (nm)': dpore,
        'Particle size (nm)': particle_size,
        'Dopant or modifier': dopant_encoded_value,
        'Pressure (Mpa)': pressure,
        'temputure ': temperature,
        'time (min)': time_min,
        'Morphology_0': morphology_features['Morphology_0'],
        'Morphology_Block': morphology_features['Morphology_Block'],
        'Morphology_Flake': morphology_features['Morphology_Flake'],
        'Morphology_Flower-like': morphology_features['Morphology_Flower-like'],
        'Morphology_Granular': morphology_features['Morphology_Granular'],
        'Morphology_Rod-shaped': morphology_features['Morphology_Rod-shaped'],
        'Morphology_cube': morphology_features['Morphology_cube'],
        'Morphology_nestlike': morphology_features['Morphology_nestlike'],
        'Morphology_sphere': morphology_features['Morphology_sphere']
    }
    
    # 转换为DataFrame
    input_df = pd.DataFrame([input_features])[feature_order]
    
    # 预测按钮
    if st.button("Predict CO₂ Adsorption Capacity", type="primary", use_container_width=True):
        try:
            model = load_model()
            prediction = model.predict(input_df)[0]
            
            # 显示结果
            st.metric(
                label="Predicted CO₂ Adsorption Capacity",
                value=f"{prediction:.3f} mmol/g",
                delta=None
            )
            
            # 结果解释
            st.success(f"The predicted CO₂ adsorption capacity is **{prediction:.3f} mmol/g**")
            
            # 可视化条
            max_capacity = 5.0
            percentage = min((prediction / max_capacity) * 100, 100)
            
            st.progress(int(percentage)/100, text=f"Capacity: {prediction:.3f} mmol/g ({percentage:.1f}% of max)")
            
            # 显示输入特征值
            with st.expander("🔍 View Model Input Details"):
                st.markdown("**Feature values sent to model:**")
                st.dataframe(input_df, use_container_width=True)
                
                # 显示编码信息
                st.markdown("**Encoding information:**")
                encoding_info = {
                    'Dopant/Modifier': f"'{selected_dopant}' → {dopant_encoded_value}",
                    'Morphology': f"'{selected_morphology}' → {morphology_mapping[selected_morphology]} = 1"
                }
                st.json(encoding_info)
            
        except FileNotFoundError:
            st.error("Model file 'CatBoost.pkl' not found. Please ensure it's in the same directory as this app.")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.info("Please check the input parameters and try again.")
    
    # 输入特征摘要
    st.header("📝 Input Summary")
    summary_data = {
        'Parameter': [
            feature_labels['SBET (m2/g)'],
            feature_labels['Vpore (cm3/g)'],
            feature_labels['dpore (nm)'],
            feature_labels['Particle size (nm)'],
            'Dopant/modifier',
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
            f"{selected_dopant} (encoded: {dopant_encoded_value})",
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
    """
    ### 📋 **Important Notes:**
    1. The model uses **exact feature encodings** from the training data
    2. **Dopant/Modifier** values are internally converted to labels 0-39
    3. **Morphology** uses one-hot encoding
    """
)

# 响应式设计
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    .stMetric {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 12px;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)






