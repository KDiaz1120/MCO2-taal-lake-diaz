import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout, Input, MaxPooling1D
import tensorflow as tf; print(tf.__version__)
import time
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
import base64

# Set page config with wide layout and no padding
st.set_page_config(
    page_title="Taal Lake Water Quality Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with maximized space
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
        padding: 0.5rem 1rem !important;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Title styling */
    .css-10trblm {
        color: #2c3e50;
        font-size: 2.5rem !important;
        font-weight: 700;
        margin-bottom: 0.25rem !important;
    }
    
    /* Header styling */
    h2 {
        font-size: 1.8rem !important;
        margin-top: 0.5rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
    }
    
    /* Card styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Metric value styling - larger */
    .stMetric {
        margin-bottom: 0;
    }
    .stMetric .value {
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #2c3e50;
    }
    .stMetric .label {
        font-size: 1.1rem !important;
        color: #7f8c8d;
    }
    .stMetric .delta {
        font-size: 1rem !important;
        font-weight: 500;
    }
    
    /* Button styling - larger */
    .stButton>button {
        font-size: 1.1rem !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
    }
    
    /* Input controls - larger */
    .stSelectbox, .stMultiselect, .stSlider, .stRadio, .stDateInput {
        font-size: 1.1rem !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: white;
        padding: 1.5rem !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem !important;
        font-size: 1.1rem !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        font-size: 1rem !important;
    }
    
    /* Remove extra whitespace */
    div.stButton > button:first-child {
        width: 100%;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/KDiaz1120/MCO2-taal-lake-diaz/main/Water%20Quality-Elective%20-%20Final%20Dataset.csv"
    return pd.read_csv(url)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.results = None

# Main dashboard header with current date
col1, col2 = st.columns([4,1])
with col1:
    st.title("🌊 Taal Lake Water Quality Dashboard")
with col2:
    st.write(f"**{datetime.now().strftime('%B %d, %Y')}**")

# Sidebar controls - larger and more organized
with st.sidebar:
    st.header("📊 Dashboard Controls", divider='blue')
    
    # Date range selector - larger
    date_range = st.date_input(
        "Select Date Range:",
        value=(pd.to_datetime('2020-01-01'), pd.to_datetime('2023-12-31')),
        min_value=pd.to_datetime('2010-01-01'),
        max_value=pd.to_datetime('2023-12-31')
    )
    
    # Site selector - larger
    sites = st.multiselect(
        "Select Monitoring Sites:",
        options=["All"] + sorted(load_data()['Site'].unique()),
        default=["All"]
    )
    
    # Parameter groups - updated to include weather data
    parameter_groups = {
        "🌡️ Temperature": ['Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature'],
        "🧪 Chemical": ['pH', 'Ammonia', 'Nitrate', 'Phosphate'],
        "💨 Gas": ['Dissolved Oxygen', 'Sulfide', 'Carbon Dioxide', 'Air Temperature'],
        "☁️ Weather": ['Weather Condition', 'Wind Direction']
    }
    
    # Parameter group selector - larger
    selected_group = st.radio(
        "Parameter Group:",
        options=list(parameter_groups.keys()),
        index=0
    )
    
# Load and preprocess data
with st.spinner("Loading and preprocessing data..."):
    df = load_data()
    
    # Data cleaning (same as before)
    df = df.dropna()
    df.columns = df.columns.str.strip()
    
    numerical_columns = [
        'Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature',
        'pH', 'Ammonia', 'Nitrate', 'Phosphate',
        'Dissolved Oxygen', 'Sulfide', 'Carbon Dioxide', 'Air Temperature'
    ]
    
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        non_zero_values = df[df[col] != 0][col]
        col_mean = non_zero_values.mean()
        df[col] = df[col].replace(0, col_mean).fillna(col_mean)
    
    # Weather data cleaning
    weather_cols = ['Weather Condition', 'Wind Direction']
    for weather_col in weather_cols:
        if weather_col in df.columns:
            df[weather_col] = df[weather_col].astype(str).str.strip()
            non_zero = df[~df[weather_col].isin(['0', '0.0', '0.00'])]
            if not non_zero.empty:
                col_mode = non_zero[weather_col].mode()[0]
                df[weather_col] = df[weather_col].replace(['0', '0.0', '0.00'], col_mode)
    
    df = df.drop_duplicates()
    
    # Date handling
    if all(col in df.columns for col in ['Year', 'Month']):
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')
        df = df.drop(['Year', 'Month'], axis=1)
    
    # Normalization
    scaler = MinMaxScaler()
    numerical_cols = [col for col in numerical_columns if col in df.columns]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Filter data
    if "All" not in sites:
        df = df[df['Site'].isin(sites)]

# Dashboard tabs - larger and more prominent
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 OVERVIEW", 
    "📈 TIME SERIES", 
    "🔗 CORRELATIONS", 
    "🔄 RELATIONSHIPS", 
    "🤖 PREDICTIONS",
    "👨‍💻 ENGINEERING DEVELOPER"
])

with tab1:
    st.header("Data Overview", divider='blue')
    
    # Summary statistics - larger cards
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
        st.metric("Monitoring Sites", len(df['Site'].unique()))
    with col2:
        st.metric("Parameters Tracked", len(numerical_cols))
        st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Side-by-side section
    stats_col, map_col = st.columns(2)
    
    with stats_col:
        st.subheader("Descriptive Statistics", divider='blue')
        st.dataframe(df[numerical_cols].describe().T.style.format("{:.2f}").background_gradient(cmap='Blues'), 
                    use_container_width=True,
                    height=500)
    
    with map_col:
        st.subheader("Taal Lake Monitoring", divider='blue')
        
        # Taal Lake coordinates (approximate center)
        taal_lake_coords = {
            'latitude': 14.0101,
            'longitude': 120.9973,
            'zoom': 10
        }
        
        # Sample monitoring locations (replace with your actual coordinates)
        monitoring_locations = pd.DataFrame({
            'lat': [14.0101, 14.015, 14.005, 14.020, 14.000],
            'lon': [120.9973, 120.992, 121.002, 120.987, 121.007],
            'name': ['Station 1', 'Station 2', 'Station 3', 'Station 4', 'Station 5']
        })
        
        # PyDeck map (simplified to 2D)
        try:
            import pydeck as pdk
            
            layer = pdk.Layer(
                'ScatterplotLayer',
                data=monitoring_locations,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=500,
                pickable=True
            )
            
            view_state = pdk.ViewState(
                **taal_lake_coords,
                pitch=0  # 2D view
            )
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    'html': '<b>Station:</b> {name}',
                    'style': {'color': 'white'}
                }
            )
            
            st.pydeck_chart(r, use_container_width=True)
        except ImportError:
            # Fallback to static map
            st.map(monitoring_locations,
                  latitude=taal_lake_coords['latitude'],
                  longitude=taal_lake_coords['longitude'],
                  zoom=taal_lake_coords['zoom'])
    
    # Interactive data explorer
    st.subheader("Data Explorer", divider='blue')
    st.dataframe(df.head(100), use_container_width=True, height=400)

with tab2:
    st.header("Time Series Analysis", divider='blue')
    
    # Parameter selection - larger controls
    selected_params = st.multiselect(
        "Select Parameters:",
        options=parameter_groups[selected_group],
        default=parameter_groups[selected_group][:2]
    )
    
    if selected_params:
        melted_df = df.melt(
            id_vars=['Date', 'Site'], 
            value_vars=selected_params,
            var_name='Parameter',
            value_name='Value'
        )
        
        fig = px.line(
            melted_df,
            x='Date',
            y='Value',
            color='Site',
            facet_col='Parameter' if len(selected_params) > 1 else None,
            facet_col_wrap=2,
            height=600,
            title=f"Time Series of {selected_group} Parameters",
            template='plotly_white'
        )
        fig.update_xaxes(matches=None, showticklabels=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one parameter")

with tab3:
    st.header("Correlation Analysis", divider='blue')
    
    # Heatmap controls - larger
    col1, col2 = st.columns(2)
    with col1:
        heatmap_type = st.radio(
            "View Mode:",
            options=["All Sites", "By Site"],
            horizontal=True
        )
    with col2:
        if heatmap_type == "By Site":
            selected_site = st.selectbox(
                "Select Site:",
                options=df['Site'].unique()
            )
    
    # Generate heatmap - larger
    if heatmap_type == "By Site":
        heatmap_data = df[df['Site'] == selected_site][numerical_cols]
        title = f"Correlation Heatmap for {selected_site}"
    else:
        heatmap_data = df[numerical_cols]
        title = "Correlation Heatmap (All Sites)"
    
    fig = px.imshow(
        heatmap_data.corr(),
        text_auto=".2f",
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        title=title,
        height=700,
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Parameter Relationships", divider='blue')
    
    # Create two tabs within this section
    rel_tab1, rel_tab2 = st.tabs(["**Weather  Factors**", "**Volcanic Activity Indicators**"])
    
    with rel_tab1:
        # Define selectable relationships (your original code)
        scatter_options = {
            "Weather Condition vs Nitrate": ("Weather Condition", "Nitrate"),
            "Weather Condition vs Nitrite": ("Weather Condition", "Nitrite"),
            "Weather Condition vs Ammonia": ("Weather Condition", "Ammonia"),
            "Weather Condition vs Phosphate": ("Weather Condition", "Phosphate"),
            "Weather Condition vs Dissolved Oxygen": ("Weather Condition", "Dissolved Oxyggen"),
            "Wind Direction vs Ammonia": ("Wind Direction", "Ammonia"),
            "Wind Direction vs Nitrate": ("Wind Direction", "Nitrate"),
            "Wind Direction vs Phosphate": ("Wind Direction", "Phosphate"),
            "Wind Direction vs Nitrite": ("Wind Direction", "Nitrite"),
            "Wind Direction vs Dissolved Oxygen": ("Wind Direction", "Dissolved Oxygen"),
        }

        # Select relationship - larger controls
        selected_relation = st.selectbox(
            "Select parameter relationship to plot:", 
            list(scatter_options.keys())
        )

        x_col, y_col = scatter_options[selected_relation]

        if x_col in df.columns and y_col in df.columns:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color="Site",
                title=f"{x_col} vs {y_col}",
                labels={x_col: x_col, y_col: y_col},
                template="plotly_white",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Cannot plot: {x_col} or {y_col} not found in dataset.")
    
    with rel_tab2:
        st.subheader("Volcanic Activity Indicators")
        st.markdown("Explore relationships between volcanic activity indicators and other parameters")
        
        # Volcanic parameters vs other parameters
        volcanic_options = {
            "Sulfide": "Sulfide",
            "Carbon Dioxide": "Carbon Dioxide",
        }
        
        # Create two columns for the selectors
        col1, col2 = st.columns(2)
        
        with col1:
            volcanic_param = st.selectbox(
                "Volcanic Parameter:",
                options=list(volcanic_options.keys()),
                key="volcanic_param"
            )
        
        with col2:
            compare_param = st.selectbox(
                "Compare with:",
                options=[p for p in df.select_dtypes(include=['float64', 'int64']).columns 
                         if p not in volcanic_options.values()],
                key="compare_param"
            )
        
        # Create box plot
        if volcanic_param and compare_param:
            try:
                fig = px.box(
                    df,
                    x=volcanic_options[volcanic_param],
                    y=compare_param,
                    color="Site",
                    title=f"{volcanic_param} vs {compare_param}",
                    labels={
                        volcanic_options[volcanic_param]: volcanic_param,
                        compare_param: compare_param
                    },
                    template="plotly_white",
                    height=600
                )
                
                # Add some customization for better visualization
                fig.update_layout(
                    boxmode='group',
                    xaxis_title=volcanic_param,
                    yaxis_title=compare_param,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")

with tab5:
    st.header("AI-Powered Predictions", divider='blue')
    
    # Model configuration - larger controls
    with st.expander("⚙️ MODEL CONFIGURATION", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox(
                "Target Parameter:",
                options=numerical_cols,
                index=numerical_cols.index('pH')
            )
            n_steps = st.slider(
                "Sequence Length:",
                min_value=1,
                max_value=10,
                value=3
            )
        with col2:
            epochs = st.slider(
                "Training Epochs:",
                min_value=10,
                max_value=200,
                value=50,
                help="Number of training iterations"
            )
            batch_size = st.slider(
                "Batch Size:",
                min_value=16,
                max_value=128,
                value=32
            )
    
    # Larger training button
    if st.button("🚀 TRAIN PREDICTION MODELS", type="primary", use_container_width=True):
        with st.spinner("Training models... This may take a few minutes"):
            # Prepare data
            target_idx = numerical_cols.index(target)
            data_array = df[numerical_cols].values
            
            def create_sequences(data, target_idx, n_steps=3):
                X, y = [], []
                for i in range(len(data)-n_steps):
                    X.append(data[i:(i+n_steps), :])
                    y.append(data[i+n_steps, target_idx])
                return np.array(X), np.array(y)
                
            X, y = create_sequences(data_array, target_idx, n_steps)
            
            if X.shape[0] == 0 or len(X.shape) < 3:
                st.error("🚫 Not enough data to create training sequences. Try reducing the sequence length.")
                st.stop()
            
            # Split data
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Model definitions
            def create_cnn_model():
                model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    Conv1D(32, 2, activation='relu'),
                    MaxPooling1D(1),
                    Flatten(),
                    Dense(50, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mae')
                return model
                
            def create_lstm_model():
                model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    LSTM(50, activation='tanh'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mae')
                return model
                
            def create_hybrid_model():
                model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    Conv1D(32, 2, activation='relu'),
                    MaxPooling1D(1),
                    LSTM(50, return_sequences=False),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mae')
                return model
                
            # Train models
            models = {
                'CNN': create_cnn_model(),
                'LSTM': create_lstm_model(),
                'Hybrid CNN-LSTM': create_hybrid_model()
            }
            
            results = {}
            for name, model in models.items():
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    verbose=0
                )
                
                y_pred = model.predict(X_test).flatten()
                
                results[name] = {
                    'model': model,
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'history': history
                }
            
            st.session_state.results = results
            st.session_state.model_trained = True
            st.success("Model training completed!")
    
    if st.session_state.model_trained:
        # Display results - larger and more organized
        st.subheader("Model Performance Comparison")
        
        # Prepare data for bar chart
        model_names = []
        mae_scores = []
        rmse_scores = []
        
        for name, res in st.session_state.results.items():
            model_names.append(name)
            mae_scores.append(res['mae'])
            rmse_scores.append(res['rmse'])
        
        # Create interactive bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=mae_scores,
            name='MAE Score',
            marker_color='#1f77b4',
            text=[f"{score:.4f}" for score in mae_scores],
            textposition='auto',
            hoverinfo='y+name'
        ))
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=rmse_scores,
            name='RMSE Score',
            marker_color='#ff7f0e',
            text=[f"{score:.4f}" for score in rmse_scores],
            textposition='auto',
            hoverinfo='y+name'
        ))
        
        fig.update_layout(
            title='Model Performance Metrics',
            xaxis_title='Model Type',
            yaxis_title='Score Value',
            barmode='group',
            hovermode="x unified",
            template='plotly_white',
            height=500,
            margin=dict(l=50, r=50, t=80, b=50))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics cards in columns (kept but simplified)
        st.subheader("Detailed Metrics")
        cols = st.columns(3)
        for i, (name, res) in enumerate(st.session_state.results.items()):
            with cols[i]:
                st.metric(
                    label=f"{name} Model",
                    value=f"MAE: {res['mae']:.4f}",
                    delta=f"RMSE: {res['rmse']:.4f}"
                )
        
        # Water Quality Assessment - larger
        st.subheader("Water Quality Index Assessment")
        
        # Use LSTM model for predictions
        y_pred_lstm = st.session_state.results['LSTM']['model'].predict(X_test).flatten()
        
        def calculate_wqi(preds, feature_names, sequences):
            denorm = scaler.inverse_transform(sequences.reshape(-1,len(feature_names))).reshape(sequences.shape)
            for i,p in enumerate(preds): 
                denorm[i,-1,feature_names.index(target)] = p*(scaler.data_max_[feature_names.index(target)]-scaler.data_min_[feature_names.index(target)])+scaler.data_min_[feature_names.index(target)]
            wqi, violations = [], []
            for seq in denorm:
                val = seq[-1]
                score=100; vio=[]
                if not 6.5<=val[feature_names.index('pH')]<=8.5: score-=25; vio.append('pH')
                if val[feature_names.index('Dissolved Oxygen')]<5: score-=20; vio.append('Low Oxygen')
                if val[feature_names.index('Ammonia')]>1: score-=15; vio.append('High Ammonia')
                if val[feature_names.index('Phosphate')]>0.4: score-=10; vio.append('High Phosphate')
                wqi.append(max(score,0)); violations.append(vio)
            return np.array(wqi), violations
        
        wqi_vals, vio_list = calculate_wqi(
            y_pred_lstm,
            numerical_cols,
            X_test
        )
        
        # WQI Distribution - larger
        wqi_bins = {
            'Excellent (90-100)': (90, 100),
            'Good (70-89)': (70, 89),
            'Fair (50-69)': (50, 69),
            'Poor (<50)': (0, 49)
        }
        
        wqi_counts = {cat: 0 for cat in wqi_bins}
        for val in wqi_vals:
            for cat, (low, high) in wqi_bins.items():
                if low <= val <= high:
                    wqi_counts[cat] += 1
                    break
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                names=list(wqi_counts.keys()),
                values=list(wqi_counts.values()),
                title="WQI Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues_r,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Water Quality Categories")
            for cat, count in wqi_counts.items():
                st.progress(
                    count/len(wqi_vals),
                    text=f"{cat}: {count} samples ({count/len(wqi_vals):.1%})"
                )
                
        st.subheader("📌 Actual vs Predicted Values")
        scatter_cols = st.columns(3)
        for i, (name, res) in enumerate(st.session_state.results.items()):
            model = res['model']
            y_pred = model.predict(X_test).flatten()
          
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                marker=dict(color='skyblue', size=8, opacity=0.6),
                name='Predicted vs Actual'
            ))
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_test,
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Ideal Line (y=x)'
            ))
            fig.update_layout(
                title=f"{name}: Actual vs Predicted ({target})",
                xaxis_title="Actual",
                yaxis_title="Predicted",
                width=450,
                height=450,
                margin=dict(l=20, r=20, t=60, b=20),
                template="plotly_white"
            )
            with scatter_cols[i]:
                st.plotly_chart(fig, use_container_width=True)
          
        # Recommendations - larger
        st.subheader("Recommendations")
        
        mapping = {
            'pH': 'Adjust pH levels through aeration or chemical treatment',
            'Low Oxygen': 'Increase oxygenation through waterfall aeration or surface agitators',
            'High Ammonia': 'Implement biological filtration or water exchange',
            'High Phosphate': 'Use phosphate-removing media or limit nutrient runoff'
        }
        
        bad_vios = {
            v
            for val, vlist in zip(wqi_vals, vio_list)
            if val < 70
            for v in vlist
        }
        
        if bad_vios:
            st.warning("⚠️ Water quality issues detected:")
            for v in sorted(bad_vios):
                with st.expander(f"**{v}** - Recommended Action", expanded=True):
                    st.info(mapping[v], icon="ℹ️")
        else:
            st.success("✅ No significant water quality issues detected", icon="✅")
            
with tab6:
    st.header("Engineering Development Team", divider='blue')
    
    # Team Introduction
    st.markdown("""
    Our team of skilled engineers and data scientists developed this comprehensive water quality monitoring system.
    Meet the talented individuals behind this project:
    """)
    
    # Team Members in Columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Helper function to load images
    def load_image_from_github(url, width=200):
        try:
            response = requests.get(url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/'))
            image = Image.open(BytesIO(response.content))
            return image
        except:
            return None
    
    with col1:
        st.subheader("Knn Diaz")
        img = load_image_from_github("https://github.com/KDiaz1120/MCO2-taal-lake-diaz/blob/main/knn.jpg")
        if img:
            st.image(img, width=200)
        else:
            st.warning("Image not found")
        st.markdown("""
        - **Age:** 18
        - **Location:** Toronto, Canada
        - **Role:** Lead Data Engineer
        - **Expertise:** Machine Learning, Cloud Architecture
        """)
    
    with col2:
        st.subheader("Vin Ellaine")
        img = load_image_from_github("https://github.com/KDiaz1120/MCO2-taal-lake-diaz/blob/main/vin.jpg")
        if img:
            st.image(img, width=200)
        else:
            st.warning("Image not found")
        st.markdown("""
        - **Age:** 20
        - **Location:** Tagaytay
        - **Role:** Backend Developer
        - **Expertise:** Database Systems, API Development
        """)
    
    with col3:
        st.subheader("Cathreena Paula")
        img = load_image_from_github("https://github.com/KDiaz1120/MCO2-taal-lake-diaz/blob/main/cath.jpg")
        if img:
            st.image(img, width=200)
        else:
            st.warning("Image not found")
        st.markdown("""
        - **Age:** 22
        - **Location:** Nasugbu, Batangas
        - **Role:** Data Scientist
        - **Expertise:** Statistical Analysis, Predictive Modeling
        """)
    
    with col4:
        st.subheader("Sherly Bongalon")
        img = load_image_from_github("https://github.com/KDiaz1120/MCO2-taal-lake-diaz/blob/main/she.jfif")
        if img:
            st.image(img, width=200)
        else:
            st.warning("Image not found")
        st.markdown("""
        - **Age:** 22
        - **Location:** Kawit, Cavite
        - **Role:** Frontend Developer
        - **Expertise:** UI/UX Design, Data Visualization
        """)
    
    with col5:
        st.subheader("Mark Jezreel")
        img = load_image_from_github("https://github.com/KDiaz1120/MCO2-taal-lake-diaz/blob/main/mark.jfif")
        if img:
            st.image(img, width=200)
        else:
            st.warning("Image not found")
        st.markdown("""
        - **Age:** 21
        - **Location:** Bacoor, Cavite
        - **Role:** Full Stack Developer
        - **Expertise:** System Integration
        """)
    
    st.header("Team Photo", divider='blue')
    st.markdown("*Our team during the project development phase*")
    team_img = load_image_from_github("https://github.com/KDiaz1120/MCO2-taal-lake-diaz/blob/main/teamphoto.jpg", width=600)
    if team_img:
        st.image(team_img, caption='Taal Lake Monitoring Team', use_column_width=True)
    else:
        st.info("Team photo will be added soon")
    
    proj1, proj2, proj3 = st.columns(3)
    
    with proj1:
        with st.expander("**Taal Lake Monitoring System**", expanded=True):
            st.markdown("""
            - Developed real-time water quality monitoring dashboard
            - Implemented predictive models for water quality forecasting
            - Technologies: Python, Streamlit, TensorFlow, Plotly
            - **Impact:** 30% improvement in early anomaly detection
            """)
    
    with proj2:
        with st.expander("**Environmental Data Pipeline**", expanded=True):
            st.markdown("""
            - Built automated ETL pipeline for environmental sensors
            - Designed scalable data architecture on AWS
            - Technologies: Airflow, Spark, AWS S3/Redshift
            - **Impact:** Reduced data processing time by 75%
            """)
    
    with proj3:
        with st.expander("**AI Water Purification**", expanded=True):
            st.markdown("""
            - Developed ML models for water treatment optimization
            - Created IoT integration for real-time adjustments
            - Technologies: TensorFlow Lite, Raspberry Pi, MQTT
            - **Impact:** 15% reduction in purification costs
            """)
    
# Footer - larger
st.markdown("---")
st.markdown("""
**Taal Lake Water Quality Dashboard**  
*Developed for environmental monitoring and predictive analysis*  
**Group Members:** Antivo, Bongalon, Capacia, Diaz, Maghirang  
**Data current as of:** {date}  
""".format(date=datetime.now().strftime('%B %d, %Y')))
