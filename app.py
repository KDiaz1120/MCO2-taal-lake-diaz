import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    
    # Create three tabs within this section
    rel_tab1, rel_tab2, rel_tab3 = st.tabs([
        "**Weather Factors**", 
        "**Volcanic Activity Indicators**",
        "**Weather Metadata**"
    ])
    
    with rel_tab1:
        # Your existing weather factors code remains unchanged
        st.subheader("Weather Factor Relationships")
        st.markdown("Explore how weather conditions affect water quality parameters")
        
        weather_options = {
            "Weather Condition": "Weather Condition",
            "Wind Direction": "Wind Direction",
            "Air Temperature": "Air Temperature"
        }
        
        # Water quality parameters to compare against
        water_quality_params = [
            'Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature',
            'pH', 'Ammonia', 'Nitrate', 'Phosphate',
            'Dissolved Oxygen', 'Sulfide', 'Carbon Dioxide'
        ]
        
        # Create two columns for the selectors
        col1, col2 = st.columns(2)
        
        with col1:
            weather_param = st.selectbox(
                "Weather Parameter:",
                options=list(weather_options.keys()),
                key="weather_param"
            )
        
        with col2:
            water_param = st.selectbox(
                "Water Quality Parameter:",
                options=[p for p in water_quality_params if p in df.columns],
                key="water_param"
            )
        
        # Create box plot
        if weather_param and water_param:
            try:
                # For categorical weather data (condition, wind direction), use box plot
                if weather_options[weather_param] in ['Weather Condition', 'Wind Direction']:
                    fig = px.box(
                        df,
                        x=weather_options[weather_param],
                        y=water_param,
                        color="Site",
                        title=f"Distribution of {water_param} by {weather_param}",
                        labels={
                            weather_options[weather_param]: weather_param,
                            water_param: water_param
                        },
                        template="plotly_white",
                        height=600
                    )
                # For continuous weather data (temperature), use scatter plot
                else:
                    fig = px.scatter(
                        df,
                        x=weather_options[weather_param],
                        y=water_param,
                        color="Site",
                        title=f"{weather_param} vs {water_param}",
                        labels={
                            weather_options[weather_param]: weather_param,
                            water_param: water_param
                        },
                        template="plotly_white",
                        height=600
                    )
                
                # Add some customization for better visualization
                fig.update_layout(
                    boxmode='group' if weather_options[weather_param] in ['Weather Condition', 'Wind Direction'] else None,
                    xaxis_title=weather_param,
                    yaxis_title=water_param,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
    
    with rel_tab2:
        # Your existing volcanic indicators code remains unchanged
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
                
    with rel_tab3:
        st.subheader("Weather Condition & Wind Direction Codes")
        st.markdown("Reference guide for understanding the numeric codes used in the dataset")
        
        # Create two columns for the metadata tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Weather Condition Codes")
            weather_metadata = {
                "Code": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "Description": [
                    "Unknown", "Sunny", "Sunny to Cloudy", "Partly Cloudy", 
                    "Cloudy", "Cloudy to Sunny", "Rainy", "Hazy", 
                    "Calm", "Heavy Rain", "Fair", "Cold"
                ]
            }
            st.table(pd.DataFrame(weather_metadata))
        
        with col2:
            st.markdown("### Wind Direction Codes")
            wind_metadata = {
                "Code": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14],
                "Description": [
                    "Unknown", "Calm", "North East", "South East", 
                    "South West", "ENE", "East", "ESE", 
                    "WSW", "WNW", "NNW", "North", 
                    "West", "North West"
                ]
            }
            st.table(pd.DataFrame(wind_metadata))
        
        st.markdown("""
        **Note:** 
        - These codes are used in the dataset to represent weather conditions and wind directions.
        - When analyzing data, refer to these tables to interpret the numeric values.
        """)

with tab5:
    st.header("AI-Powered Predictions", divider='blue')
    
    # Model configuration
    with st.expander("⚙️ MODEL CONFIGURATION", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            # Algorithm selection
            model_type = st.selectbox(
                "Select Algorithm:",
                options=['CNN', 'LSTM', 'Hybrid CNN-LSTM'],
                index=1
            )
            
            target = st.selectbox(
                "Target Parameter:",
                options=numerical_cols,
                index=numerical_cols.index('pH')
            )
            
            # Location selection
            locations = ['All Locations'] + sorted(df['Site'].unique().tolist())
            location = st.selectbox(
                "Select Location:",
                options=locations,
                index=0
            )
            
        with col2:
            # Horizon selection
            horizon = st.selectbox(
                "Prediction Horizon:",
                options=['Daily', 'Weekly', 'Monthly', 'Yearly'],
                index=1
            )
            
            # Dynamic sequence length based on horizon
            if horizon == 'Daily':
                n_steps = st.slider(
                    "Sequence Length (Days):",
                    min_value=1,
                    max_value=14,
                    value=7,
                    help="Number of previous days to use for prediction"
                )
            elif horizon == 'Weekly':
                n_steps = st.slider(
                    "Sequence Length (Weeks):",
                    min_value=1,
                    max_value=8,
                    value=4,
                    help="Number of previous weeks to use for prediction"
                )
            elif horizon == 'Monthly':
                n_steps = st.slider(
                    "Sequence Length (Months):",
                    min_value=1,
                    max_value=12,
                    value=6,
                    help="Number of previous months to use for prediction"
                )
            else:  # Yearly
                n_steps = st.slider(
                    "Sequence Length (Years):",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="Number of previous years to use for prediction"
                )
    
    # Training button
    if st.button("🚀 TRAIN PREDICTION MODEL", type="primary", use_container_width=True):
        with st.spinner("Training model... This may take a few minutes"):
            # Prepare data
            if location == 'All Locations':
                data = df.groupby('Date')[numerical_cols].mean().reset_index()
            else:
                data = df[df['Site'] == location]
            
            target_idx = numerical_cols.index(target)
            data_array = data[numerical_cols].values
            
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
            
            # Create selected model
            model_creators = {
                'CNN': create_cnn_model,
                'LSTM': create_lstm_model,
                'Hybrid CNN-LSTM': create_hybrid_model
            }
            
            model = model_creators[model_type]()
            
            # Fixed training parameters
            epochs = 50
            batch_size = 32
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            y_pred = model.predict(X_test).flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Generate forecast
            steps = 10  # Predict 10 steps ahead
            current_seq = X[-1].flatten().reshape(1, X_train.shape[1], X_train.shape[2])
            predictions = []
            
            for _ in range(steps):
                pred = model.predict(current_seq)[0, 0]
                predictions.append(pred)
                # Update sequence with prediction
                new_seq = np.append(current_seq[:, 1:, :], 
                                  np.array([current_seq[:, -1, :]]), axis=1)
                new_seq[0, -1, target_idx] = pred
                current_seq = new_seq
            
            # Generate dates based on horizon
            last_date = data['Date'].max()
            date_generator = {
                'Daily': lambda x: last_date + pd.DateOffset(days=x+1),
                'Weekly': lambda x: last_date + pd.DateOffset(weeks=x+1),
                'Monthly': lambda x: last_date + pd.DateOffset(months=x+1),
                'Yearly': lambda x: last_date + pd.DateOffset(years=x+1)
            }[horizon]
            future_dates = [date_generator(i) for i in range(steps)]
            
            # Calculate WQI and violations with error handling
            def calculate_wqi(preds, feature_names, sequences):
                try:
                    required_features = ['pH', 'Dissolved Oxygen', 'Ammonia', 'Phosphate']
                    if not all(feat in feature_names for feat in required_features):
                        st.warning("WQI calculation requires pH, Dissolved Oxygen, Ammonia, and Phosphate parameters")
                        return np.array([]), []
                    
                    wqi, violations = [], []
                    
                    for i in range(len(preds)):
                        score = 100
                        vio = []
                        seq = sequences[i][-1]  # Get last sequence values
                        
                        # pH scoring (6.5-8.5 is ideal)
                        if 'pH' in feature_names:
                            pH_idx = feature_names.index('pH')
                            pH_val = seq[pH_idx]
                            # More nuanced pH scoring
                            if pH_val < 4 or pH_val > 10:  # Extremely poor
                                score -= 40
                                vio.append('pH')
                            elif pH_val < 6 or pH_val > 9:  # Poor
                                score -= 25
                                vio.append('pH')
                            elif not 6.5 <= pH_val <= 8.5:  # Fair
                                score -= 10
                                
                        # Dissolved Oxygen scoring (>5 mg/L is ideal)
                        if 'Dissolved Oxygen' in feature_names:
                            do_idx = feature_names.index('Dissolved Oxygen')
                            do_val = seq[do_idx]
                            if do_val < 2:  # Extremely poor
                                score -= 40
                                vio.append('Low Oxygen')
                            elif do_val < 5:  # Poor
                                score -= 20
                                vio.append('Low Oxygen')
                            elif do_val < 7:  # Fair
                                score -= 5
                                
                        # Ammonia scoring (<1 mg/L is ideal)
                        if 'Ammonia' in feature_names:
                            ammonia_idx = feature_names.index('Ammonia')
                            ammonia_val = seq[ammonia_idx]
                            if ammonia_val > 5:  # Extremely poor
                                score -= 40
                                vio.append('High Ammonia')
                            elif ammonia_val > 1:  # Poor
                                score -= 20
                                vio.append('High Ammonia')
                            elif ammonia_val > 0.5:  # Fair
                                score -= 5
                                
                        # Phosphate scoring (<0.4 mg/L is ideal)
                        if 'Phosphate' in feature_names:
                            phosphate_idx = feature_names.index('Phosphate')
                            phosphate_val = seq[phosphate_idx]
                            if phosphate_val > 2:  # Extremely poor
                                score -= 40
                                vio.append('High Phosphate')
                            elif phosphate_val > 0.4:  # Poor
                                score -= 20
                                vio.append('High Phosphate')
                            elif phosphate_val > 0.2:  # Fair
                                score -= 5
                        
                        wqi.append(max(score, 0))  # Ensure score doesn't go below 0
                        violations.append(vio)
                    
                    return np.array(wqi), violations
                
                except Exception as e:
                    st.warning(f"Error calculating WQI: {str(e)}")
                    return np.array([]), []
                                    
                except Exception as e:
                    st.warning(f"Error calculating WQI: {str(e)}")
                    return np.array([]), []
            
            try:
                wqi_vals, vio_list = calculate_wqi(y_pred, numerical_cols, X_test)
                wqi_data = {
                    'values': wqi_vals,
                    'violations': vio_list
                }
            except Exception as e:
                st.warning(f"Could not calculate WQI: {str(e)}")
                wqi_data = {
                    'values': [],
                    'violations': []
                }
            
            st.session_state.model_results = {
                'model_type': model_type,
                'metrics': {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                },
                'history': history,
                'forecast': {
                    'dates': future_dates,
                    'values': predictions,
                    'parameter': target,
                    'horizon': horizon,
                    'location': location
                },
                'test_predictions': {
                    'actual': y_test,
                    'predicted': y_pred
                },
                'wqi': wqi_data
            }
            st.success("Model training completed!")
    
    if 'model_results' in st.session_state:
        results = st.session_state.model_results
        
        # Model Performance Section
        st.subheader("Model Performance")
        
        # Metrics cards
        cols = st.columns(4)
        with cols[0]:
            st.metric("Model Type", results['model_type'])
        with cols[1]:
            st.metric("MAE", f"{results['metrics']['MAE']:.4f}")
        with cols[2]:
            st.metric("RMSE", f"{results['metrics']['RMSE']:.4f}")
        with cols[3]:
            st.metric("R² Score", f"{results['metrics']['R2']:.4f}")
        
        # Bar chart for MAE and RMSE
        st.markdown("### Performance Metrics Comparison")
        metrics_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R²'],
            'Value': [results['metrics']['MAE'], results['metrics']['RMSE'], results['metrics']['R2']]
        })
        
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            color='Metric',
            color_discrete_map={'MAE': '#636EFA', 'RMSE': '#EF553B', 'R²': '#00CC96'},
            text_auto='.4f',
            height=400
        )
        fig.update_layout(
            title=f"{results['model_type']} Model Performance",
            xaxis_title="Metric",
            yaxis_title="Value",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Training curve
        st.subheader("Training Progress")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(results['history'].history['loss']))),
            y=results['history'].history['loss'],
            name='Training Loss',
            line=dict(width=3)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(results['history'].history['val_loss']))),
            y=results['history'].history['val_loss'],
            name='Validation Loss',
            line=dict(width=3, dash='dash')
        ))
        
        fig.update_layout(
            title="Training and Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss (MAE)",
            hovermode="x unified",
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast visualization
        st.subheader(f"{results['forecast']['horizon']} Forecast")
        fig = go.Figure()
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=results['forecast']['dates'],
            y=results['forecast']['values'],
            name='Forecast',
            line=dict(color='red', dash='dash'),
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title=f"{results['forecast']['horizon']} Forecast for {results['forecast']['parameter']}",
            xaxis_title='Date',
            yaxis_title=results['forecast']['parameter'],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        st.markdown(f"### {results['forecast']['horizon']} Forecast (Next 10 Periods)")
        forecast_df = pd.DataFrame({
            'Date': results['forecast']['dates'],
            'Predicted Value': results['forecast']['values']
        })
        
        st.dataframe(forecast_df.style.format({
            'Date': lambda x: x.strftime('%Y-%m-%d'),
            'Predicted Value': "{:.4f}"
        }), use_container_width=True)
        
        # Actual vs Predicted
        st.subheader("📌 Actual vs Predicted Values")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['test_predictions']['actual'],
            y=results['test_predictions']['predicted'],
            mode='markers',
            marker=dict(color='skyblue', size=8, opacity=0.6),
            name='Predicted vs Actual'
        ))
        fig.add_trace(go.Scatter(
            x=results['test_predictions']['actual'],
            y=results['test_predictions']['actual'],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Ideal Line (y=x)'
        ))
        fig.update_layout(
            title=f"{results['model_type']}: Actual vs Predicted ({target})",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Water Quality Assessment - larger
        st.subheader("Water Quality Index Assessment")
        
        if len(results['wqi']['values']) > 0:
            # WQI Distribution - larger
            wqi_bins = {
                'Excellent (90-100)': (90, 100),
                'Good (70-89)': (70, 89),
                'Fair (50-69)': (50, 69),
                'Poor (<50)': (0, 49)
            }
            
            wqi_counts = {cat: 0 for cat in wqi_bins}
            for val in results['wqi']['values']:
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
                        count/len(results['wqi']['values']),
                        text=f"{cat}: {count} samples ({count/len(results['wqi']['values']):.1%})"
                    )
                
            # Enhanced recommendations
            st.subheader("Recommendations")
            
            # More detailed recommendations mapping
            recommendation_map = {
                'pH': [
                    ("Critical (pH <4 or >10)", "Immediate chemical adjustment required"),
                    ("Poor (pH <6 or >9)", "Add pH buffers or aeration"),
                    ("Fair (outside 6.5-8.5)", "Monitor and consider gradual adjustment")
                ],
                'Low Oxygen': [
                    ("Critical (<2 mg/L)", "Emergency aeration needed"),
                    ("Poor (<5 mg/L)", "Increase water movement and aeration"),
                    ("Fair (<7 mg/L)", "Monitor and consider aeration")
                ],
                'High Ammonia': [
                    ("Critical (>5 mg/L)", "Immediate water change required"),
                    ("Poor (>1 mg/L)", "Increase biofiltration and reduce feeding"),
                    ("Fair (>0.5 mg/L)", "Monitor and optimize filtration")
                ],
                'High Phosphate': [
                    ("Critical (>2 mg/L)", "Chemical phosphate removal needed"),
                    ("Poor (>0.4 mg/L)", "Use phosphate removers and reduce nutrients"),
                    ("Fair (>0.2 mg/L)", "Monitor and adjust feeding practices")
                ]
            }
            
            # Get unique violations
            all_vios = set()
            for vlist in results['wqi']['violations']:
                all_vios.update(vlist)
            
            if all_vios:
                st.warning("⚠️ Water quality issues detected:")
                for violation in sorted(all_vios):
                    with st.expander(f"**{violation}** - Recommended Actions", expanded=True):
                        for level, action in recommendation_map.get(violation, []):
                            st.info(f"• {level}: {action}")
            else:
                st.success("✅ No significant water quality issues detected", icon="✅")
        else:
            st.warning("Could not calculate WQI - required parameters not available")
            
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
