import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, IsolatedForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Wind Turbine Failure Protection System",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff0000;
    }
    .alert-warning {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffa500;
    }
    .alert-normal {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'anomaly_model' not in st.session_state:
    st.session_state.anomaly_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Generate synthetic wind turbine data
def generate_turbine_data(n_samples=1000, failure_rate=0.15):
    np.random.seed(42)

    # Normal operating conditions
    normal_samples = int(n_samples * (1 - failure_rate))
    failure_samples = n_samples - normal_samples

    # Normal data
    wind_speed_normal = np.random.normal(12, 3, normal_samples)
    rotor_speed_normal = wind_speed_normal * 8 + np.random.normal(0, 5, normal_samples)
    power_output_normal = (wind_speed_normal ** 3) * 0.5 + np.random.normal(0, 50, normal_samples)
    temperature_normal = np.random.normal(65, 8, normal_samples)
    vibration_normal = np.random.normal(2, 0.5, normal_samples)
    bearing_temp_normal = np.random.normal(70, 10, normal_samples)
    oil_pressure_normal = np.random.normal(50, 5, normal_samples)

    # Failure data (anomalous patterns)
    wind_speed_failure = np.random.normal(12, 3, failure_samples)
    rotor_speed_failure = wind_speed_failure * 8 + np.random.normal(20, 15, failure_samples)  # Abnormal rotor speed
    power_output_failure = (wind_speed_failure ** 3) * 0.3 + np.random.normal(-100, 80, failure_samples)  # Reduced power
    temperature_failure = np.random.normal(85, 15, failure_samples)  # Higher temperature
    vibration_failure = np.random.normal(5, 2, failure_samples)  # High vibration
    bearing_temp_failure = np.random.normal(95, 15, failure_samples)  # High bearing temp
    oil_pressure_failure = np.random.normal(35, 10, failure_samples)  # Low oil pressure

    # Combine data
    wind_speed = np.concatenate([wind_speed_normal, wind_speed_failure])
    rotor_speed = np.concatenate([rotor_speed_normal, rotor_speed_failure])
    power_output = np.concatenate([power_output_normal, power_output_failure])
    temperature = np.concatenate([temperature_normal, temperature_failure])
    vibration = np.concatenate([vibration_normal, vibration_failure])
    bearing_temp = np.concatenate([bearing_temp_normal, bearing_temp_failure])
    oil_pressure = np.concatenate([oil_pressure_normal, oil_pressure_failure])
    failure = np.concatenate([np.zeros(normal_samples), np.ones(failure_samples)])

    # Create timestamps
    start_time = datetime.now() - timedelta(hours=n_samples)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]

    df = pd.DataFrame({
        'timestamp': timestamps,
        'wind_speed': wind_speed,
        'rotor_speed': rotor_speed,
        'power_output': power_output,
        'temperature': temperature,
        'vibration': vibration,
        'bearing_temp': bearing_temp,
        'oil_pressure': oil_pressure,
        'failure': failure
    })

    return df

# Train ML models
def train_models(data):
    X = data[['wind_speed', 'rotor_speed', 'power_output', 'temperature', 'vibration', 'bearing_temp', 'oil_pressure']]
    y = data['failure']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)

    # Isolated Forest for anomaly detection
    anomaly_model = IsolatedForest(contamination=0.15, random_state=42)
    anomaly_model.fit(X_train_scaled)

    # Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return rf_model, anomaly_model, scaler, accuracy, X_test_scaled, y_test, y_pred

# Real-time simulation
def simulate_realtime_reading():
    failure_prob = np.random.random()

    if failure_prob < 0.85:  # Normal operation
        wind_speed = np.random.normal(12, 3)
        rotor_speed = wind_speed * 8 + np.random.normal(0, 5)
        power_output = (wind_speed ** 3) * 0.5 + np.random.normal(0, 50)
        temperature = np.random.normal(65, 8)
        vibration = np.random.normal(2, 0.5)
        bearing_temp = np.random.normal(70, 10)
        oil_pressure = np.random.normal(50, 5)
    else:  # Failure conditions
        wind_speed = np.random.normal(12, 3)
        rotor_speed = wind_speed * 8 + np.random.normal(20, 15)
        power_output = (wind_speed ** 3) * 0.3 + np.random.normal(-100, 80)
        temperature = np.random.normal(85, 15)
        vibration = np.random.normal(5, 2)
        bearing_temp = np.random.normal(95, 15)
        oil_pressure = np.random.normal(35, 10)

    return {
        'timestamp': datetime.now(),
        'wind_speed': max(0, wind_speed),
        'rotor_speed': max(0, rotor_speed),
        'power_output': max(0, power_output),
        'temperature': max(0, temperature),
        'vibration': max(0, vibration),
        'bearing_temp': max(0, bearing_temp),
        'oil_pressure': max(0, oil_pressure)
    }

# Predict failure
def predict_failure(reading, model, scaler):
    features = np.array([[
        reading['wind_speed'],
        reading['rotor_speed'],
        reading['power_output'],
        reading['temperature'],
        reading['vibration'],
        reading['bearing_temp'],
        reading['oil_pressure']
    ]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    return prediction, probability

# Main app
st.markdown('<h1 class="main-header">🌬️ Wind Turbine Failure Protection System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")

    st.subheader("Training Data")
    n_samples = st.slider("Training samples", 500, 5000, 1000, 100)
    failure_rate = st.slider("Failure rate", 0.05, 0.30, 0.15, 0.05)

    if st.button("🔄 Generate & Train Models", type="primary"):
        with st.spinner("Generating data and training models..."):
            st.session_state.historical_data = generate_turbine_data(n_samples, failure_rate)
            rf_model, anomaly_model, scaler, accuracy, X_test, y_test, y_pred = train_models(st.session_state.historical_data)
            st.session_state.rf_model = rf_model
            st.session_state.anomaly_model = anomaly_model
            st.session_state.scaler = scaler
            st.session_state.model_accuracy = accuracy
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.model_trained = True
            st.success(f"✅ Models trained! Accuracy: {accuracy:.2%}")

    st.divider()

    if st.session_state.model_trained:
        st.subheader("Real-Time Simulation")
        simulation_speed = st.select_slider("Update interval (seconds)", options=[0.5, 1, 2, 3, 5], value=1)

        if st.button("▶️ Start Simulation", type="primary"):
            st.session_state.simulation_running = True

        if st.button("⏸️ Stop Simulation"):
            st.session_state.simulation_running = False

    st.divider()
    st.markdown("### 📊 System Status")
    if st.session_state.model_trained:
        st.success("✅ Models Active")
    else:
        st.warning("⚠️ Models Not Trained")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-Time Monitor", "🧠 Model Performance", "📈 Historical Analysis", "🔍 Anomaly Detection"])

with tab1:
    if not st.session_state.model_trained:
        st.warning("⚠️ Please generate and train models first using the sidebar.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        # Placeholders for real-time metrics
        metric_wind = col1.empty()
        metric_power = col2.empty()
        metric_temp = col3.empty()
        metric_status = col4.empty()

        # Alert placeholder
        alert_placeholder = st.empty()

        # Charts placeholders
        chart_col1, chart_col2 = st.columns(2)
        chart1_placeholder = chart_col1.empty()
        chart2_placeholder = chart_col2.empty()

        chart_col3, chart_col4 = st.columns(2)
        chart3_placeholder = chart_col3.empty()
        chart4_placeholder = chart_col4.empty()

        # Initialize data buffers for charts
        if 'buffer_data' not in st.session_state:
            st.session_state.buffer_data = {
                'timestamp': [],
                'wind_speed': [],
                'power_output': [],
                'temperature': [],
                'vibration': [],
                'rotor_speed': [],
                'bearing_temp': [],
                'oil_pressure': [],
                'failure_prob': []
            }

        # Simulation loop
        if st.session_state.simulation_running:
            for _ in range(50):  # Limit iterations
                if not st.session_state.simulation_running:
                    break

                # Get new reading
                reading = simulate_realtime_reading()
                prediction, probability = predict_failure(reading, st.session_state.rf_model, st.session_state.scaler)

                # Update buffers
                st.session_state.buffer_data['timestamp'].append(reading['timestamp'])
                st.session_state.buffer_data['wind_speed'].append(reading['wind_speed'])
                st.session_state.buffer_data['power_output'].append(reading['power_output'])
                st.session_state.buffer_data['temperature'].append(reading['temperature'])
                st.session_state.buffer_data['vibration'].append(reading['vibration'])
                st.session_state.buffer_data['rotor_speed'].append(reading['rotor_speed'])
                st.session_state.buffer_data['bearing_temp'].append(reading['bearing_temp'])
                st.session_state.buffer_data['oil_pressure'].append(reading['oil_pressure'])
                st.session_state.buffer_data['failure_prob'].append(probability[1])

                # Keep only last 50 readings
                if len(st.session_state.buffer_data['timestamp']) > 50:
                    for key in st.session_state.buffer_data:
                        st.session_state.buffer_data[key] = st.session_state.buffer_data[key][-50:]

                # Update metrics
                metric_wind.metric("Wind Speed", f"{reading['wind_speed']:.1f} m/s")
                metric_power.metric("Power Output", f"{reading['power_output']:.0f} kW")
                metric_temp.metric("Temperature", f"{reading['temperature']:.1f} °C")

                # Status and alert
                failure_prob_pct = probability[1] * 100

                if prediction == 1 or failure_prob_pct > 70:
                    metric_status.metric("Status", "🔴 CRITICAL", delta="Failure Detected")
                    alert_placeholder.markdown(f"""
                    <div class="alert-critical">
                        <h3>🚨 CRITICAL ALERT: Failure Detected!</h3>
                        <p><strong>Failure Probability: {failure_prob_pct:.1f}%</strong></p>
                        <p>Immediate maintenance required. Initiating safety shutdown procedures.</p>
                        <ul>
                            <li>Vibration: {reading['vibration']:.2f} mm/s</li>
                            <li>Bearing Temp: {reading['bearing_temp']:.1f} °C</li>
                            <li>Oil Pressure: {reading['oil_pressure']:.1f} bar</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif failure_prob_pct > 40:
                    metric_status.metric("Status", "🟡 WARNING", delta="Elevated Risk")
                    alert_placeholder.markdown(f"""
                    <div class="alert-warning">
                        <h3>⚠️ WARNING: Elevated Failure Risk</h3>
                        <p><strong>Failure Probability: {failure_prob_pct:.1f}%</strong></p>
                        <p>Schedule inspection soon. Monitor closely.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    metric_status.metric("Status", "🟢 NORMAL", delta="Operating Normally")
                    alert_placeholder.markdown(f"""
                    <div class="alert-normal">
                        <h3>✅ System Operating Normally</h3>
                        <p><strong>Failure Probability: {failure_prob_pct:.1f}%</strong></p>
                        <p>All parameters within normal range.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Update charts
                if len(st.session_state.buffer_data['timestamp']) > 1:
                    # Wind Speed & Power
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=st.session_state.buffer_data['timestamp'],
                                             y=st.session_state.buffer_data['wind_speed'],
                                             name='Wind Speed', line=dict(color='blue')))
                    fig1.update_layout(title='Wind Speed Over Time', xaxis_title='Time', yaxis_title='m/s', height=250)
                    chart1_placeholder.plotly_chart(fig1, use_container_width=True)

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=st.session_state.buffer_data['timestamp'],
                                             y=st.session_state.buffer_data['power_output'],
                                             name='Power Output', line=dict(color='green'), fill='tozeroy'))
                    fig2.update_layout(title='Power Output Over Time', xaxis_title='Time', yaxis_title='kW', height=250)
                    chart2_placeholder.plotly_chart(fig2, use_container_width=True)

                    # Temperature & Vibration
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(x=st.session_state.buffer_data['timestamp'],
                                             y=st.session_state.buffer_data['temperature'],
                                             name='Temperature', line=dict(color='red')))
                    fig3.add_trace(go.Scatter(x=st.session_state.buffer_data['timestamp'],
                                             y=st.session_state.buffer_data['bearing_temp'],
                                             name='Bearing Temp', line=dict(color='orange')))
                    fig3.update_layout(title='Temperature Monitoring', xaxis_title='Time', yaxis_title='°C', height=250)
                    chart3_placeholder.plotly_chart(fig3, use_container_width=True)

                    # Failure Probability
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(x=st.session_state.buffer_data['timestamp'],
                                             y=st.session_state.buffer_data['failure_prob'],
                                             name='Failure Probability',
                                             line=dict(color='purple', width=3),
                                             fill='tozeroy'))
                    fig4.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
                    fig4.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
                    fig4.update_layout(title='Failure Probability', xaxis_title='Time', yaxis_title='Probability', height=250)
                    chart4_placeholder.plotly_chart(fig4, use_container_width=True)

                time.sleep(simulation_speed)

with tab2:
    if not st.session_state.model_trained:
        st.warning("⚠️ Please generate and train models first using the sidebar.")
    else:
        st.header("🧠 Machine Learning Model Performance")

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Accuracy", f"{st.session_state.model_accuracy:.2%}")
        col2.metric("Model Type", "Random Forest")
        col3.metric("Features Used", "7")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Normal', 'Failure'],
            y=['Normal', 'Failure'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        fig_cm.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual', height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(st.session_state.y_test, st.session_state.y_pred,
                                      target_names=['Normal', 'Failure'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

        # Feature Importance
        st.subheader("Feature Importance")
        feature_names = ['Wind Speed', 'Rotor Speed', 'Power Output', 'Temperature', 'Vibration', 'Bearing Temp', 'Oil Pressure']
        importances = st.session_state.rf_model.feature_importances_
        feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)

        fig_importance = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                               title='Feature Importance for Failure Prediction')
        st.plotly_chart(fig_importance, use_container_width=True)

with tab3:
    if st.session_state.historical_data is None:
        st.warning("⚠️ Please generate training data first using the sidebar.")
    else:
        st.header("📈 Historical Data Analysis")

        data = st.session_state.historical_data

        # Summary statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Failure Distribution")
            failure_counts = data['failure'].value_counts()
            fig_pie = px.pie(values=failure_counts.values, names=['Normal', 'Failure'],
                           title='Normal vs Failure Cases',
                           color_discrete_sequence=['#28a745', '#dc3545'])
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Power Output Distribution")
            fig_hist = px.histogram(data, x='power_output', color='failure',
                                   title='Power Output by Status',
                                   labels={'failure': 'Status'},
                                   color_discrete_map={0: '#28a745', 1: '#dc3545'})
            st.plotly_chart(fig_hist, use_container_width=True)

        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        corr_data = data[['wind_speed', 'rotor_speed', 'power_output', 'temperature', 'vibration', 'bearing_temp', 'oil_pressure']].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_data.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig_corr.update_layout(title='Feature Correlation Heatmap', height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Time series comparison
        st.subheader("Parameter Comparison: Normal vs Failure")
        sample_normal = data[data['failure'] == 0].sample(min(100, len(data[data['failure'] == 0])))
        sample_failure = data[data['failure'] == 1].sample(min(100, len(data[data['failure'] == 1])))

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=sample_normal['wind_speed'], y=sample_normal['vibration'],
                                        mode='markers', name='Normal', marker=dict(color='green', size=8)))
        fig_compare.add_trace(go.Scatter(x=sample_failure['wind_speed'], y=sample_failure['vibration'],
                                        mode='markers', name='Failure', marker=dict(color='red', size=8)))
        fig_compare.update_layout(title='Wind Speed vs Vibration', xaxis_title='Wind Speed (m/s)',
                                 yaxis_title='Vibration (mm/s)', height=400)
        st.plotly_chart(fig_compare, use_container_width=True)

with tab4:
    if not st.session_state.model_trained:
        st.warning("⚠️ Please generate and train models first using the sidebar.")
    else:
        st.header("🔍 Anomaly Detection Analysis")

        st.markdown("""
        The Isolation Forest algorithm identifies anomalous patterns in turbine behavior that deviate
        from normal operating conditions, even when they don't result in immediate failure.
        """)

        # Anomaly detection on test data
        anomaly_predictions = st.session_state.anomaly_model.predict(st.session_state.X_test)
        anomaly_scores = st.session_state.anomaly_model.score_samples(st.session_state.X_test)

        # Convert to binary (1 = normal, -1 = anomaly)
        n_anomalies = np.sum(anomaly_predictions == -1)
        n_normal = np.sum(anomaly_predictions == 1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(anomaly_predictions))
        col2.metric("Normal Samples", n_normal)
        col3.metric("Anomalies Detected", n_anomalies)

        # Anomaly score distribution
        st.subheader("Anomaly Score Distribution")
        fig_scores = px.histogram(x=anomaly_scores, nbins=50,
                                 title='Distribution of Anomaly Scores',
                                 labels={'x': 'Anomaly Score', 'y': 'Count'})
        fig_scores.add_vline(x=np.percentile(anomaly_scores, 15), line_dash="dash",
                           line_color="red", annotation_text="Anomaly Threshold")
        st.plotly_chart(fig_scores, use_container_width=True)

        # 3D visualization of anomalies
        st.subheader("3D Anomaly Visualization")
        test_df = pd.DataFrame(st.session_state.X_test,
                              columns=['wind_speed', 'rotor_speed', 'power_output', 'temperature',
                                      'vibration', 'bearing_temp', 'oil_pressure'])
        test_df['anomaly'] = anomaly_predictions
        test_df['anomaly_label'] = test_df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

        fig_3d = px.scatter_3d(test_df.sample(min(500, len(test_df))),
                              x='wind_speed', y='vibration', z='bearing_temp',
                              color='anomaly_label',
                              color_discrete_map={'Normal': 'green', 'Anomaly': 'red'},
                              title='3D Anomaly Detection: Wind Speed vs Vibration vs Bearing Temperature')
        st.plotly_chart(fig_3d, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Wind Turbine Failure Protection System | Powered by Machine Learning | Real-time Monitoring & Predictive Maintenance</p>
</div>
""", unsafe_allow_html=True)
