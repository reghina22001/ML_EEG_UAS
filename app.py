"""
===============================================================================
STREAMLIT APP - KLASIFIKASI SINYAL EEG
Deep Learning Model Deployment
===============================================================================
Aplikasi web untuk demo model deep learning klasifikasi sinyal EEG
Model: CNN1D, LSTM, CNN-LSTM Hybrid, EEGNet
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
import io
import os
from pathlib import Path

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="EEG Signal Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konstanta Preprocessing (HARUS SAMA dengan training)
SAMPLING_RATE = 200  # Hz
LOWCUT = 0.5
HIGHCUT = 45.0
EPOCH_LENGTH = 4  # detik
MODEL_DIR = "models"

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNGSI PREPROCESSING ====================

@st.cache_data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Band-pass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data


@st.cache_data
def notch_filter(data, fs, freq=50.0, quality=30):
    """Notch filter untuk powerline noise"""
    b, a = signal.iirnotch(freq, quality, fs)
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data


def preprocess_eeg_data(data):
    """
    Preprocessing pipeline untuk data EEG
    
    Args:
        data: numpy array shape (n_samples, n_channels)
    
    Returns:
        epochs: numpy array shape (n_epochs, samples_per_epoch, n_channels)
    """
    with st.spinner("🔄 Applying filters..."):
        # 1. Band-pass filter
        filtered_data = butter_bandpass_filter(data, LOWCUT, HIGHCUT, SAMPLING_RATE)
        
        # 2. Notch filter
        filtered_data = notch_filter(filtered_data, SAMPLING_RATE, freq=50.0)
    
    with st.spinner("✂️ Segmenting signal..."):
        # 3. Epoching
        samples_per_epoch = int(SAMPLING_RATE * EPOCH_LENGTH)
        n_samples = filtered_data.shape[0]
        n_epochs = n_samples // samples_per_epoch
        
        # Trim data
        trimmed_data = filtered_data[:n_epochs * samples_per_epoch]
        
        # Reshape ke epochs
        epochs = trimmed_data.reshape(n_epochs, samples_per_epoch, -1)
    
    with st.spinner("📊 Normalizing..."):
        # 4. Baseline correction
        epochs = epochs - np.mean(epochs, axis=1, keepdims=True)
        
        # 5. Z-score normalization
        mean = np.mean(epochs, axis=(0, 1), keepdims=True)
        std = np.std(epochs, axis=(0, 1), keepdims=True)
        normalized_epochs = (epochs - mean) / (std + 1e-8)
    
    return normalized_epochs


# ==================== FUNGSI LOAD MODEL (FIXED) ====================

def load_model_robust(model_path):
    """
    Load model dengan multiple fallback strategies
    Mengatasi error: batch_shape, dtype, dll
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    # Strategy 1: Load without compile (PALING AMAN)
    try:
        with tf.keras.utils.custom_object_scope({}):
            model = keras.models.load_model(model_path, compile=False)
        
        # Re-compile dengan config standard
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model, "success"
    
    except Exception as e1:
        # Strategy 2: Normal load (fallback)
        try:
            model = keras.models.load_model(model_path)
            return model, "success"
        
        except Exception as e2:
            # Strategy 3: Load with custom objects
            try:
                custom_objects = {
                    'InputLayer': keras.layers.InputLayer,
                }
                model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                return model, "success"
            
            except Exception as e3:
                # All strategies failed
                error_msg = f"Load failed: {str(e3)[:150]}"
                return None, error_msg


@st.cache_resource
def load_model(model_name):
    """Load trained model with robust error handling"""
    model_path = os.path.join(MODEL_DIR, f"{model_name}_final.h5")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model tidak ditemukan: {model_path}")
        st.info("""
        **Cara fix:**
        1. Pastikan file model ada di folder `models/`
        2. Format nama file: `{model_name}_final.h5`
        3. Contoh: `CNN1D_final.h5`, `LSTM_final.h5`
        """)
        return None
    
    # Try robust loading
    with st.spinner(f"Loading {model_name}..."):
        model, status = load_model_robust(model_path)
    
    if model is None:
        st.error(f"❌ Error loading {model_name}")
        st.error(f"Detail: {status}")
        
        with st.expander("🔧 Solusi untuk Error Ini"):
            st.markdown("""
            **Error ini terjadi karena incompatibility TensorFlow versions.**
            
            ### Solusi 1: Re-save Model (di Colab)
            ```python
            from tensorflow import keras
            
            # Load model lama
            model = keras.models.load_model('model_lama.h5', compile=False)
            
            # Re-compile
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save ulang
            model.save('model_baru.h5')
            ```
            
            ### Solusi 2: Gunakan SavedModel format
            ```python
            # Di Colab, save dengan format SavedModel
            model.save('model_folder/')  # Tanpa .h5
            
            # Di Streamlit, load dari folder
            model = keras.models.load_model('model_folder/')
            ```
            
            ### Solusi 3: Export weights only
            ```python
            # Di Colab
            model.save_weights('model_weights.h5')
            
            # Di Streamlit, rebuild model lalu load weights
            ```
            """)
        return None
    
    return model


def get_available_models():
    """Get list of available models"""
    if not os.path.exists(MODEL_DIR):
        return []
    
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_final.h5')]
    model_names = [f.replace('_final.h5', '') for f in model_files]
    return model_names


# ==================== FUNGSI VISUALISASI ====================

def plot_eeg_signal(data, title="EEG Signal", max_channels=5):
    """Plot time series EEG signal"""
    n_channels = min(max_channels, data.shape[1])
    time = np.arange(data.shape[0]) / SAMPLING_RATE
    
    fig = go.Figure()
    
    for i in range(n_channels):
        fig.add_trace(go.Scatter(
            x=time,
            y=data[:, i] + i * 3,  # Offset untuk visualisasi
            mode='lines',
            name=f'Channel {i+1}',
            line=dict(width=1)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (Normalized + Offset)",
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_power_spectrum(data, channel=0):
    """Plot power spectral density"""
    f, psd = signal.welch(data[:, channel], fs=SAMPLING_RATE, nperseg=256)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=f,
        y=psd,
        mode='lines',
        fill='tozeroy',
        line=dict(color='steelblue', width=2)
    ))
    
    # Add frequency band regions
    bands = {
        'Delta': (0.5, 4, 'rgba(255, 0, 0, 0.1)'),
        'Theta': (4, 8, 'rgba(255, 165, 0, 0.1)'),
        'Alpha': (8, 13, 'rgba(255, 255, 0, 0.1)'),
        'Beta': (13, 30, 'rgba(0, 255, 0, 0.1)'),
        'Gamma': (30, 45, 'rgba(0, 0, 255, 0.1)')
    }
    
    for band_name, (low, high, color) in bands.items():
        fig.add_vrect(
            x0=low, x1=high,
            fillcolor=color,
            layer="below",
            line_width=0,
            annotation_text=band_name,
            annotation_position="top left"
        )
    
    fig.update_layout(
        title=f"Power Spectral Density - Channel {channel+1}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (V²/Hz)",
        yaxis_type="log",
        height=400,
        template='plotly_white',
        xaxis_range=[0, 50]
    )
    
    return fig


def plot_prediction_confidence(predictions):
    """Plot prediction confidence"""
    classes = ['Training', 'Online']
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=predictions[0],
            marker_color=['#3498db', '#e74c3c'],
            text=[f'{p:.2%}' for p in predictions[0]],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis_range=[0, 1.1],
        height=400,
        template='plotly_white'
    )
    
    return fig


# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown('<div class="main-header">🧠 EEG Signal Classification</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/brain.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📊 Model Demo", "📈 Model Comparison", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        st.info(f"""
        **Preprocessing Settings:**
        - Sampling Rate: {SAMPLING_RATE} Hz
        - Bandpass: {LOWCUT}-{HIGHCUT} Hz
        - Epoch Length: {EPOCH_LENGTH}s
        """)
    
    # Main Content
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Model Demo":
        show_demo_page()
    elif page == "📈 Model Comparison":
        show_comparison_page()
    elif page == "ℹ️ About":
        show_about_page()


# ==================== PAGE: HOME ====================

def show_home_page():
    st.markdown('<div class="sub-header">Welcome to EEG Signal Classification App</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Models Available", len(get_available_models()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Input Channels", "20")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Classes", "2")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Overview
    st.markdown("### 📋 Project Overview")
    st.markdown("""
    This application demonstrates deep learning models for EEG signal classification.
    The models are trained to classify EEG signals into two categories:
    - **Training Data**: Signals from training sessions
    - **Online Data**: Signals from online/testing sessions
    
    **Available Models:**
    1. **CNN1D**: 1D Convolutional Neural Network for temporal feature extraction
    2. **LSTM**: Long Short-Term Memory for temporal dependencies
    3. **CNN-LSTM Hybrid**: Combined architecture for spatial-temporal features
    4. **EEGNet**: State-of-the-art architecture specifically designed for EEG
    """)
    
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. Go to **📊 Model Demo** page
    2. Upload your EEG data (CSV format)
    3. Select a model
    4. Click **Predict** to see results!
    """)


# ==================== PAGE: DEMO ====================

def show_demo_page():
    st.markdown('<div class="sub-header">Model Demo & Prediction</div>', unsafe_allow_html=True)
    
    # Check available models
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No models found! Please place model files in 'models/' directory.")
        st.info("Expected format: `ModelName_final.h5`")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model:",
        available_models,
        help="Choose a trained model for prediction"
    )
    
    # File upload
    st.markdown("### 📁 Upload EEG Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file (samples × channels)",
        type=['csv'],
        help="CSV file with shape (n_samples, n_channels)"
    )
    
    if uploaded_file is not None:
        try:
            # Load data dengan handling berbagai format
            try:
                # Try 1: Load tanpa header (numeric only)
                data = pd.read_csv(uploaded_file, header=None).values
            except:
                # Try 2: Load dengan header, ambil numeric columns
                df = pd.read_csv(uploaded_file)
                # Drop non-numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                data = df[numeric_cols].values
            
            # Validate shape
            if len(data.shape) != 2:
                st.error(f"❌ Data harus 2D array (samples × channels), got shape: {data.shape}")
                st.stop()
            
            if data.shape[1] != 20:
                st.warning(f"⚠️ Expected 20 channels, got {data.shape[1]} channels")
                st.info("Model trained with 20 channels. Results may be inaccurate.")
            
            if data.shape[0] < 800:
                st.error(f"❌ Data terlalu pendek! Minimal 800 samples (4 detik), got {data.shape[0]}")
                st.info(f"Duration: {data.shape[0] / SAMPLING_RATE:.2f} seconds")
                st.stop()
            
            st.success(f"✅ Data loaded: {data.shape[0]} samples × {data.shape[1]} channels")
            
            # Display data info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Samples", f"{data.shape[0]:,}")
                st.metric("Duration", f"{data.shape[0] / SAMPLING_RATE:.2f} seconds")
            
            with col2:
                st.metric("Channels", data.shape[1])
                st.metric("Sampling Rate", f"{SAMPLING_RATE} Hz")
            
            # Data quality check
            with st.expander("📊 Data Quality Check"):
                st.write("**Data Statistics:**")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std', 'Min', 'Max', 'NaN Count'],
                    'Value': [
                        f"{data.mean():.4f}",
                        f"{data.std():.4f}",
                        f"{data.min():.4f}",
                        f"{data.max():.4f}",
                        f"{np.isnan(data).sum()}"
                    ]
                })
                st.table(stats_df)
                
                # Check for issues
                issues = []
                if np.isnan(data).any():
                    issues.append("⚠️ Data contains NaN values")
                if np.isinf(data).any():
                    issues.append("⚠️ Data contains Inf values")
                if data.std() == 0:
                    issues.append("⚠️ Data has zero variance")
                
                if issues:
                    for issue in issues:
                        st.warning(issue)
                else:
                    st.success("✅ Data quality OK")
            
            # Visualize raw signal
            st.markdown("### 📊 Raw Signal Visualization")
            fig_signal = plot_eeg_signal(data[:1000], "Raw EEG Signal (First 5 seconds)")
            st.plotly_chart(fig_signal, use_container_width=True)
            
            # Preprocess button
            if st.button("🔄 Preprocess & Predict", type="primary"):
                with st.spinner("Processing..."):
                    # Preprocess
                    epochs = preprocess_eeg_data(data)
                    
                    st.success(f"✅ Preprocessing complete! Created {epochs.shape[0]} epochs")
                    
                    # Load model
                    model = load_model(selected_model)
                    
                    if model is not None:
                        # Predict
                        predictions = model.predict(epochs, verbose=0)
                        
                        # Get average prediction
                        avg_pred = np.mean(predictions, axis=0)
                        pred_class = np.argmax(avg_pred)
                        confidence = avg_pred[pred_class]
                        
                        # Display results
                        st.markdown("### 🎯 Prediction Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.metric("Predicted Class", "Training" if pred_class == 0 else "Online")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        with col3:
                            st.metric("Epochs Processed", epochs.shape[0])
                        
                        # Plot confidence
                        fig_conf = plot_prediction_confidence(predictions.mean(axis=0, keepdims=True))
                        st.plotly_chart(fig_conf, use_container_width=True)
                        
                        # Show per-epoch predictions
                        with st.expander("📋 Per-Epoch Predictions"):
                            pred_df = pd.DataFrame({
                                'Epoch': range(1, len(predictions) + 1),
                                'Training Prob': predictions[:, 0],
                                'Online Prob': predictions[:, 1],
                                'Predicted Class': ['Training' if p[0] > p[1] else 'Online' for p in predictions]
                            })
                            st.dataframe(pred_df, use_container_width=True)
                        
                        # Visualize preprocessed signal
                        st.markdown("### 📊 Preprocessed Signal")
                        fig_processed = plot_eeg_signal(
                            epochs[0], 
                            f"Preprocessed Epoch 1 (Model Input)",
                            max_channels=5
                        )
                        st.plotly_chart(fig_processed, use_container_width=True)
                        
                        # Power spectrum
                        fig_psd = plot_power_spectrum(epochs[0])
                        st.plotly_chart(fig_psd, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Make sure your CSV file has the correct format (samples × channels)")


# ==================== PAGE: COMPARISON ====================

def show_comparison_page():
    st.markdown('<div class="sub-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    
    # Check if comparison results exist
    results_file = "assets/model_comparison_detailed.csv"
    
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        
        st.markdown("### 📊 Performance Metrics")
        st.dataframe(df, use_container_width=True)
        
        # Bar chart comparison
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        for metric, color in zip(metrics, colors):
            if metric in df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df['Model'],
                    y=df[metric].astype(float),
                    marker_color=color,
                    text=[f'{v:.3f}' for v in df[metric].astype(float)],
                    textposition='outside'
                ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis_range=[0, 1.1],
            barmode='group',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        if 'F1-Score' in df.columns:
            best_idx = df['F1-Score'].astype(float).idxmax()
            best_model = df.loc[best_idx, 'Model']
            best_f1 = df.loc[best_idx, 'F1-Score']
            
            st.success(f"🏆 **Best Model:** {best_model} (F1-Score: {best_f1})")
        
        # Show confusion matrices if available
        st.markdown("### 🔢 Confusion Matrices")
        cm_image = "assets/confusion_matrices.png"
        if os.path.exists(cm_image):
            st.image(cm_image, caption="Confusion Matrices for All Models", use_container_width=True)
        else:
            st.info("Confusion matrix image not available")
    
    else:
        st.warning("⚠️ Model comparison results not found!")
        st.info(f"Expected file: `{results_file}`")
        st.markdown("""
        **To generate comparison results:**
        1. Run Tahap 4 (Evaluation) in Colab
        2. Download `model_comparison_detailed.csv` from results
        3. Place in `assets/` folder
        """)


# ==================== PAGE: ABOUT ====================

def show_about_page():
    st.markdown('<div class="sub-header">About This Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📖 Project Information
    
    **Title:** Deep Learning for EEG Signal Classification
    
    **Objective:** Develop and compare multiple deep learning architectures for 
    classifying EEG signals into training and online categories.
    
    ### 🎯 Methodology
    
    #### 1. Preprocessing Pipeline
    - **Bandpass Filter:** 0.5-45 Hz to remove DC drift and high-frequency noise
    - **Notch Filter:** 50 Hz to remove powerline interference
    - **Epoching:** Segment continuous signal into 4-second epochs
    - **Baseline Correction:** Remove DC offset from each epoch
    - **Z-score Normalization:** Standardize amplitude across channels
    
    #### 2. Model Architectures
    
    **CNN1D (Convolutional Neural Network 1D)**
    - Extracts local temporal features from EEG signals
    - 3 convolutional blocks with increasing filters (64→128→256)
    - Global average pooling for feature aggregation
    
    **LSTM (Long Short-Term Memory)**
    - Captures long-term temporal dependencies
    - Bidirectional architecture for forward and backward context
    - 2 LSTM layers (128 and 64 units)
    
    **CNN-LSTM Hybrid**
    - Combines spatial feature extraction (CNN) with temporal modeling (LSTM)
    - CNN layers reduce dimensionality
    - LSTM layers process temporal sequence
    
    **EEGNet**
    - State-of-the-art architecture designed specifically for EEG
    - Depthwise separable convolutions for efficiency
    - Spatial and temporal convolutions
    
    #### 3. Training Configuration
    - **Loss Function:** Sparse Categorical Crossentropy
    - **Optimizer:** Adam (lr=0.001)
    - **Batch Size:** 32
    - **Early Stopping:** Patience 15 epochs
    - **Learning Rate Reduction:** Factor 0.5, patience 5 epochs
    
    ### 📊 Evaluation Metrics
    - **Accuracy:** Overall classification accuracy
    - **Precision:** Positive predictive value
    - **Recall:** Sensitivity (True Positive Rate)
    - **F1-Score:** Harmonic mean of precision and recall
    - **Confusion Matrix:** Detailed classification breakdown
    
    ### 🔬 Dataset
    - **Subjects:** 25 participants
    - **Channels:** 20 EEG electrodes
    - **Sampling Rate:** 200 Hz
    - **Classes:** 2 (Training, Online)
    - **Data Split:** 70% train, 15% validation, 15% test
    
    ### 👥 Team
    - **Developer:** Reghina(29), Nabila(71), Tabina(76)
    - **Institution:** Program Studi Teknik Informatika, Universitas Padjadjaran.
    - **Course:** Machine Learning
    - **Year:** 2025
    
    ### 📚 References
    1. Lawhern et al. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs"
    2. Schirrmeister et al. (2017). "Deep learning with convolutional neural networks for EEG decoding"
    
    """)


# ==================== RUN APP ====================

if __name__ == "__main__":
    main()