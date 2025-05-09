import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import random
from typing import List, Tuple, Dict, Optional

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Nuclear Reactor Monitoring System",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
SEQUENCE_LENGTH = 18  # Number of time steps (10-second intervals = 180 seconds)
TCN_MODEL_PATH = "../models/prediction/saved_model/model_fold_2.h5"
HYBRID_MODEL_PATH = "../models/classification/DL/saved_models/best_model.h5"
SCALER_PATH = "../models/prediction/scaler.pkl"
FEATURE_COLS_PATH = "../models/prediction/feature_columns.json"
ACCIDENT_TYPES = ["FLB","LLB","LOCA","LOCAC","LR","MD","RI","RW","SGATR","SGBTR","SLBIC","SLBOC"]  

class ReactorMonitor:
    def __init__(self):
        # Load models and preprocessors
        self.load_models()
        self.current_data_buffer = []
        self.scram_detected = False
        self.accident_type = None
        self.confidence = 0.0
    
    def load_models(self):
        """Initialize preprocessors and use dummy models for testing"""
        try:
            # # Check if model files exist but don't actually try to load them
            # if not os.path.exists(TCN_MODEL_PATH):
            #     st.sidebar.warning(f"⚠️ TCN model file not found: {TCN_MODEL_PATH}")
            # else:
            #     st.sidebar.info(f"✅ TCN model file found (not loaded): {TCN_MODEL_PATH}")
            
            # if not os.path.exists(HYBRID_MODEL_PATH):
            #     st.sidebar.warning(f"⚠️ Hybrid model file not found: {HYBRID_MODEL_PATH}")
            # else:
            #     st.sidebar.info(f"✅ Hybrid model file found (not loaded): {HYBRID_MODEL_PATH}")
            
            # Load scaler for feature normalization if available
            if os.path.exists(SCALER_PATH):
                try:
                    self.scaler = joblib.load(SCALER_PATH)
                    # st.sidebar.success("✅ Scaler loaded successfully")
                except Exception as e:
                    # st.sidebar.warning(f"⚠️ Error loading scaler: {str(e)}")
                    self.scaler = None
            else:
                # st.sidebar.warning(f"⚠️ Scaler file not found: {SCALER_PATH}")
                self.scaler = None
            
            # Load feature columns if available
            if os.path.exists(FEATURE_COLS_PATH):
                try:
                    import json
                    with open(FEATURE_COLS_PATH, 'r') as f:
                        self.feature_columns = json.load(f)
                    # st.sidebar.success("✅ Feature columns loaded successfully")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Error loading feature columns: {str(e)}")
                    self.feature_columns = None
            else:
                st.sidebar.warning(f"⚠️ Feature columns file not found: {FEATURE_COLS_PATH}")
                self.feature_columns = None
            
            # Using dummy models for testing
            # st.sidebar.info("ℹ️ Using dummy prediction mode for testing")
            
            return True
        except Exception as e:
            st.sidebar.error(f"❌ Error initializing: {str(e)}")
            self.scaler = None
            self.feature_columns = None
            return False
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess the input data for model prediction"""
        # Extract the relevant features in the correct order
        if self.feature_columns:
            # Filter and reorder columns based on feature_columns
            try:
                data = data[self.feature_columns]
            except KeyError:
                # If columns don't match, just use all columns
                pass
        
        # Normalize data if scaler is available
        if self.scaler:
            try:
                data_np = data.values
                data_np = self.scaler.transform(data_np)
            except Exception:
                data_np = data.values
        else:
            data_np = data.values
        
        return data_np
    
    def predict_scram(self, data_sequence: np.ndarray) -> Tuple[bool, float]:
        """Dummy prediction for reactor scram"""
        # Simulate scram detection - more likely as buffer fills up
        buffer_fill_ratio = len(self.current_data_buffer) / SEQUENCE_LENGTH
        base_probability = 0.1 + (buffer_fill_ratio * 0.1)  # Increases with more data
        
        # Add some randomness based on the data values
        if data_sequence.size > 0:
            data_factor = np.mean(np.abs(data_sequence)) * 0.01
            probability = min(0.95, base_probability + data_factor + random.uniform(-0.1, 0.1))
        else:
            probability = base_probability + random.uniform(-0.1, 0.1)
        
        # More likely to trigger alarm as we approach full buffer
        threshold = 0.7 if buffer_fill_ratio > 0.8 else 0.9
        return probability > threshold, probability
    
    def classify_accident(self, data_sequence: np.ndarray) -> Tuple[str, float]:
        """Dummy classification for accident type"""
        # Simple random selection of accident type with high confidence
        idx = random.randint(0, len(ACCIDENT_TYPES) - 1)
        confidence = random.uniform(0.7, 0.99)
        return ACCIDENT_TYPES[idx], confidence
    
    def update_buffer(self, new_data: pd.DataFrame):
        """Update the data buffer with new data"""
        preprocessed_data = self.preprocess_data(new_data)
        
        # Add to buffer, keeping only the most recent SEQUENCE_LENGTH points
        self.current_data_buffer.append(preprocessed_data)
        if len(self.current_data_buffer) > SEQUENCE_LENGTH:
            self.current_data_buffer.pop(0)
    
    def process_data(self) -> Dict:
        """Process the current data buffer and return predictions"""
        results = {
            "scram_detected": False,
            "scram_probability": 0.0,
            "accident_type": None,
            "accident_confidence": 0.0
        }
        
        # Check if we have enough data points
        if len(self.current_data_buffer) < SEQUENCE_LENGTH:
            results["message"] = f"Waiting for more data... ({len(self.current_data_buffer)}/{SEQUENCE_LENGTH})"
            return results
        
        # Create data sequence for prediction
        data_sequence = np.array(self.current_data_buffer)
        
        # Predict scram
        scram_detected, scram_probability = self.predict_scram(data_sequence)
        results["scram_detected"] = scram_detected
        results["scram_probability"] = scram_probability
        
        # If scram is detected, classify accident type
        if scram_detected:
            accident_type, confidence = self.classify_accident(data_sequence)
            results["accident_type"] = accident_type
            results["accident_confidence"] = confidence
            results["message"] = f"⚠️ Reactor scram predicted! Accident type: {accident_type} (confidence: {confidence:.2f})"
            self.scram_detected = True
            self.accident_type = accident_type
            self.confidence = confidence
        else:
            results["message"] = "✅ Normal operation - No issues detected"
            
        return results

def create_dashboard():
    """Create the Streamlit dashboard"""
    st.title("⚛️ Nuclear Reactor Monitoring System")
    
    # Initialize session state for monitoring
    if 'monitor' not in st.session_state:
        st.session_state.monitor = ReactorMonitor()
        st.session_state.buffer = []
        st.session_state.results_history = []
        st.session_state.running = False
        st.session_state.last_data = None
        st.session_state.alert_status = "normal"  # normal, warning, critical
    
    # Ensure all required session state variables exist
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'buffer' not in st.session_state:
        st.session_state.buffer = []
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []
    if 'last_data' not in st.session_state:
        st.session_state.last_data = None
    if 'alert_status' not in st.session_state:
        st.session_state.alert_status = "normal"
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # File uploader for CSV data
        uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button("Start Monitoring", type="primary", disabled=st.session_state.running)
        
        with col2:
            stop_button = st.button("Stop Monitoring", type="secondary", disabled=not st.session_state.running)
        
        # Manual input for testing
        st.divider()
        st.subheader("Test Controls")
        test_button = st.button("Simulate Scram", help="Simulate a reactor scram for testing")
    
    # Main display area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Status panel
        status_container = st.container()
        with status_container:
            if st.session_state.alert_status == "normal":
                st.success("✅ SYSTEM NORMAL - No issues detected")
            elif st.session_state.alert_status == "warning":
                st.warning("⚠️ WARNING - Potential issues detected")
            elif st.session_state.alert_status == "critical":
                st.error("🚨 CRITICAL - Reactor scram predicted")
        
        # Data visualization
        st.subheader("Reactor Parameters")
        chart_container = st.container()
    
    with col2:
        # Prediction results
        st.subheader("Prediction Results")
        results_container = st.container()
        
        with results_container:
            st.metric("Scram Probability", f"{st.session_state.monitor.confidence:.2%}" if st.session_state.monitor.confidence else "0.00%")
            
            if st.session_state.monitor.scram_detected:
                st.error(f"⚠️ Accident Type: {st.session_state.monitor.accident_type}")
                st.error(f"⚠️ Confidence: {st.session_state.monitor.confidence:.2%}")
    
    # Process the uploaded file
    if uploaded_file is not None and not st.session_state.running:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            st.session_state.data_preview = df.head()
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data_preview)
            
            # Enable start button
            st.sidebar.success("✅ Data loaded successfully. Click 'Start Monitoring' to begin.")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")
    
    # Handle start monitoring
    if start_button and 'uploaded_data' in st.session_state:
        st.session_state.running = True
        st.session_state.buffer = []
        st.session_state.results_history = []
        st.session_state.monitor.scram_detected = False
        st.session_state.monitor.accident_type = None
        st.session_state.monitor.confidence = 0.0
        st.session_state.alert_status = "normal"
        st.experimental_rerun()
    
    # Handle stop monitoring
    if stop_button:
        st.session_state.running = False
        st.experimental_rerun()
    
    # Test button for simulating scram
    if test_button:
        st.session_state.monitor.scram_detected = True
        st.session_state.monitor.accident_type = "LOCA"
        st.session_state.monitor.confidence = 0.95
        st.session_state.alert_status = "critical"
        st.experimental_rerun()
    
    # Simulate real-time monitoring when running
    if st.session_state.running and 'uploaded_data' in st.session_state:
        # Get the data
        data = st.session_state.uploaded_data
        
        # Create a placeholder for updating status
        status_placeholder = st.empty()
        chart_placeholder = chart_container.empty()
        
        # Process each row in the data to simulate real-time
        data_iter = st.empty()
        
        # Initialize progress
        progress_bar = st.progress(0)
        
        # Process the data row by row
        for i, row in enumerate(data.iterrows()):
            if not st.session_state.running:
                break
                
            # Update progress
            progress = i / len(data)
            progress_bar.progress(progress)
            
            # Get current row data
            current_data = row[1].to_frame().T
            
            # Update the data buffer
            st.session_state.monitor.update_buffer(current_data)
            
            # Process the data and get results
            results = st.session_state.monitor.process_data()
            
            # Update status based on results
            if results["scram_detected"]:
                st.session_state.alert_status = "critical"
            elif results["scram_probability"] > 0.3:
                st.session_state.alert_status = "warning"
            else:
                st.session_state.alert_status = "normal"
            
            # Update display
            with status_placeholder:
                st.info(f"Processing time step {i+1}/{len(data)}: {results['message']}")
            
            # Update chart with latest data
            with chart_placeholder:
                if len(st.session_state.buffer) > 0:
                    chart_data = pd.DataFrame(st.session_state.buffer)
                    st.line_chart(chart_data)
            
            # Add to buffer for visualization
            if len(st.session_state.buffer) > 100:
                st.session_state.buffer.pop(0)
            st.session_state.buffer.append({
                "Temperature": current_data["TAVG"].values[0] if "TAVG" in current_data.columns else 0,
                "Pressure": current_data["P"].values[0] if "P" in current_data.columns else 0,
                "Scram Probability": results["scram_probability"]
            })
            
            # Save results
            st.session_state.results_history.append(results)
            
            # Pause to simulate real-time
            time.sleep(0.1)
        
        # Finished processing
        progress_bar.progress(1.0)
        st.session_state.running = False
        st.success("✅ Monitoring completed!")
        
        # Show final status
        if st.session_state.monitor.scram_detected:
            st.error(f"🚨 Reactor scram detected! Accident type: {st.session_state.monitor.accident_type} with {st.session_state.monitor.confidence:.2%} confidence")
        else:
            st.success("✅ No reactor scram detected during the monitoring period")

if __name__ == "__main__":
    create_dashboard() 