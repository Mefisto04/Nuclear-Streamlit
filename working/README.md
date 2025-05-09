# Nuclear Reactor Monitoring System

This Streamlit application monitors nuclear reactor parameters and detects potential issues using PyTorch machine learning models.

## Features

- Real-time monitoring of reactor parameters
- Prediction of reactor scram events using a TCN-Attention model
- Classification of accident types using a Hybrid LSTM-GRU model
- Interactive visualization of reactor parameters
- Simulated real-time monitoring of uploaded CSV data

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/nuclear-reactor-monitoring.git
cd nuclear-reactor-monitoring
```

2. Install the required dependencies:

```bash
pip install -r working/requirements.txt
```

## Usage

1. Make sure you have the trained models in the correct locations:

   - TCN model: `models/prediction/saved_model/model_fold2.pt`
   - Hybrid model: `models/classification/DL/saved_models/best_model.pt`
   - Scaler: `models/prediction/scaler.pkl`
   - Feature columns: `models/prediction/feature_columns.json`

2. Run the Streamlit app:

```bash
cd working
streamlit run app.py
```

3. Access the application in your web browser at `http://localhost:8501`

## Using the Application

1. **Upload Data**: Upload a CSV file containing reactor parameter data.
2. **Start Monitoring**: Click the "Start Monitoring" button to begin processing the data.
3. **View Results**: The app will display real-time monitoring results and visualize key parameters.
4. **Testing**: Use the "Simulate Scram" button to test how the app responds to a simulated reactor scram event.

## Data Format

The CSV file should include reactor parameters similar to the reference data (`NPPAD/LOCA/1.csv`). Important columns include:

- `TIME`: Time stamps
- `TAVG`: Average temperature
- `P`: Pressure
- Various other reactor parameters

## How It Works

1. **Scram Prediction**: A TCN-Attention model analyzes 18 time steps (180 seconds) of reactor data to predict if a scram event will occur.
2. **Accident Classification**: If a scram is predicted, a Hybrid LSTM-GRU model classifies the type of accident.
3. **Visualization**: The application displays reactor parameters and prediction results in real-time.

## Development

The application consists of:

- `app.py`: Main Streamlit application
- Machine learning models in the `models/` directory
- Example data in the `NPPAD/` directory

## License

[MIT License](LICENSE)
