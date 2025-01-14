# Artificial Neural Network for Flood Prediction

This project implements an Artificial Neural Network (ANN) to predict the *Index Flood* based on environmental features such as area, wetness, and other parameters from hydrological datasets. The network leverages modular code organization, enabling customization and scalability.

## Features

- **Modular Design**: Organized into distinct modules for data processing, model training, prediction, and visualization.
- **Customizable ANN**: Hyperparameters like hidden layers, epochs, and learning rates can be adjusted in a configuration file (`config/config.py`).
- **Data Standardization**: Implements robust data preprocessing for improved model performance.
- **Interactive Visualization**: Plots for validation errors and predicted vs. actual floods are generated using `plotly`.
- **Evaluation Metrics**: Includes metrics like Mean Absolute Error (MAE) and correlation coefficients for model evaluation.
- **Comparison Tool**: Compare ANN predictions with external data (e.g., LINEST predictions) using a provided Excel file.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Modules](#modules)

---

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo-link.git
    cd artificial-neural-network
    ```

2. Set up a virtual environment (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Place the required Excel files (`FEHDataStudent.xlsx` and `compare-data.xlsx`) in the `data` directory.

---

## Usage

1. **Train the Model**: Run the main script to process the data, train the model, and visualize results:

    ```bash
    python artificial-neural-network.py
    ```

2. **Hyperparameter Tuning**: Adjust hyperparameters such as learning rate, hidden layer size, and number of epochs in `config/config.py`.

3. **Compare Predictions**: To evaluate model predictions against external data (e.g., LINEST results), ensure `compare-data.xlsx` is in the `data` directory, then execute the script as usual.

4. **Output**: The script generates interactive plots and prints evaluation metrics to the console.

---

## File Structure

```plaintext
artificial-neural-network/
├── config/
│   ├── config.py           # Configuration file for paths and hyperparameters
├── data/
│   ├── FEHDataStudent.xlsx # Input dataset for training and evaluation
│   ├── compare-data.xlsx   # External dataset for prediction comparison
├── models/
│   ├── flood_predictor.py  # Prediction and evaluation logic
│   ├── neural_network.py   # Neural network training logic
├── processing/
│   ├── data_processor.py   # Data loading and cleaning
│   ├── data_splitter.py    # Data splitting logic
│   ├── data_standardiser.py # Data standardization
├── visualisation/
│   ├── result_visualiser.py # Visualization logic for results and metrics
├── artificial-neural-network.py  # Main script for training, prediction, and evaluation
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation