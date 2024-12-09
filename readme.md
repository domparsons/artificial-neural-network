# Artificial Neural Network for Flood Prediction

This project implements an Artificial Neural Network (ANN) to predict the *Index Flood* based on environmental features such as area, wetness, and other parameters from hydrological datasets. The network is trained using a custom dataset and evaluated for its performance using error metrics like Mean Absolute Error (MAE) and correlation coefficients.

## Features

- Customizable ANN structure with hyperparameters.
- Data standardization and splitting into training, validation, and test sets.
- Flood prediction using a trained model with interactive data visualization using `plotly`.
- Validation and evaluation metrics, including error rates and precision.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo-link.git
    cd artificial-neural-network
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Make sure you have the necessary Excel data files (`FEHDataStudent.xlsx` and `compare-data.xlsx`) located in the project root directory.

## Usage

1. **Train the Model**: After setting up the project, you can train the model by running:

    ```bash
    python artificial-neural-network.py
    ```

2. **Hyperparameters**: You can customize the hyperparameters like `epochs`, `learning rates`, and `hidden layers` in the `config.yaml` file.

3. **Plotting**: Interactive plots for validation errors and the predicted vs actual index flood can be generated, and they'll open automatically in your browser.

4. **Comparing Data**: Use the `compare_with_linest` function to compare model predictions with LINEST predictions from the Excel file.