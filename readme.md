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

Dependencies

Make sure to install the following dependencies before running the project. These can be found in requirements.txt:

pandas
plotly
openpyxl (for reading .xlsx files)
numpy
scikit-learn
To install all dependencies:

bash
Copy code
pip install -r requirements.txt
Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with a detailed explanation of your changes.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For any issues, questions, or suggestions, feel free to reach out at your-email@example.com or create an issue on the GitHub repository.

markdown
Copy code

---

### Explanation of Each Section:

1. **Project Title**: The title describes the core function of your project.
2. **Description**: Provides a concise overview of the project's purpose.
3. **Features**: Highlights the important capabilities of the system.
4. **Installation**: Step-by-step instructions for setting up the project, including installing dependencies and preparing the environment.
5. **Usage**: Explains how to run the project and what the user should expect during execution.
6. **File Structure**: A breakdown of the project files and their purposes, helping users understand where to find relevant code and data.
7. **Configuration**: Describes how users can configure parameters for their own use, including hyperparameters and file paths.
8. **Dependencies**: Lists the Python packages required to run the project.
9. **Contributing**: For developers interested in contributing to the project.
10. **License**: Ensures others know the terms under which they can use the code.
11. **Contact**: A way for people to reach out with issues or questions.