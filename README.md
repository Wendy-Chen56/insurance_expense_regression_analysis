# Insurance Expense Regression Analysis

A reproducible Python project that uses a real-world medical insurance dataset to examine how BMI and age relate to annual medical expenses. The project implements linear regression from first principles and compares the Normal Equation with Gradient Descent.

## Business Question

Can BMI and age help explain differences in medical insurance expenses, and does adding age improve predictive performance compared with using BMI alone?

This project evaluates three models:

1. BMI-only linear regression baseline
2. BMI and age regression using the Normal Equation
3. BMI and age regression using Gradient Descent

## Key Findings

| Model | BMI Coefficient | Age Coefficient | R² |
|---|---:|---:|---:|
| BMI-Only Baseline | 394.33 | N/A | 0.0394 |
| BMI + Age Normal Equation | 333.39 | 241.90 | 0.1173 |
| BMI + Age Gradient Descent | 333.39 | 241.90 | 0.1173 |

Adding age increased R² from 0.0394 to 0.1173, an improvement of 0.0779, or 7.79 percentage points. Holding BMI constant, each additional year of age is associated with approximately $241.90 in higher predicted medical expenses.

The Normal Equation and Gradient Descent produced the same coefficients and evaluation metrics, confirming that both methods converged to the same ordinary least squares solution.

The two-feature model is not ready for deployment because it explains only 11.73% of the variation in expenses. Additional predictors, validation on unseen data, and further model evaluation would be required before using the model for business decisions.

## Skills Demonstrated

- Loading and inspecting a real-world dataset
- Implementing multiple linear regression from first principles
- Solving a general Normal Equation with matrix operations
- Standardizing multiple features for Gradient Descent
- Converting standardized coefficients back to original units
- Calculating MSE, RMSE, MAE, and R²
- Comparing analytical and iterative optimization methods
- Exporting reproducible model results
- Translating model coefficients into business language
- Evaluating model limitations and deployment readiness

## Repository Contents

```text
insurance_expense_regression_analysis/
├── 01_simple_linear.py
├── 02_ols_normal_equation.py
├── 03_gradient_descent.py
├── 04_compare_models_visual.py
├── assignment_two_feature_model.py
├── assignment_memo.md
├── linear_regression_lab.ipynb
├── requirements.txt
├── README.md
├── data/
│   ├── insurance-premium-prediction/
│   │   └── insurance.csv
│   └── model_comparison_bmi_expenses.png
└── reports/
    └── assignment_results.csv
```

## Dataset

The project uses the Insurance Premium Prediction dataset available through Kaggle.

The dataset contains 1,338 observations. This analysis uses:

- `bmi`: body mass index
- `age`: age of the insured individual
- `expenses`: annual medical insurance expenses

A local copy of `insurance.csv` is included so the project can run without requiring the user to create a Kaggle account.

## Before You Begin

You need:

- A Windows or macOS computer
- Internet access for the initial software and package installation
- Python 3.10 or newer
- Visual Studio Code
- Approximately 10–20 minutes for setup

No previous Python or Git experience is required.

## Install Python

### Windows

1. Visit [python.org/downloads](https://www.python.org/downloads/).
2. Download the current Python 3 installer for Windows.
3. Open the installer.
4. Select the checkbox labeled `Add python.exe to PATH`.
5. Select `Install Now`.
6. Wait for the installation to finish.

### macOS

1. Visit [python.org/downloads](https://www.python.org/downloads/).
2. Download the current Python 3 installer for macOS.
3. Open the downloaded `.pkg` file.
4. Follow the installation instructions.

## Install Visual Studio Code

1. Visit [code.visualstudio.com](https://code.visualstudio.com/).
2. Download the version for your operating system.
3. Install Visual Studio Code using the default options.
4. Open Visual Studio Code.
5. Select the Extensions icon on the left.
6. Search for `Python`.
7. Install the Python extension published by Microsoft.
8. Search for `Jupyter`.
9. Install the Jupyter extension published by Microsoft.

## Download the Project

Choose either Option A or Option B.

### Option A: Download ZIP — Recommended for Beginners

1. Open the project’s GitHub page:

   [Insurance Expense Regression Analysis](https://github.com/Wendy-Chen56/insurance_expense_regression_analysis)

2. Select the green `Code` button.
3. Select `Download ZIP`.
4. Open the computer’s Downloads folder.
5. Right-click the downloaded ZIP file.
6. On Windows, select `Extract All`.
7. On macOS, double-click the ZIP file.
8. Open Visual Studio Code.
9. Select `File` and then `Open Folder`.
10. Select the extracted project folder.
11. Select `Open` or `Select Folder`.

### Option B: Clone with Git

This option requires [Git](https://git-scm.com/downloads).

Open Terminal on macOS or PowerShell on Windows and run:

```bash
git clone https://github.com/Wendy-Chen56/insurance_expense_regression_analysis.git
cd insurance_expense_regression_analysis
```

Then open the cloned folder in Visual Studio Code.

## Windows Setup Instructions

### 1. Open the VS Code Terminal

In Visual Studio Code, select:

```text
Terminal → New Terminal
```

The Terminal should open at the bottom of the screen.

### 2. Confirm Python Is Installed

Enter:

```powershell
python --version
```

You should see Python 3.10 or newer.

If `python` is not recognized, try:

```powershell
py --version
```

### 3. Create a Virtual Environment

Enter:

```powershell
python -m venv venv
```

If your computer uses `py`, enter:

```powershell
py -m venv venv
```

### 4. Allow Activation for the Current Terminal

Enter:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If PowerShell asks for confirmation, type `Y` and press Enter.

This change applies only to the current Terminal window.

### 5. Activate the Virtual Environment

Enter:

```powershell
.\venv\Scripts\Activate.ps1
```

The Terminal should begin with:

```text
(venv)
```

### 6. Install the Required Packages

Enter:

```powershell
python -m pip install -r requirements.txt
```

Wait until the installation finishes.

### 7. Run the Main Analysis

Enter:

```powershell
python assignment_two_feature_model.py
```

## macOS Setup Instructions

### 1. Open the VS Code Terminal

In Visual Studio Code, select:

```text
Terminal → New Terminal
```

### 2. Confirm Python Is Installed

Enter:

```bash
python3 --version
```

You should see Python 3.10 or newer.

### 3. Create a Virtual Environment

Enter:

```bash
python3 -m venv venv
```

### 4. Activate the Virtual Environment

Enter:

```bash
source venv/bin/activate
```

The Terminal should begin with:

```text
(venv)
```

### 5. Install the Required Packages

Enter:

```bash
python3 -m pip install -r requirements.txt
```

Wait until the installation finishes.

### 6. Run the Main Analysis

Enter:

```bash
python3 assignment_two_feature_model.py
```

## Expected Results

A successful run begins by showing:

```text
Number of observations: 1338
```

The main model results should be approximately:

```text
Two-Feature Normal Equation
w0 (intercept): -6437.35
w1 (bmi):        333.39
w2 (age):        241.90
R^2:               0.1173

Two-Feature Gradient Descent
w0 (intercept): -6437.35
w1 (bmi):        333.39
w2 (age):        241.90
R^2:               0.1173

BMI-Only Baseline
w0 (intercept):  1178.18
w1 (bmi):         394.33
R^2:                0.0394
```

The script also creates or updates:

```text
reports/assignment_results.csv
```

Open this CSV file in Visual Studio Code or Excel to view the model comparison table.

Small differences in the last decimal place may occur because of floating-point calculations.

## Run the Learning Scripts

To review how the project develops from a single-feature model to a model comparison, run the scripts in this order.

### Windows

```powershell
python 01_simple_linear.py
python 02_ols_normal_equation.py
python 03_gradient_descent.py
python 04_compare_models_visual.py
```

### macOS

```bash
python3 01_simple_linear.py
python3 02_ols_normal_equation.py
python3 03_gradient_descent.py
python3 04_compare_models_visual.py
```

## Run the Jupyter Notebook

With the virtual environment activated, enter:

### Windows

```powershell
python -m notebook
```

### macOS

```bash
python3 -m notebook
```

A browser window should open.

1. Select `linear_regression_lab.ipynb`.
2. Confirm that the selected kernel uses the project’s `venv`.
3. Select `Run All`.
4. Wait for all cells to finish.

## Troubleshooting

### `python` Is Not Recognized

Python may not have been added to PATH.

On Windows, try:

```powershell
py --version
```

If that works, replace `python` with `py` in the setup commands.

If neither command works, reinstall Python and select `Add python.exe to PATH`.

### PowerShell Blocks Activate.ps1

Enter:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

### `ModuleNotFoundError`

Confirm that the Terminal begins with `(venv)`, and then run:

```powershell
python -m pip install -r requirements.txt
```

On macOS, run:

```bash
python3 -m pip install -r requirements.txt
```

### `No module named kagglehub`

Activate the virtual environment and reinstall the packages:

```powershell
python -m pip install -r requirements.txt
```

### The Dataset Cannot Be Found

Confirm that this file exists:

```text
data/insurance-premium-prediction/insurance.csv
```

Run commands from the project’s main folder, not from the `data` or `reports` folder.

### The Wrong Python Interpreter Is Selected in VS Code

1. Press `Ctrl + Shift + P` on Windows or `Command + Shift + P` on macOS.
2. Search for `Python: Select Interpreter`.
3. Select the Python interpreter located inside the project’s `venv`.

### The Notebook Cannot Find Installed Packages

Use the kernel menu in the upper-right corner of the notebook and select the interpreter from the project’s `venv`.

## Analytical Challenges and Lessons Learned

One challenge was extending the Normal Equation from a single-feature shortcut to a general matrix solution that supports multiple predictors. The two features also needed to be standardized before Gradient Descent so that differences in their numerical scales would not create unstable updates.

After optimization, the Gradient Descent coefficients had to be converted back to the original BMI and age units. This step was necessary so the coefficients could be directly compared with the Normal Equation and interpreted by non-technical stakeholders.

The identical results from both methods provided an important validation check. However, the low R² showed that numerical agreement between algorithms does not automatically mean that a model has strong predictive performance.

## Limitations and Next Steps

The current model uses only BMI and age and does not capture many other factors related to medical expenses. Future analysis should consider additional variables such as smoking status, number of children, sex, and region.

Before deployment, the data should be separated into training, validation, and test sets. Future work could also examine interaction effects, non-linear relationships, residual diagnostics, and alternative predictive models.

## Reproduction Checklist

A successful independent reproduction should confirm that:

- The virtual environment can be created and activated.
- All packages install without errors.
- The dataset contains 1,338 observations.
- The main script runs without modification.
- Both two-feature methods produce approximately `R² = 0.1173`.
- The BMI-only baseline produces approximately `R² = 0.0394`.
- `reports/assignment_results.csv` is created successfully.

## Author

Wendy Chen  
Master of Science in Business Analytics  
University of the Pacific