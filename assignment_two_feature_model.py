import csv
from pathlib import Path


def solve_linear_system(
    matrix: list[list[float]],
    vector: list[float]
) -> list[float]:
    """Solve A*x = b using Gauss-Jordan elimination."""

    n = len(vector)

    # Build augmented matrix [A | b]
    aug = [
        [float(value) for value in matrix[i]] + [float(vector[i])]
        for i in range(n)
    ]

    for col in range(n):
        # Find the best pivot row
        pivot_row = max(
            range(col, n),
            key=lambda row: abs(aug[row][col])
        )

        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Singular matrix; cannot solve system.")

        # Swap rows
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Make pivot equal to 1
        pivot = aug[col][col]
        aug[col] = [value / pivot for value in aug[col]]

        # Eliminate this column from all other rows
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [
                    aug[row][j] - factor * aug[col][j]
                    for j in range(n + 1)
                ]

    return [aug[i][-1] for i in range(n)]


def fit_normal_equation_two_features(
    bmi: list[float],
    age: list[float],
    expenses: list[float]
)-> tuple[float, float, float]:
    """Fit expenses = w0 + w1*bmi + w2*age using the Normal Equation."""

    n = len(expenses)

    sum_bmi = sum(bmi)
    sum_age = sum(age)
    sum_y = sum(expenses)

    sum_bmi2 = sum(x * x for x in bmi)
    sum_age2 = sum(x * x for x in age)
    sum_bmi_age = sum(x1 * x2 for x1, x2 in zip(bmi, age))

    sum_bmi_y = sum(x * y for x, y in zip(bmi, expenses))
    sum_age_y = sum(x * y for x, y in zip(age, expenses))

    # X^T X for intercept + bmi + age
    xtx = [
        [n,           sum_bmi,      sum_age],
        [sum_bmi,     sum_bmi2,     sum_bmi_age],
        [sum_age,     sum_bmi_age,  sum_age2],
    ]

    # X^T y
    xty = [
        sum_y,
        sum_bmi_y,
        sum_age_y,
    ]

    w0, w1, w2 = solve_linear_system(xtx, xty)

    return w0, w1, w2


def mse(y_true: list[float], y_pred: list[float]) -> float:
    """Mean Squared Error."""
    n = len(y_true)
    return sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(y_true, y_pred)
    ) / n


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Root Mean Squared Error."""
    return mse(y_true, y_pred) ** 0.5


def mae(y_true: list[float], y_pred: list[float]) -> float:
    """Mean Absolute Error."""
    n = len(y_true)
    return sum(
        abs(actual - predicted)
        for actual, predicted in zip(y_true, y_pred)
    ) / n


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    """R-squared score."""
    y_mean = sum(y_true) / len(y_true)

    ss_res = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(y_true, y_pred)
    )

    ss_tot = sum(
        (actual - y_mean) ** 2
        for actual in y_true
    )

    return 1.0 - ss_res / ss_tot


def standardize(values: list[float]) -> tuple[list[float], float, float]:
    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    std = variance ** 0.5

    if std == 0:
        return [0.0 for _ in values], mean, 1.0

    standardized = [
        (value - mean) / std
        for value in values
    ]

    return standardized, mean, std

def fit_normal_equation_single_feature(
    x: list[float],
    y: list[float]
) -> tuple[float, float]:
    """Fit y = w0 + w1*x using the Normal Equation for one feature."""

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(v * v for v in x)
    sum_xy = sum(vx * vy for vx, vy in zip(x, y))

    # X^T X = [[n, sum_x], [sum_x, sum_x2]]
    a = float(n)
    b = sum_x
    c = sum_x
    d = sum_x2

    det = a * d - b * c

    if det == 0:
        raise ValueError(
            "Singular matrix in normal equation; cannot invert X^T X."
        )

    inv_xtx = [
        [d / det, -b / det],
        [-c / det, a / det]
    ]

    xty = [sum_y, sum_xy]

    w0 = (
        inv_xtx[0][0] * xty[0]
        + inv_xtx[0][1] * xty[1]
    )

    w1 = (
        inv_xtx[1][0] * xty[0]
        + inv_xtx[1][1] * xty[1]
    )

    return w0, w1



def fit_gradient_descent_two_features(
    bmi: list[float],
    age: list[float],
    expenses: list[float],
    learning_rate: float = 0.05,
    epochs: int = 10000
) -> tuple[float, float, float]:
    """Fit expenses = w0 + w1*bmi + w2*age using Gradient Descent."""

    # Standardize both features
    bmi_std, bmi_mean, bmi_std_dev = standardize(bmi)
    age_std, age_mean, age_std_dev = standardize(age)

    # Start weights at zero
    w0 = 0.0
    w1 = 0.0
    w2 = 0.0

    n = len(expenses)

    # Gradient Descent
    for epoch in range(1, epochs + 1):
        predictions = [
            w0 + w1 * bmi_i + w2 * age_i
            for bmi_i, age_i in zip(bmi_std, age_std)
        ]

        errors = [
            predicted - actual
            for predicted, actual in zip(predictions, expenses)
        ]

        grad_w0 = (2.0 / n) * sum(errors)

        grad_w1 = (2.0 / n) * sum(
            error * bmi_i
            for error, bmi_i in zip(errors, bmi_std)
        )

        grad_w2 = (2.0 / n) * sum(
            error * age_i
            for error, age_i in zip(errors, age_std)
        )

        w0 -= learning_rate * grad_w0
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

    # Convert standardized weights back to original units
    w1_original = w1 / bmi_std_dev
    w2_original = w2 / age_std_dev

    w0_original = (
        w0
        - w1_original * bmi_mean
        - w2_original * age_mean
    )

    return w0_original, w1_original, w2_original

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "insurance-premium-prediction"
    / "insurance.csv"
)

bmi = []
age = []
expenses = []

with open(DATA_PATH, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        bmi.append(float(row["bmi"]))
        age.append(float(row["age"]))
        expenses.append(float(row["expenses"]))


print("Number of observations:", len(expenses))
print("First 5 BMI values:", bmi[:5])
print("First 5 age values:", age[:5])
print("First 5 expense values:", expenses[:5])

w0_ne, w1_ne, w2_ne = fit_normal_equation_two_features(
    bmi,
    age,
    expenses
)

preds_ne = [
    w0_ne + w1_ne * bmi_i + w2_ne * age_i
    for bmi_i, age_i in zip(bmi, age)
]

mse_ne = mse(expenses, preds_ne)
rmse_ne = rmse(expenses, preds_ne)
mae_ne = mae(expenses, preds_ne)
r2_ne = r2_score(expenses, preds_ne)

print()
print("Two-Feature Normal Equation")
print(f"w0 (intercept): {w0_ne:.2f}")
print(f"w1 (bmi):       {w1_ne:.2f}")
print(f"w2 (age):       {w2_ne:.2f}")
print(f"MSE:            {mse_ne:.2f}")
print(f"RMSE:           {rmse_ne:.2f}")
print(f"MAE:            {mae_ne:.2f}")
print(f"R^2:            {r2_ne:.4f}")


w0_gd, w1_gd, w2_gd = fit_gradient_descent_two_features(
    bmi,
    age,
    expenses
)

preds_gd = [
    w0_gd + w1_gd * bmi_i + w2_gd * age_i
    for bmi_i, age_i in zip(bmi, age)
]

mse_gd = mse(expenses, preds_gd)
rmse_gd = rmse(expenses, preds_gd)
mae_gd = mae(expenses, preds_gd)
r2_gd = r2_score(expenses, preds_gd)

print()
print("Two-Feature Gradient Descent")
print(f"w0 (intercept): {w0_gd:.2f}")
print(f"w1 (bmi):       {w1_gd:.2f}")
print(f"w2 (age):       {w2_gd:.2f}")
print(f"MSE:            {mse_gd:.2f}")
print(f"RMSE:           {rmse_gd:.2f}")
print(f"MAE:            {mae_gd:.2f}")
print(f"R^2:            {r2_gd:.4f}")


# ---------------------------------------------------------
# BMI-only baseline from 02_ols_normal_equation.py
# ---------------------------------------------------------

w0_base, w1_base = fit_normal_equation_single_feature(
    bmi,
    expenses
)

preds_base = [
    w0_base + w1_base * bmi_i
    for bmi_i in bmi
]

mse_base = mse(expenses, preds_base)
rmse_base = rmse(expenses, preds_base)
mae_base = mae(expenses, preds_base)
r2_base = r2_score(expenses, preds_base)

print()
print("BMI-Only Baseline")
print(f"w0 (intercept): {w0_base:.2f}")
print(f"w1 (bmi):       {w1_base:.2f}")
print(f"MSE:            {mse_base:.2f}")
print(f"RMSE:           {rmse_base:.2f}")
print(f"MAE:            {mae_base:.2f}")
print(f"R^2:            {r2_base:.4f}")

print()
print("Model Comparison")
print("-" * 105)

print(
    f"{'Model':<30}"
    f"{'w0':>12}"
    f"{'w1 (BMI)':>12}"
    f"{'w2 (Age)':>12}"
    f"{'MSE':>16}"
    f"{'RMSE':>12}"
    f"{'MAE':>12}"
    f"{'R^2':>10}"
)

print("-" * 105)

print(
    f"{'BMI-Only Baseline':<30}"
    f"{w0_base:>12.2f}"
    f"{w1_base:>12.2f}"
    f"{'N/A':>12}"
    f"{mse_base:>16.2f}"
    f"{rmse_base:>12.2f}"
    f"{mae_base:>12.2f}"
    f"{r2_base:>10.4f}"
)

print(
    f"{'Two-Feature Normal Equation':<30}"
    f"{w0_ne:>12.2f}"
    f"{w1_ne:>12.2f}"
    f"{w2_ne:>12.2f}"
    f"{mse_ne:>16.2f}"
    f"{rmse_ne:>12.2f}"
    f"{mae_ne:>12.2f}"
    f"{r2_ne:>10.4f}"
)

print(
    f"{'Two-Feature Gradient Descent':<30}"
    f"{w0_gd:>12.2f}"
    f"{w1_gd:>12.2f}"
    f"{w2_gd:>12.2f}"
    f"{mse_gd:>16.2f}"
    f"{rmse_gd:>12.2f}"
    f"{mae_gd:>12.2f}"
    f"{r2_gd:>10.4f}"
)

print("-" * 105)

# ---------------------------------------------------------
# Save comparison results to CSV
# ---------------------------------------------------------

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

RESULTS_PATH = REPORTS_DIR / "assignment_results.csv"

with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "w0",
        "w1_bmi",
        "w2_age",
        "MSE",
        "RMSE",
        "MAE",
        "R2",
    ])

    writer.writerow([
        "BMI-Only Baseline",
        w0_base,
        w1_base,
        "",
        mse_base,
        rmse_base,
        mae_base,
        r2_base,
    ])

    writer.writerow([
        "Two-Feature Normal Equation",
        w0_ne,
        w1_ne,
        w2_ne,
        mse_ne,
        rmse_ne,
        mae_ne,
        r2_ne,
    ])

    writer.writerow([
        "Two-Feature Gradient Descent",
        w0_gd,
        w1_gd,
        w2_gd,
        mse_gd,
        rmse_gd,
        mae_gd,
        r2_gd,
    ])

print()
print(f"Results saved to: {RESULTS_PATH}")
