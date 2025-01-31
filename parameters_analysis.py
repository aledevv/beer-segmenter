import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Leggi i dati
data = pd.read_csv('saved_parameters.txt', header=None,
                   names=['frame', 'radius', 'center_x', 'center_y', 'min_len'])

# Crea array X per la regressione
X = np.arange(len(data)).reshape(-1, 1)

def fit_equations(x, y, param_name):
    # Regressione lineare
    slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)
    linear_eq = f"y = {slope:.4f}x + {intercept:.4f}"
    linear_r2 = r2_score(y, slope * x.flatten() + intercept)
    
    # Regressione polinomiale (grado 2)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(x)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    poly_eq = f"y = {poly_reg.coef_[2]:.4f}x² + {poly_reg.coef_[1]:.4f}x + {poly_reg.intercept_:.4f}"
    poly_r2 = r2_score(y, poly_reg.predict(X_poly))
    
    # Plot dei dati e delle regressioni
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, color='blue', alpha=0.5, label='Dati reali')
    
    # Plot regressione lineare
    plt.plot(x, slope * x.flatten() + intercept, 'r-', 
             label=f'Lineare (R² = {linear_r2:.4f})')
    
    # Plot regressione polinomiale
    plt.plot(x, poly_reg.predict(X_poly), 'g-', 
             label=f'Polinomiale (R² = {poly_r2:.4f})')
    
    plt.title(f'Regressione per {param_name}')
    plt.xlabel('Frame number')
    plt.ylabel(param_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{param_name.lower()}_regression.png')
    plt.close()
    
    return {
        'linear': {'equation': linear_eq, 'r2': linear_r2},
        'polynomial': {'equation': poly_eq, 'r2': poly_r2}
    }

# Analizza ogni parametro
parameters = {
    'Radius': data['radius'],
    'Center_X': data['center_x'],
    'Center_Y': data['center_y']
}

results = {}
for param_name, values in parameters.items():
    results[param_name] = fit_equations(X, values, param_name)
    
    print(f"\nEquazioni per {param_name}:")
    print("Lineare:")
    print(f"  Equazione: {results[param_name]['linear']['equation']}")
    print(f"  R²: {results[param_name]['linear']['r2']:.4f}")
    print("Polinomiale:")
    print(f"  Equazione: {results[param_name]['polynomial']['equation']}")
    print(f"  R²: {results[param_name]['polynomial']['r2']:.4f}")

# Analisi delle variazioni per segmenti
def analyze_segments(data, param_name, values):
    # Trova i punti dove la pendenza cambia significativamente
    diff = np.diff(values)
    mean_diff = np.mean(np.abs(diff))
    std_diff = np.std(np.abs(diff))
    
    change_points = [0]
    for i in range(1, len(diff)):
        if abs(diff[i] - diff[i-1]) > 2 * std_diff:
            change_points.append(i)
    change_points.append(len(values)-1)
    
    print(f"\nAnalisi segmenti per {param_name}:")
    for i in range(len(change_points)-1):
        start = change_points[i]
        end = change_points[i+1]
        segment = values[start:end+1]
        
        if len(segment) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(segment)), segment)
            print(f"\nSegmento {i+1} (frames {start}-{end}):")
            print(f"Equazione: y = {slope:.4f}x + {intercept:.4f}")
            print(f"R²: {r_value**2:.4f}")

# Analizza i segmenti per ogni parametro
for param_name, values in parameters.items():
    analyze_segments(data, param_name, values)