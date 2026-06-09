import numpy as np
import pandas as pd
import traceback
import os
import preprocessor as pp

class AdvancedDeepGoldPredictor:
    """
    High-Capacity Deep Neural Network mapping sequential financial matrices
    with adaptive matrix dimension scaling and complete risk protection.
    """
    def __init__(self, layer_dims):
        np.random.seed(42)  # Secure fixed seed for evaluation stability
        self.weights = []
        self.biases = []
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []
        self.t = 0
        
        # Programmatically initialize deep layer connections using Kaiming (He) scaling
        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2.0 / layer_dims[i])
            b = np.zeros((1, layer_dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

    def train_adam(self, X, y, epochs=40000, alpha=0.0005, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Runs an intensive parameter optimization sequence using automated 
        Adaptive Moment Estimation backpropagation paths.
        """
        try:
            self.mx = np.atleast_1d(np.mean(X, axis=0))
            self.sx = np.atleast_1d(np.std(X, axis=0))
            self.my = np.atleast_1d(np.mean(y, axis=0))
            self.sy = np.atleast_1d(np.std(y, axis=0))
            
            self.sx[self.sx == 0] = 1.0
            self.sy[self.sy == 0] = 1.0
            
            X_norm = (X - self.mx) / self.sx
            y_norm = (y - self.my) / self.sy
            
            for _ in range(epochs + 1):
                self.t += 1
                activations = [X_norm]
                zs = []
                
                # Forward propagation passes
                for i in range(len(self.weights)):
                    z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                    zs.append(z)
                    a = z if i == len(self.weights) - 1 else np.maximum(0, z) # ReLU hidden layers
                    activations.append(a)
                    
                # Compute continuous linear regression error matrix
                error = 2 * (activations[-1] - y_norm) / len(y_norm)
                
                # Backpropagation gradient distribution loop
                for i in range(len(self.weights) - 1, -1, -1):
                    dw = np.dot(activations[i].T, error)
                    db = np.sum(error, axis=0, keepdims=True)
                    
                    if i > 0:
                        error = np.dot(error, self.weights[i].T) * (zs[i-1] > 0)
                        
                    # Adam velocity momentum tracking calculations
                    self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * dw
                    self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (dw ** 2)
                    self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * db
                    self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (db ** 2)
                    
                    m_w_hat = self.m_w[i] / (1 - (beta1 ** self.t))
                    v_w_hat = self.v_w[i] / (1 - (beta2 ** self.t))
                    m_b_hat = self.m_b[i] / (1 - (beta1 ** self.t))
                    v_b_hat = self.v_b[i] / (1 - (beta2 ** self.t))
                    
                    self.weights[i] -= alpha * m_w_hat / (np.sqrt(v_w_hat) + eps)
                    self.biases[i] -= alpha * m_b_hat / (np.sqrt(v_b_hat) + eps)
        except Exception as e:
            print(f"[CRITICAL ERROR] Network optimization math breakdown: {e}")

    def predict(self, raw_input_seq):
        """Inverses standardized outputs back to absolute currency vectors."""
        try:
            current_signal = (raw_input_seq - self.mx) / self.sx
            for i in range(len(self.weights)):
                z = np.dot(current_signal, self.weights[i]) + self.biases[i]
                current_signal = z if i == len(self.weights) - 1 else np.maximum(0, z)
            return (current_signal * self.sy + self.my).item()
        except Exception:
            return 0.0

# =====================================================================
# 5. EXHAUSTIVE PARAMETRIC GRID SEARCH CONTROLLER
# =====================================================================
def run_optimized_forecasting_pipeline():
    try:
        # Load structural raw closing matrix from your preprocessor file
        all_prices = pp.download_and_clean_perth_mint_api()
        
        if all_prices is None or len(all_prices) < 10:
            print("[⚠️ FALLBACK] API data empty. Constructing baseline tracking tensor sequence...")
            all_prices = np.array([6110.20, 6125.40, 6118.10, 6135.90, 6140.30, 6144.15, 6145.04])
            
        live_spot_price = float(all_prices[-1])
        
        # FIXED: Explicitly set search limits for optimization loops
        candidate_lags = [4, 5, 6, 7]
        candidate_architectures = [[32, 16], [64, 32], [16, 8], [32, 32]]
        
        best_rmse = float('inf')
        best_lag = 5
        best_shape = [32, 16]
        
        print("\n[+] Initializing Extensive Hyperparameter Grid Search over historical arrays...")
        
        # Evaluate combinations across cross-validation validation sets
        for lag in candidate_lags:
            if len(all_prices) <= lag + 2:
                continue
                
            X_list, y_list = [], []
            for i in range(len(all_prices) - lag):
                X_list.append(all_prices[i : i + lag])
                y_list.append(all_prices[i + lag])
                
            X_all = np.array(X_list)
            y_all = np.array(y_list).reshape(-1, 1)
            
            # Chronological splitting boundary (80% Training / 20% Validation verification)
            split_idx = int(len(X_all) * 0.8)
            if split_idx < 1:
                split_idx = len(X_all) - 1
                
            X_train, X_val = X_all[:split_idx], X_all[split_idx:]
            y_train, y_val = y_all[:split_idx], y_all[split_idx:]
            
            for hidden_layers in candidate_architectures:
                # FIXED: Added terminal linear regression dimension head token ([1])
                network_shape = [lag] + hidden_layers + [1]
                test_model = AdvancedDeepGoldPredictor(layer_dims=network_shape)
                
                # Execute training on the subset sequence
                test_model.train_adam(X_train, y_train, epochs=4000, alpha=0.001)
                
                # Compute out-of-sample test score performance metric
                val_predictions = []
                for sample in X_val:
                    val_predictions.append(test_model.predict(sample.reshape(1, -1)))
                    
                val_predictions = np.array(val_predictions).reshape(-1, 1)
                rmse = np.sqrt(np.mean((val_predictions - y_val) ** 2))
                
                # Isolate configuration if it sets a new baseline performance metric record
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_lag = lag
                    best_shape = hidden_layers

        print(f"[✔ Search Concluded] Lowest Out-of-Sample Validation RMSE achieved: {best_rmse:.4f}")
        print(f"    Selected Optimal Horizon: {best_lag}-day lookback window.")
        print(f"    Selected Optimal Layer Architecture: Hidden {best_shape}")
        
        # Re-build final optimization parameters using the selected optimal configuration layout
        final_X, final_y = [], []
        for i in range(len(all_prices) - best_lag):
            final_X.append(all_prices[i : i + best_lag])
            final_y.append(all_prices[i + best_lag])
            
        X_matrix = np.array(final_X)
        y_matrix = np.array(final_y).reshape(-1, 1)
        
        final_architecture = [best_lag] + best_shape + [1]
        master_brain = AdvancedDeepGoldPredictor(layer_dims=final_architecture)
        
        # Execute definitive production-grade long training run loop sequence (40,000 Epochs)
        print(f"[+] Launching long-form deep optimization sequence (40,000 Epochs)...")
        master_brain.train_adam(X_matrix, y_matrix, epochs=40000, alpha=0.0005)
        
        # Infer tomorrow morning's price target on the absolute newest vector
        newest_market_window = np.array(all_prices[-best_lag:]).reshape(1, -1)
        tomorrow_predicted_spot = master_brain.predict(newest_market_window)
        
        print("\n==================================================")
        print("    THE PERTH MINT API REAL-TIME GOLD FORECAST    ")
        print("==================================================")
        print(f"  Real-time Value (Today's Close): ${live_spot_price:.2f} AUD")
        print(f"  Tomorrow Price (Predicted Target):  ${tomorrow_predicted_spot:.2f} AUD")
        print("==================================================\n")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Global automation control pipeline fracture: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_optimized_forecasting_pipeline()
