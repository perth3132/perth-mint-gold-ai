import numpy as np
import traceback
import preprocessor as pp

class CompactDeepGoldPredictor:
    """Optimized Deep Neural Network running the Adam Optimizer via programmatic loops."""
    def __init__(self, layer_dims=None):
        if layer_dims is None:
            layer_dims = [20, 32, 16, 1]
        np.random.seed(101)
        self.weights = []
        self.biases = []
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []
        self.t = 0
        
        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2.0 / layer_dims[i])
            b = np.zeros((1, layer_dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

    def train_adam(self, X, y, epochs=12000, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
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
            for i in range(len(self.weights)):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                zs.append(z)
                a = z if i == len(self.weights) - 1 else np.maximum(0, z)
                activations.append(a)
                
            error = 2 * (activations[-1] - y_norm) / len(y_norm)
            for i in range(len(self.weights) - 1, -1, -1):
                dw = np.dot(activations[i].T, error)
                db = np.sum(error, axis=0, keepdims=True)
                if i > 0:
                    error = np.dot(error, self.weights[i].T) * (zs[i-1] > 0)
                    
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

    def predict(self, raw_input_seq):
        current_signal = (raw_input_seq - self.mx) / self.sx
        for i in range(len(self.weights)):
            z = np.dot(current_signal, self.weights[i]) + self.biases[i]
            current_signal = z if i == len(self.weights) - 1 else np.maximum(0, z)
        return (current_signal * self.sy + self.my).item()

def run_pipeline():
    try:
        # 1. BUG FIX: Pull data array once directly from the month API loader script
        all_prices = pp.download_and_clean_perth_mint_api()
        
        if all_prices is None or len(all_prices) < 10:
            print("[⚠️ FALLBACK] API data empty. Constructing baseline tracking tensor matrix sequence...")
            all_prices = np.array([6110.20, 6125.40, 6118.10, 6135.90, 6140.30, 6144.15, 6145.04])
            
        # Isolate the newest single raw calculation scalar mark from the tail index
        live_spot_price = float(all_prices[-1])
            
        # Standard 5-day sliding momentum window calculation configuration
        window_size = 5
        X_list, y_list = [], []
        for i in range(len(all_prices) - window_size):
            X_list.append(all_prices[i : i + window_size])
            y_list.append(all_prices[i + window_size])
            
        X_matrix = np.array(X_list)
        y_matrix = np.array(y_list).reshape(-1, 1)
        
        brain = CompactDeepGoldPredictor(layer_dims=[window_size, 32, 16, 1])
        brain.train_adam(X_matrix, y_matrix, epochs=12000, alpha=0.001)
        
        newest_live_window = np.array(all_prices[-window_size:]).reshape(1, -1)
        tomorrow_predicted_spot = brain.predict(newest_live_window)
        
        print("\n==================================================")
        print("    THE PERTH MINT API REAL-TIME GOLD FORECAST    ")
        print("==================================================")
        print(f"  Real-time Value (Today's Close): ${live_spot_price:.2f} AUD")
        print(f"  Tomorrow Price (Predicted Target):  ${tomorrow_predicted_spot:.2f} AUD")
        print("==================================================\n")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Core pipeline processing execution crash: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()
