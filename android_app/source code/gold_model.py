import math
import random

class AdvancedDeepGoldPredictor:
    """THE MODEL: Manages data standardization, matrix parameters, and neural training."""
    def __init__(self, layer_dims):
        random.seed(42)
        self.layer_dims = layer_dims
        self.weights = []
        self.biases = []
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []
        self.t = 0
        
        for i in range(len(layer_dims) - 1):
            in_d, out_d = layer_dims[i], layer_dims[i+1]
            scale = math.sqrt(2.0 / in_d)
            self.weights.append([[random.gauss(0, 1) * scale for _ in range(out_d)] for _ in range(in_d)])
            self.biases.append([0.0 for _ in range(out_d)])
            self.m_w.append([[0.0 for _ in range(out_d)] for _ in range(in_d)])
            self.v_w.append([[0.0 for _ in range(out_d)] for _ in range(in_d)])
            self.m_b.append([0.0 for _ in range(out_d)])
            self.v_b.append([0.0 for _ in range(out_d)])

    def train_adam(self, X, y, epochs=500, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        N = len(X)
        if N == 0: return
        feats = len(X[0])
        
        self.mx = [sum(X[r][c] for r in range(N)) / N for c in range(feats)]
        self.sx = []
        for c in range(feats):
            var = sum((X[r][c] - self.mx[c])**2 for r in range(N)) / N
            self.sx.append(math.sqrt(var) if var > 0 else 1.0)
            
        y_flat = [item if not isinstance(item, list) else item[0] for item in y]
        self.my = sum(y_flat) / N
        y_var = sum((val - self.my)**2 for val in y_flat) / N
        self.sy = math.sqrt(y_var) if y_var > 0 else 1.0
        
        X_norm = [[(X[r][c] - self.mx[c]) / self.sx[c] for c in range(feats)] for r in range(N)]
        y_norm = [(y_flat[r] - self.my) / self.sy for r in range(N)]
        
        for _ in range(epochs + 1):
            self.t += 1
            for r in range(N):
                activations = [X_norm[r]]
                zs = []
                for i in range(len(self.layer_dims) - 1):
                    w, b = self.weights[i], self.biases[i]
                    z_row = [sum(activations[-1][in_c] * w[in_c][out_c] for in_c in range(len(w))) + b[out_c] for out_c in range(len(w[0]))]
                    zs.append(z_row)
                    activations.append(z_row if i == len(self.weights) - 1 else [max(0.0, v) for v in z_row])
                
                error = [2.0 * (activations[-1][c] - y_norm[r]) / N for c in range(len(activations[-1]))]
                for i in range(len(self.layer_dims) - 2, -1, -1):
                    w = self.weights[i]
                    act = activations[i]
                    dw = [[act[in_c] * error[out_c] for out_c in range(len(w[0]))] for in_c in range(len(w))]
                    db = [error[out_c] for out_c in range(len(w[0]))]
                    
                    if i > 0:
                        next_error = [0.0 for _ in range(len(self.weights[i-1][0]))]
                        for in_c in range(len(w)):
                            if zs[i-1][in_c] > 0:
                                next_error[in_c] = sum(error[out_c] * w[in_c][out_c] for out_c in range(len(w[0])))
                        error = next_error
                        
                    for in_c in range(len(w)):
                        for out_c in range(len(w[0])):
                            self.m_w[i][in_c][out_c] = beta1 * self.m_w[i][in_c][out_c] + (1 - beta1) * dw[in_c][out_c]
                            self.v_w[i][in_c][out_c] = beta2 * self.v_w[i][in_c][out_c] + (1 - beta2) * (dw[in_c][out_c]**2)
                            self.weights[i][in_c][out_c] -= alpha * (self.m_w[i][in_c][out_c] / (1.0 - beta1**self.t)) / (math.sqrt(self.v_w[i][in_c][out_c] / (1.0 - beta2**self.t)) + eps)
                            
                    for out_c in range(len(w[0])):
                        self.m_b[i][out_c] = beta1 * self.m_b[i][out_c] + (1 - beta1) * db[out_c]
                        self.v_b[i][out_c] = beta2 * self.v_b[i][out_c] + (1 - beta2) * (db[out_c]**2)
                        self.biases[i][out_c] -= alpha * (self.m_b[i][out_c] / (1.0 - beta1**self.t)) / (math.sqrt(self.v_b[i][out_c] / (1.0 - beta2**self.t)) + eps)

    def predict(self, raw_input_seq):
        current_signal = [(raw_input_seq[c] - self.mx[c]) / self.sx[c] for c in range(len(raw_input_seq))]
        for i in range(len(self.layer_dims) - 1):
            w, b = self.weights[i], self.biases[i]
            current_signal = [sum(current_signal[in_c] * w[in_c][out_c] for in_c in range(len(w))) + b[out_c] for out_c in range(len(w[0]))]
            if i < len(self.layer_dims) - 2:
                current_signal = [max(0.0, v) for v in current_signal]
        return current_signal[0] * self.sy + self.my
