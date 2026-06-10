import os
import re
import threading
from datetime import datetime
from kivy.app import App
from kivy.clock import mainthread

# Import your untouched, functional read-only dependencies natively
import preprocessor as pp
from gold_model import AdvancedDeepGoldPredictor
from gold_view import MobileUI

API_URL = "http://perthmint.com"
LOCAL_CSV_FILE = "perth_mint_api_gold.csv"

class GoldForecastingController(App):
    """THE CONTROLLER: Connects Model and View components cleanly via thread hooks."""
    def build(self):
        self.view = MobileUI(controller=self)
        return self.view

    def start_pipeline_execution(self, instance):
        self.view.btn.disabled = True
        self.view.result_label.text = "Processing neural networks..."
        self.view.trend_indicator.text = "CALCULATING FORCE VECTORS..."
        self.view.trend_indicator.color = (0.9, 0.6, 0, 1)
        threading.Thread(target=self.run_background_math).start()

    def run_background_math(self):
        try:
            # 1. Invoke your perfect working preprocessor directly to download data
            all_prices = pp.download_and_clean_perth_mint_api()
            if all_prices is None or len(all_prices) < 10:
                self.push_error_to_ui("Data mapping array extraction unreached."); return
                
            # Convert clean data items explicitly into standard float scalars
            prices_list = [float(val) for val in all_prices]
            live_spot = prices_list[-1]
            window_size = 5
            
            # 2. Extract corresponding date strings directly from the saved CSV to sync layout baselines
            dates_list = []
            if os.path.exists(LOCAL_CSV_FILE):
                with open(LOCAL_CSV_FILE, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.read().split('\n') if line.strip()][1:]
                    for line in lines:
                        clean_line = line.replace('"', '').strip()
                        if ',' in clean_line:
                            parts = clean_line.split(',')
                            # Isolate Column 0 (Date-Time string) cleanly
                            dates_list.append(parts[0].strip())
            
            # Fallback path if local file reading meets device lock restrictions
            if len(dates_list) < len(prices_list):
                dates_list = [f"Day-{i}" for i in range(len(prices_list))]

            # 3. Structure feature lookback matrix windows
            X_list, y_list = [], []
            for i in range(len(prices_list) - window_size):
                X_list.append(prices_list[i : i + window_size])
                # Keep target array flat to match your uncompacted gold_model expectations
                y_list.append(prices_list[i + window_size])
                
            # 4. Train your untouched gold_model core using the perfectly aligned data arrays
            brain = AdvancedDeepGoldPredictor(layer_dims=[window_size, 32, 16, 1])
            brain.train_adam(X_list, y_list, epochs=50, alpha=0.01)
            
            raw_prediction = brain.predict(prices_list[-window_size:])
            
            # Safely extract the raw scalar forecast out of its output structure
            if isinstance(raw_prediction, list):
                tomorrow_pred = float(raw_prediction[0])
            else:
                tomorrow_pred = float(raw_prediction)
            
            # 5. Export calculated metrics to the local text notepad log file database
            try:
                with open("gold_market_trend_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Close: ${live_spot:.2f} AUD | AI Predict: ${tomorrow_pred:.2f} AUD\n")
            except Exception: 
                pass
            
            # 6. Push final synchronized values to the screen layout interface
            # Slices trailing lookback elements to match 12-day window coordinates
            self.push_results_to_ui(live_spot, tomorrow_pred, prices_list[-12:], dates_list[-12])
            
        except Exception as e:
            self.push_error_to_ui(str(e))

    @mainthread
    def push_results_to_ui(self, today, tomorrow, historical_prices, start_date):
        self.view.result_label.text = f"Today's Close: ${today:.2f} AUD\nTomorrow Forecast: ${tomorrow:.2f} AUD"
        
        # Safely pass data metrics directly to your relative box layout chart view
        self.view.graph.update_graph_data(historical_prices, tomorrow, start_date)
        
        if tomorrow > today:
            self.view.trend_indicator.text = "SIGNAL: UP TREND ⬆ (BUY ALERT)"
            self.view.trend_indicator.color = (0, 0.9, 0.3, 1)
        else:
            self.view.trend_indicator.text = "SIGNAL: DOWN TREND ⬇ (SELL ALERT)"
            self.view.trend_indicator.color = (1, 0.2, 0.2, 1)
            
        self.view.btn.disabled = False

    @mainthread
    def push_error_to_ui(self, error_message):
        self.view.result_label.text = "Calculation failed."
        self.view.trend_indicator.text = f"ERROR: {error_message}"
        self.view.trend_indicator.color = (1, 0, 0, 1)
        self.view.btn.disabled = False

if __name__ == '__main__':
    GoldForecastingController().run()
