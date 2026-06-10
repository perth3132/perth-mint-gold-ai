import os
import re
import numpy as np
import requests

# PRESERVED PRECISE API ADDRESS - EXPLICITLY LOCKED DOWN
API_URL = "http://www.perthmint.com/api/exchangerate/metal/retail/priceHistoryCSV?currency=AUD&metal=Gold&timeSpanShort=true&range=days"
LOCAL_CSV_FILE = "perth_mint_api_gold.csv"

def download_and_clean_perth_mint_api():
    """
    Downloads raw text from the exact user-specified address, skips the top title row,
    and splits by commas to isolate Column 0 (Date & Time) and Column 1 (Price 1).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,application/csv,*/*"
    }
    
    try:
        # Download from the exact requested endpoint address
        response = requests.get(API_URL, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"[CRITICAL ERROR] API connection refused. Status Code: {response.status_code}")
            return None
            
        raw_content = response.text.strip()
        with open(LOCAL_CSV_FILE, 'w', encoding='utf-8') as f:
            f.write(raw_content)
            
        # Isolate rows cleanly by raw text string line breaks
        raw_lines = [line.strip() for line in raw_content.split('\n') if line.strip()]
        if len(raw_lines) < 2:
            print("[CRITICAL ERROR] Downloaded API payload contains insufficient rows.")
            return None
            
        # Bypass the very first line text title block description row
        data_lines = raw_lines[1:]
        clean_prices = []
        
        print("\n=== [DIAGNOSTIC DISPLAY] PRINTING ALL DATA READ FROM THE CSV ===")
        for idx, line in enumerate(data_lines):
            if ',' in line:
                # Split row natively by commas into a raw string array list
                parts = line.split(',')
                
                # Column 0 is the Date-Time string block
                raw_datetime = parts[0].strip()
                
                # Column 1 is Price 1 (The targeted gold spot base rate value)
                raw_price = parts[1].strip() if len(parts) > 1 else ""
                
                # Sanitize currency symbols and commas to isolate raw float scalars
                sanitized_price = re.sub(r'[^\d.]', '', raw_price)
                
                # PRINT OUT EVERY INDIVIDUAL TIMESTAMP AND PRICE TO YOUR SCREEN
                print(f"Row {idx+1:02d} | Date-Time: {raw_datetime} | Price 1 (Isolated): ${sanitized_price}")
                
                if sanitized_price:
                    clean_prices.append(float(sanitized_price))
            else:
                print(f"Row {idx+1:02d} | [UNABLE TO PARSE ROW]: {line}")
        print("===================================================================\n")
        
        if len(clean_prices) == 0:
            return None
            
        return np.array(clean_prices)
    except Exception as e:
        print(f"[CRITICAL ERROR] Preprocessor structural layout crash: {e}")
        return None
