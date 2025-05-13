import websocket
import json
import numpy as np
import os
import time
import threading
import csv
from datetime import datetime, timedelta

SYMBOL = "btcusdt"
STOCK_NAME = "BINANCE"

# 날짜 범위 설정
START_DATE = "2025-05-01"
END_DATE_STR = "2025-05-10"
BASE_DATE = datetime.strptime(START_DATE, "%Y-%m-%d")
END_DATE = datetime.strptime(END_DATE_STR, "%Y-%m-%d")

LOBSTER_DIR = f"data/{STOCK_NAME}/{STOCK_NAME}_{START_DATE}_{END_DATE_STR}"
os.makedirs(LOBSTER_DIR, exist_ok=True)

N_TICKS_PER_DAY = 1000
MAX_MESSAGES_PER_DAY = 1000  # ✅ 메시지 제한 개수 설정
INTERVAL = 0.1

orderbook_seq = []
agg_message_rows = []
saved_days = 0
trade_id = 0  # 가짜 order_id

current_date = BASE_DATE
lock = threading.Lock()

def on_orderbook_message(ws, message):
    global orderbook_seq, saved_days, current_date

    data = json.loads(message)
    bids = data.get("bids", [])[:10]
    asks = data.get("asks", [])[:10]
    if len(bids) < 10 or len(asks) < 10:
        return
    # 1️⃣ 현재 시간 기록 (datetime 객체 → 문자열)
    now = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")  # 예: '2023-01-09 10:17:40 PM'
    snapshot = [now]
    for i in range(10):
        snapshot += [
            float(bids[i][0]), float(bids[i][1]),
            float(asks[i][0]), float(asks[i][1])
        ]

    with lock:
        orderbook_seq.append(snapshot)

    if len(orderbook_seq) >= N_TICKS_PER_DAY:
        with lock:
            date_str = current_date.strftime("%Y-%m-%d")
            base_name = f"{date_str}_34200000_57600000"
            ob_path = os.path.join(LOBSTER_DIR, f"BINANCE_{base_name}_orderbook_10.csv")
            msg_path = os.path.join(LOBSTER_DIR, f"BINANCE_{base_name}_message.csv")

            # Orderbook 저장
            np.savetxt(ob_path, np.array(orderbook_seq, dtype=object), delimiter=",", fmt='%s')

            # Message 저장
            with open(msg_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "event_type", "order_id", "size", "price", "direction"])
                writer.writerows(agg_message_rows)

            print(f"[Day {saved_days+1}] Saved: {ob_path}, {msg_path}")

            saved_days += 1
            current_date = BASE_DATE + timedelta(days=saved_days)
            orderbook_seq.clear()
            agg_message_rows.clear()

            if current_date > END_DATE:
                print("✅ 설정한 END_DATE까지 수집 완료. 종료합니다.")
                os._exit(0)

def on_aggtrade_message(ws, message):
    global trade_id

    # ✅ 메시지 개수 제한
    if len(agg_message_rows) >= MAX_MESSAGES_PER_DAY:
        return

    data = json.loads(message)
    trade_time = int(data["T"] // 1000) - 34200  # seconds, 보정
    price = float(data["p"])
    size = float(data["q"])
    is_maker = data["m"]
    direction = 0 if is_maker else 1

    row = [trade_time, 4, trade_id, size, price, direction]
    trade_id += 1

    with lock:
        agg_message_rows.append(row)

def start_orderbook_ws():
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth10@100ms",
        on_message=on_orderbook_message,
        on_open=lambda ws: print("✅ Orderbook WebSocket connected."),
        on_close=lambda ws, code, msg: print("🛑 Orderbook WebSocket closed.")
    )
    ws.run_forever()

def start_aggtrade_ws():
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade",
        on_message=on_aggtrade_message,
        on_open=lambda ws: print("✅ AggTrade WebSocket connected."),
        on_close=lambda ws, code, msg: print("🛑 AggTrade WebSocket closed.")
    )
    ws.run_forever()

if __name__ == "__main__":
    t1 = threading.Thread(target=start_orderbook_ws)
    t2 = threading.Thread(target=start_aggtrade_ws)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
