import websocket
import json
import numpy as np
import os
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

# 저장 디렉토리
LOBSTER_DIR = f"data/{STOCK_NAME}/{STOCK_NAME}_{START_DATE}_{END_DATE_STR}"
os.makedirs(LOBSTER_DIR, exist_ok=True)

# 메시지 및 오더북 개수 제한
MAX_MESSAGES_PER_DAY = 1000

# 동기 제어 및 상태 변수
lock = threading.Lock()
aligned_orderbook_rows = []
agg_message_rows = []
trade_id = 0
current_date = BASE_DATE
latest_snapshot = None

# 오더북 메시지 수신 시 최신 스냅샷 갱신
def on_orderbook_message(ws, message):
    global latest_snapshot
    data = json.loads(message)
    bids = data.get("bids", [])[:10]
    asks = data.get("asks", [])[:10]
    if len(bids) < 10 or len(asks) < 10:
        return
    # 숫자 데이터만 저장
    snapshot = []
    for i in range(10):
        snapshot.extend([
            float(bids[i][0]), float(bids[i][1]),
            float(asks[i][0]), float(asks[i][1])
        ])
    with lock:
        latest_snapshot = snapshot

# AggTrade 메시지 수신 시 저장 및 플러시 처리
def on_aggtrade_message(ws, message):
    global trade_id, current_date, latest_snapshot
    data = json.loads(message)
    # 메시지 개수 제한
    with lock:
        if len(agg_message_rows) >= MAX_MESSAGES_PER_DAY:
            return
    # 메시지 데이터 구성
    trade_time = int(data["T"] // 1000) - 34200
    price = float(data["p"])
    size = float(data["q"])
    direction = 0 if data.get("m", False) else 1
    row = [trade_time, 4, trade_id, size, price, direction]
    with lock:
        agg_message_rows.append(row)
        # 최신 스냅샷 매칭
        if latest_snapshot is not None:
            aligned_orderbook_rows.append(latest_snapshot.copy())
        trade_id += 1

    # 일일 한도 초과 시 저장 및 초기화
    with lock:
        if len(agg_message_rows) >= MAX_MESSAGES_PER_DAY:
            date_str = current_date.strftime("%Y-%m-%d")
            base_name = f"{date_str}_34200000_57600000"
            msg_path = os.path.join(LOBSTER_DIR, f"{STOCK_NAME}_{base_name}_message.csv")
            ob_path  = os.path.join(LOBSTER_DIR, f"{STOCK_NAME}_{base_name}_orderbook.csv")

            # 파일 쓰기 (헤더 없이 데이터만)
            with open(msg_path, "w", newline="") as mf:
                writer = csv.writer(mf)
                writer.writerows(agg_message_rows)
            with open(ob_path, "w", newline="") as of:
                writer = csv.writer(of)
                writer.writerows(aligned_orderbook_rows)

            print(f"[Day {current_date.strftime('%Y-%m-%d')}] Saved: {msg_path}, {ob_path}")

            # 초기화 및 날짜 갱신
            agg_message_rows.clear()
            aligned_orderbook_rows.clear()
            trade_id = 0
            current_date += timedelta(days=1)
            if current_date > END_DATE:
                print("✅ 설정한 END_DATE까지 수집 완료. 종료합니다.")
                os._exit(0)

# WebSocket 스레드 시작
def start_orderbook_ws():
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth10@100ms",
        on_message=on_orderbook_message,
        on_open=lambda ws: print("✅ Orderbook WS connected."),
        on_close=lambda ws, code, msg: print("🛑 Orderbook WS closed.")
    )
    ws.run_forever()


def start_aggtrade_ws():
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade",
        on_message=on_aggtrade_message,
        on_open=lambda ws: print("✅ AggTrade WS connected."),
        on_close=lambda ws, code, msg: print("🛑 AggTrade WS closed.")
    )
    ws.run_forever()

if __name__ == "__main__":
    t1 = threading.Thread(target=start_orderbook_ws)
    t2 = threading.Thread(target=start_aggtrade_ws)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
