import os
import pandas as pd
import numpy as np
import torch
import constants as cst
from constants import SamplingType
from utils.utils_data import z_score_orderbook, labeling


class BinanceDataBuilder:
    def __init__(
        self,
        data_dir,
        date_trading_days,
        split_rates,
        sampling_type,
        sampling_time,
        sampling_quantity,
    ):
        self.n_lob_levels = cst.N_LOB_LEVELS
        self.data_dir = data_dir
        self.date_trading_days = date_trading_days
        self.split_rates = split_rates

        self.sampling_type = sampling_type
        self.sampling_time = sampling_time
        self.sampling_quantity = sampling_quantity

    def prepare_save_datasets(self):
        # 기존 저장된 Binance 데이터 경로로 이동
        input_path = os.path.join(
            self.data_dir,
            "binance",
            f"binance_{self.date_trading_days[0]}_{self.date_trading_days[1]}"
        )
        assert os.path.exists(input_path), f"입력 데이터 경로가 존재하지 않습니다: {input_path}"

        save_dir = os.path.join(self.data_dir, "BINANCE")
        os.makedirs(save_dir, exist_ok=True)

        self.dataframes = []
        self._prepare_dataframes(input_path)

        path_where_to_save = save_dir
        train_input = self.dataframes[0].values
        val_input = self.dataframes[1].values
        test_input = self.dataframes[2].values
        self.train_set = np.concatenate([train_input, self.train_labels_horizons.values], axis=1)
        self.val_set = np.concatenate([val_input, self.val_labels_horizons.values], axis=1)
        self.test_set = np.concatenate([test_input, self.test_labels_horizons.values], axis=1)
        self._save(path_where_to_save)

    def _prepare_dataframes(self, path):
        COLUMNS_NAMES = {"orderbook": ["timestamp",
                                        "sell1", "vsell1", "buy1", "vbuy1",
                                        "sell2", "vsell2", "buy2", "vbuy2",
                                        "sell3", "vsell3", "buy3", "vbuy3",
                                        "sell4", "vsell4", "buy4", "vbuy4",
                                        "sell5", "vsell5", "buy5", "vbuy5",
                                        "sell6", "vsell6", "buy6", "vbuy6",
                                        "sell7", "vsell7", "buy7", "vbuy7",
                                        "sell8", "vsell8", "buy8", "vbuy8",
                                        "sell9", "vsell9", "buy9", "vbuy9",
                                        "sell10", "vsell10", "buy10", "vbuy10"]}
        self.num_trading_days = len(os.listdir(path))
        split_days = self._split_days()
        self._create_dataframes_splitted(path, split_days, COLUMNS_NAMES)

        train_input = self.dataframes[0].values
        val_input = self.dataframes[1].values
        test_input = self.dataframes[2].values

        for i in range(len(cst.LOBSTER_HORIZONS)):
            train_labels = labeling(train_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i])
            val_labels = labeling(val_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i])
            test_labels = labeling(test_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i])

            pad_train = np.full((train_input.shape[0] - train_labels.shape[0],), np.inf)
            pad_val = np.full((val_input.shape[0] - val_labels.shape[0],), np.inf)
            pad_test = np.full((test_input.shape[0] - test_labels.shape[0],), np.inf)

            if i == 0:
                self.train_labels_horizons = pd.DataFrame(np.concatenate([train_labels, pad_train])[:, None], columns=[f"label_h{cst.LOBSTER_HORIZONS[i]}"])
                self.val_labels_horizons = pd.DataFrame(np.concatenate([val_labels, pad_val])[:, None], columns=[f"label_h{cst.LOBSTER_HORIZONS[i]}"])
                self.test_labels_horizons = pd.DataFrame(np.concatenate([test_labels, pad_test])[:, None], columns=[f"label_h{cst.LOBSTER_HORIZONS[i]}"])
            else:
                self.train_labels_horizons[f"label_h{cst.LOBSTER_HORIZONS[i]}"] = np.concatenate([train_labels, pad_train])
                self.val_labels_horizons[f"label_h{cst.LOBSTER_HORIZONS[i]}"] = np.concatenate([val_labels, pad_val])
                self.test_labels_horizons[f"label_h{cst.LOBSTER_HORIZONS[i]}"] = np.concatenate([test_labels, pad_test])

        self._normalize_dataframes()

    def _create_dataframes_splitted(self, path, split_days, COLUMNS_NAMES):
        train_orderbooks, val_orderbooks, test_orderbooks = None, None, None

        # 1️⃣ orderbook 파일만 필터링
        orderbook_files = sorted([
            f for f in os.listdir(path)
            if f.endswith("_orderbook_10.csv")
        ])
        
        for i, filename in enumerate(orderbook_files):
            f = os.path.join(path, filename)

            df_ob = pd.read_csv(
                f,
                names=COLUMNS_NAMES["orderbook"],
                dtype={col: float for col in COLUMNS_NAMES["orderbook"] if col != "timestamp"},
                parse_dates=["timestamp"],  # 선택사항
                on_bad_lines="skip"  # 혹시 잘못된 줄 방지
                )
            
            # 2️⃣ Optional: 시간 기반 샘플링
            if self.sampling_type == SamplingType.TIME:
                df_ob = self._sampling_time(df_ob, self.sampling_time)

            # 3️⃣ 데이터 분할
            if i < split_days[0]:
                    train_orderbooks = df_ob if train_orderbooks is None else pd.concat([train_orderbooks, df_ob], axis=0)
            elif split_days[0] <= i < split_days[1]:
                val_orderbooks = df_ob if val_orderbooks is None else pd.concat([val_orderbooks, df_ob], axis=0)
            else:
                test_orderbooks = df_ob if test_orderbooks is None else pd.concat([test_orderbooks, df_ob], axis=0)
            
        # 4️⃣ timestamp 열 제거 후 저장
        self.dataframes = [
            train_orderbooks.drop(columns=["timestamp"]),
            val_orderbooks.drop(columns=["timestamp"]),
            test_orderbooks.drop(columns=["timestamp"]),
        ]

    def _normalize_dataframes(self):
        for i in range(len(self.dataframes)):
            if i == 0:
                self.dataframes[i], mean_size, mean_price, std_size, std_price = z_score_orderbook(self.dataframes[i])
            else:
                self.dataframes[i], _, _, _, _ = z_score_orderbook(self.dataframes[i], mean_size, mean_price, std_size, std_price)

    def _save(self, path):
        np.save(os.path.join(path, "train.npy"), self.train_set)
        np.save(os.path.join(path, "val.npy"), self.val_set)
        np.save(os.path.join(path, "test.npy"), self.test_set)

    def _split_days(self):
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        print(f"There are {train} days for training, {val - train} days for validation and {test - val} days for testing")
        return [train, val, test]

    def _sampling_time(self, dataframe, time):
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], errors='coerce')
        dataframe = dataframe.set_index('timestamp').resample(time).first().dropna().reset_index()
        return dataframe
