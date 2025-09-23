# -*- coding: utf-8 -*-
import os
import time
import tqdm

for lag in [0, 3]:
    for nn_on in [0, 1]:
        for nn_type in (["FCNN", "GRU", "RNN", "LSTM"] if nn_on else ["FCNN"]):
            for elbo_weight in [0, 1]:
                for pinn_weight in [0, 0.5]:
                    for year in ["all"]:
                        print(f"py fast_train.py --NN_TYPE {nn_type} --PINN_WEIGHT {pinn_weight} --ELBO_WEIGHT {elbo_weight} --Lag {lag} --Year {year} --NN_ON {nn_on}")
                        os.system(
                            f"py fast_train.py --NN_TYPE {nn_type} --PINN_WEIGHT {pinn_weight} --ELBO_WEIGHT {elbo_weight} --Lag {lag} --Year {year} --NN_ON {nn_on}"
                        )
                        for _ in tqdm.tqdm(range(10)):
                            time.sleep(1)
