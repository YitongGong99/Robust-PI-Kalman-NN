# -*- coding: utf-8 -*-
from model import Exo_Var_NN, EKF, dataset_pt, Variational_ParamBank, evaluation_plot
import torch
import tqdm
import itertools
import datetime
import argparse
import os
import time
import gc

torch.manual_seed(42)

def main(args: argparse.Namespace):
    Lag = args.Lag    
    PINN_WEIGHT = args.PINN_WEIGHT
    NLL_WEIGHT = args.NLL_WEIGHT
    ELBO_WEIGHT = args.ELBO_WEIGHT
    VI = True if ELBO_WEIGHT else False
    K_MC = 1 if VI else 1
    LR = 1e-3 if VI else 1e-1
    NN_TYPE = args.NN_TYPE
    NN_ON = args.NN_ON

    Range = args.Year
    if Range == "all":
        DataSet = dataset_pt(
            # start=datetime.datetime(YEAR, 1, 1),
            # train_split=datetime.datetime(YEAR, 8, 31),
            # valid_split=datetime.datetime(YEAR, 10, 31),
            # end=datetime.datetime(YEAR, 12, 31),
            fls_lag=Lag,
            order_diff=1,
            pca_components=10,
        )
    else:
        YEAR = int(Range)
        DataSet = dataset_pt(
            start=datetime.datetime(YEAR, 1, 1),
            train_split=datetime.datetime(YEAR+2, 12, 31),
            valid_split=datetime.datetime(YEAR+3, 12, 31),
            end=datetime.datetime(YEAR+4, 12, 31),
            fls_lag=Lag,
            order_diff=1,
            pca_components=10,
        )

    PATH = os.path.join(
        "experiment_output",
        f"{NN_TYPE.upper() if NN_ON else 'Baseline'}_"
        f"{DataSet.start.strftime('%Y%m%d')}-{DataSet.end.strftime('%Y%m%d')}_"
        f"Lag-{Lag}_"
        f"VI-{'T' if VI or ELBO_WEIGHT else 'F'}_"
        f"PINN-{'T' if PINN_WEIGHT else 'F'}_"
        f"MSE-{'T' if NLL_WEIGHT else 'F'}"
    )
    os.makedirs(PATH, exist_ok=True)

    DataLoader = torch.utils.data.DataLoader(DataSet, batch_size=1, shuffle=False)

    Prior = (torch.Tensor([2 * torch.log(torch.tensor(DataSet.data[("OBSERVATIONS", "HV60")][0])), 0, 0]), torch.eye(3))
    GARCH_Net = Exo_Var_NN(nn_type=NN_TYPE, target="GARCH")
    CARMA_Net = Exo_Var_NN(nn_type=NN_TYPE, target="CARMA")
    EKF_Net = EKF(*Prior)

    VP_GARCH = Variational_ParamBank(GARCH_Net)
    VP_CARMA = Variational_ParamBank(CARMA_Net)
    VP_EKF = Variational_ParamBank(EKF_Net)

    Optimizer = torch.optim.Adam(
        itertools.chain(VP_EKF.parameters(), VP_GARCH.parameters(), VP_CARMA.parameters()),
        lr=LR
    )
    Lr_Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=10, gamma=0.9)

    Epochs = 80

    EKF_Net.train()
    GARCH_Net.train()
    CARMA_Net.train()
    VP_EKF.train()
    VP_GARCH.train()
    VP_CARMA.train()

    Best_Loss = torch.inf
    Best_Valid_Loss = torch.inf
    patience_max = 10
    patience = 0
    for epoch in range(Epochs):
        train_loss = .0
        valid_loss = .0
        test_loss = .0
        Optimizer.zero_grad()
        for k_mc in range(K_MC):
            EKF_sample = VP_EKF.sample(vi=VI)
            CARMA_sample = VP_CARMA.sample(vi=VI)
            GARCH_sample = VP_GARCH.sample(vi=VI)

            GARCH_Impacts = torch.func.functional_call(GARCH_Net, GARCH_sample, args=(),
                                                       kwargs={"x": DataSet.exo_var}) * NN_ON
            CARMA_Impacts = torch.func.functional_call(CARMA_Net, CARMA_sample, args=(),
                                                       kwargs={"x": DataSet.exo_var}) * NN_ON
            EKF_Net.clear_history(*Prior)
            pbar = tqdm.tqdm(DataLoader, desc=f"Epoch {epoch + 1}/{Epochs}, MC: {k_mc + 1}/{K_MC}")
            for batch in pbar:
                idx = batch["order"]
                torch.func.functional_call(
                    EKF_Net, EKF_sample, args=(), kwargs={
                        "observation": batch["log_return"][0],
                        "multistep_truth": batch["multistep_truth"],
                        "exo_impacts": {
                            "GARCH": GARCH_Impacts[idx - 1],
                            "CARMA": CARMA_Impacts[idx - 1]
                        },
                        "risk_free_rate": batch["risk_free_rate"][0],
                        "lag": Lag,
                        "data_type": batch["data_type"][0]
                    }
                )

            nll_loss_train = .0
            nll_loss_valid = .0
            nll_loss_test = .0
            pinn_loss_train = .0
            pinn_loss_valid = .0
            pinn_loss_test = .0
            elbo_loss_train = .0
            elbo_loss_valid = .0
            elbo_loss_test = .0

            if NLL_WEIGHT:
                nll_loss_train += torch.stack(
                    [x for x, y in zip(EKF_Net.multistep_prediction_error, EKF_Net.data_type) if y == "train"]
                ).mean()
                nll_loss_valid += torch.stack(
                    [x for x, y in zip(EKF_Net.multistep_prediction_error, EKF_Net.data_type) if y == "valid"]
                ).mean()
                nll_loss_test += torch.stack(
                    [x for x, y in zip(EKF_Net.multistep_prediction_error, EKF_Net.data_type) if y == "test"]
                ).mean()
            if PINN_WEIGHT:
                pinn_loss_train += torch.exp(torch.stack(
                    [x for x, y in zip(EKF_Net.d_log_relative_risk_density[1:], EKF_Net.data_type[1:]) if y == "train"]
                ).cumsum(dim=0)).mean()
                pinn_loss_valid = torch.exp(torch.stack(
                    [x for x, y in zip(EKF_Net.d_log_relative_risk_density[1:], EKF_Net.data_type[1:]) if y == "valid"]
                ).cumsum(dim=0)).mean()
                pinn_loss_test = torch.exp(torch.stack(
                    [x for x, y in zip(EKF_Net.d_log_relative_risk_density[1:], EKF_Net.data_type[1:]) if y == "test"]
                ).cumsum(dim=0)).mean()
            if ELBO_WEIGHT:
                elbo_loss_train = torch.stack(
                    [x for x, y in zip(EKF_Net.elbo_b_history, EKF_Net.data_type[:-1 - Lag]) if y == "train"]
                ).mean() + torch.stack(
                    [x for x, y in zip(EKF_Net.elbo_c_history, EKF_Net.data_type[:-1 - Lag]) if y == "train"]
                ).mean() + torch.stack(
                    [x for x, y in zip(EKF_Net.elbo_d_history, EKF_Net.data_type[:-1 - Lag]) if y == "train"]
                ).mean()
                elbo_loss_valid = torch.stack(
                    [x for x, y in zip(EKF_Net.elbo_b_history, EKF_Net.data_type[:-1 - Lag]) if y == "valid"]
                ).mean() + torch.stack(
                    [x for x, y in zip(EKF_Net.elbo_c_history, EKF_Net.data_type[:-1 - Lag]) if y == "valid"]
                ).mean() + torch.stack(
                    [x for x, y in zip(EKF_Net.elbo_d_history, EKF_Net.data_type[:-1 - Lag]) if y == "valid"]
                ).mean()
                elbo_loss_test = torch.stack(
                    [x for x, y in zip(EKF_Net.elbo_b_history, EKF_Net.data_type[:-1 - Lag]) if y == "test"]
                ).mean() + torch.stack(
                    [x for x, y in zip(EKF_Net.elbo_c_history, EKF_Net.data_type[:-1 - Lag]) if y == "test"]
                ).mean() + torch.stack(
                    [x for x, y in zip(EKF_Net.elbo_d_history, EKF_Net.data_type[:-1 - Lag]) if y == "test"]
                ).mean()

            train_loss += (
                                      NLL_WEIGHT * nll_loss_train + PINN_WEIGHT * pinn_loss_train - ELBO_WEIGHT * elbo_loss_train) / K_MC
            valid_loss += (
                                      NLL_WEIGHT * nll_loss_valid + PINN_WEIGHT * pinn_loss_valid - ELBO_WEIGHT * elbo_loss_valid) / K_MC
            test_loss += (
                                     NLL_WEIGHT * nll_loss_test + PINN_WEIGHT * pinn_loss_test - ELBO_WEIGHT * elbo_loss_test) / K_MC

            del nll_loss_train, nll_loss_valid, nll_loss_test, pinn_loss_train, pinn_loss_valid, pinn_loss_test, elbo_loss_train, elbo_loss_valid, elbo_loss_test

        Saved = False
        if train_loss < Best_Loss:
            Best_Loss = train_loss.item()
            time.sleep(0.5)
            torch.save(VP_EKF.state_dict(), os.path.join(PATH, "EKF_params.pth"))
            torch.save(VP_GARCH.state_dict(), os.path.join(PATH, "GARCH_params.pth"))
            torch.save(VP_CARMA.state_dict(), os.path.join(PATH, "CARMA_params.pth"))
            Saved = True
        if epoch + 1 >= 50:
            if valid_loss < Best_Valid_Loss:
                Best_Valid_Loss = valid_loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= patience_max:
                    break
        print(
            f"Train loss: {train_loss.item(): .6f}, "
            f"Valid loss: {valid_loss.item(): .6f}, "
            f"Test loss: {test_loss.item(): .6f}, "
            f"Saved: {Saved}"
        )
        train_loss.backward()
        Optimizer.step()
        Lr_Scheduler.step()

        gc.collect()

    VP_CARMA.load_state_dict(torch.load(os.path.join(PATH, "CARMA_params.pth")))
    VP_GARCH.load_state_dict(torch.load(os.path.join(PATH, "GARCH_params.pth")))
    VP_EKF.load_state_dict(torch.load(os.path.join(PATH, "EKF_params.pth")))

    VP_EKF.eval()
    VP_GARCH.eval()
    VP_CARMA.eval()

    EKF_sample = VP_EKF.sample(vi=VI)
    CARMA_sample = VP_CARMA.sample(vi=VI)
    GARCH_sample = VP_GARCH.sample(vi=VI)

    GARCH_Impacts = torch.func.functional_call(GARCH_Net, GARCH_sample, args=(), kwargs={"x": DataSet.exo_var})
    CARMA_Impacts = torch.func.functional_call(CARMA_Net, CARMA_sample, args=(), kwargs={"x": DataSet.exo_var})
    EKF_Net.clear_history(*Prior)
    pbar = tqdm.tqdm(DataLoader, desc="Evaluation")
    for batch in pbar:
        idx = batch["order"]
        torch.func.functional_call(
            EKF_Net, EKF_sample, args=(), kwargs={
                "observation": batch["log_return"][0],
                "multistep_truth": batch["multistep_truth"],
                "exo_impacts": {
                    "GARCH": GARCH_Impacts[idx - 1],
                    "CARMA": CARMA_Impacts[idx - 1]
                },
                "risk_free_rate": batch["risk_free_rate"][0],
                "lag": Lag,
                "data_type": batch["data_type"][0]
            }
        )
    evaluation_plot(
        EKF_Net, DataSet, lag=Lag, theta=EKF_sample["theta"],
        title=f"Evaluation: {NN_TYPE} {'PINN' if PINN_WEIGHT else ''} {'NLL' if NLL_WEIGHT else ''} {'VI' if ELBO_WEIGHT else ''}",
        path=os.path.join(PATH, "Evaluation.pdf")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PI-Kalman-NN model")
    parser.add_argument("--Epochs", type=int, default=80, help="Number of epochs to train")
    parser.add_argument("--Lag", type=int, default=0, help="Lag for the obsersvations")
    parser.add_argument("--VI", type=bool, default=False, help="Use Variational Inference")
    parser.add_argument("--K_MC", type=int, default=1, help="Number of Monte Carlo samples")
    parser.add_argument("--LR", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--NN_TYPE", type=str, default="GRU", help="Type of RNN (GRU or LSTM)")
    parser.add_argument("--NN_ON", type=int, default=1, help="Use NN to train the model")
    parser.add_argument("--PINN_WEIGHT", type=float, default=0, help="Weight for the PINN loss")
    parser.add_argument("--NLL_WEIGHT", type=float, default=1, help="Weight for the NLL loss")
    parser.add_argument("--ELBO_WEIGHT", type=float, default=0, help="Weight for the ELBO loss")
    parser.add_argument("--Year", type=str, default="all", help="Year of the data")
    args = parser.parse_args()

    main(args)
