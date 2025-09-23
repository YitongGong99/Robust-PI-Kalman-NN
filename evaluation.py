# -*- coding: utf-8 -*-
import torch
import datetime
from model import EKF, dataset_pt, Exo_Var_NN, Variational_ParamBank
import tqdm
import os
import matplotlib.pyplot as plt


class Evaluator(EKF):
    path = "experiment_output"
    dataset: dataset_pt
    NN_Type: str
    VI: bool
    PINN: bool
    NN_ON: bool

    def __init__(self, lag: int, year=2016):
        self.lag = lag
        year = int(year)
        if year == 2016:
            self.dataset = dataset_pt(
                fls_lag=self.lag,
                pca_components=10,
                order_diff=1
            )
        else:
            self.dataset = dataset_pt(
                start=datetime.datetime(year, 1, 1),
                train_split=datetime.datetime(year + 2, 12, 31),
                valid_split=datetime.datetime(year + 3, 12, 31),
                end=datetime.datetime(year + 4, 12, 31),
                fls_lag=self.lag,
                pca_components=10,
                order_diff=1
            )
        super().__init__(*(
            torch.Tensor([2 * torch.log(torch.tensor(self.dataset.data[("OBSERVATIONS", "HV60")][0])), 0, 0]),
            torch.eye(3)
        ))

    def evaluate(self, nn_type: str, vi: bool, pinn: bool, nn_on: bool):

        self.NN_Type = nn_type
        self.VI = vi
        self.PINN = pinn
        self.NN_ON = nn_on

        vi = "T" if vi else "F"
        pinn = "T" if pinn else "F"

        param_path = os.path.join(
            self.path,
            "_".join(
                [
                    f"{nn_type.upper()}" if nn_on else "Baseline",
                    f"{self.dataset.start.strftime('%Y%m%d')}-{self.dataset.end.strftime('%Y%m%d')}",
                    f"Lag-{self.lag}",
                    f"VI-{vi}",
                    f"PINN-{pinn}",
                    f"MSE-T"
                ]
            )
        )

        garch_net = Exo_Var_NN(nn_type=nn_type, target="GARCH")
        carma_net = Exo_Var_NN(nn_type=nn_type, target="CARMA")

        vp_ekf = Variational_ParamBank(self)
        vp_garch = Variational_ParamBank(garch_net)
        vp_carma = Variational_ParamBank(carma_net)

        vp_ekf.load_state_dict(torch.load(os.path.join(param_path, "EKF_params.pth")))
        vp_carma.load_state_dict(torch.load(os.path.join(param_path, "CARMA_params.pth")))
        vp_garch.load_state_dict(torch.load(os.path.join(param_path, "GARCH_params.pth")))

        self.eval()
        garch_net.eval()
        carma_net.eval()
        vp_ekf.eval()
        vp_garch.eval()
        vp_carma.eval()

        garch_impacts = torch.func.functional_call(garch_net, vp_garch.sample(vi=False), self.dataset.exo_var) * nn_on
        carma_impacts = torch.func.functional_call(carma_net, vp_carma.sample(vi=False), self.dataset.exo_var) * nn_on

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)
        pbar = tqdm.tqdm(dataloader, desc="Evaluating EKF")
        self.clear_history(
            *(
                torch.Tensor([2 * torch.log(torch.tensor(self.dataset.data[("OBSERVATIONS", "HV60")][0])), 0, 0]),
                torch.eye(3)
            )
        )

        for batch in pbar:
            idx = batch["order"]
            torch.func.functional_call(
                self, vp_ekf.sample(vi=False), args=(), kwargs={
                    "observation": batch["log_return"][0],
                    "multistep_truth": batch["multistep_truth"],
                    "exo_impacts": {
                        "GARCH": garch_impacts[idx - 1],
                        "CARMA": carma_impacts[idx - 1]
                    },
                    "risk_free_rate": batch["risk_free_rate"][0],
                    "lag": self.dataset.fls_lag,
                    "data_type": batch["data_type"][0]
                }
            )

    def plot_data_split(self, ax_: plt.Axes):
        ax_.axvspan(self.dataset.start, self.dataset.train_split, color="green", alpha=0.1)
        ax_.axvspan(self.dataset.train_split, self.dataset.valid_split, color="red", alpha=0.1)
        ax_.axvspan(self.dataset.valid_split, self.dataset.end, color="blue", alpha=0.1)

    @staticmethod
    def data_split(target_series: list, type_series: list, key: str = "train"):
        return torch.stack([x for x, y in zip(target_series, type_series) if y.lower() == key])

    def observation_plot(self, ax_: plt.Axes, scale: int = 2):
        dates = [datetime.datetime.fromtimestamp(x).date() for x in self.dataset.dates]

        ax_.plot(dates, self.dataset.log_return.detach().numpy(), label="truth", alpha=0.8, linewidth=2)
        ax_.plot(dates[1:-1], torch.stack([x[0] for x in self.observation_prediction_history[1:]]).detach().numpy(),
                 label="EKF", linewidth=2)
        ax_.fill_between(
            x=dates[1:-1],
            y1=(
                    torch.stack(
                        [x[0] for x in self.observation_prediction_history[1:]]
                    ) - scale * torch.stack(
                [
                    torch.pow(x[1], 0.5).flatten() for x in self.observation_prediction_history[1:]
                ]
            )
            ).flatten().detach().numpy(),
            y2=(torch.stack([x[0] for x in self.observation_prediction_history[1:]]) + scale * torch.stack(
                [torch.pow(x[1], 0.5).flatten() for x in
                 self.observation_prediction_history[1:]])).flatten().detach().numpy(),
            color="grey", alpha=0.5, label=r"2$\sigma$ CI"
        )
        ax_.legend()
        ax_.grid()
        ax_.set_ylabel("Log Return")

        self.plot_data_split(ax_=ax_)

    @property
    def observation_diff_series(self):
        diff = torch.stack([x for x in self.observations[1:]]) - torch.stack(
            [x[0] for x in self.observation_prediction_history[1:]])
        truth = torch.stack([x for x in self.observations[1:]])
        return diff, truth, {
            "res": diff.abs().mean().item(),
            "truth": truth.abs().mean().item()
        }, {
            "res": diff.std().item(),
            "truth": truth.std().item()
        }

    def observation_error_report(self, scale: int = 2):
        data_type = [x for x in self.data_type[1:]]

        truth_abs_mean = {}
        truth_std = {}
        log_return_rmse = {}
        log_price_rmse = {}
        pred_coverage = {}
        res_abs_mean = {}
        res_std = {}

        for key in ["train", "valid", "test"]:
            truth = torch.stack([x for x in self.observations[1:]])
            truth = self.data_split(truth, data_type, key)
            pred_mean = torch.stack([x[0] for x in self.observation_prediction_history[1:]])
            pred_mean = self.data_split(pred_mean, data_type, key)
            pred_cov = torch.stack([x[1] for x in self.observation_prediction_history[1:]])
            pred_cov = self.data_split(pred_cov, data_type, key)
            upper = pred_mean + scale * torch.pow(pred_cov, 0.5)
            lower = pred_mean - scale * torch.pow(pred_cov, 0.5)
            coverage = ((truth <= upper) & (truth >= lower)).float().mean().item()
            pred_coverage.update({key: coverage})
            log_return_rmse.update({key: torch.pow(torch.nn.MSELoss()(truth, pred_mean), 0.5).item()})
            log_price_rmse.update({key: torch.pow(torch.nn.MSELoss()(truth.cumsum(0), pred_mean.cumsum(0)), 0.5).item()})
            res = truth - pred_mean
            res_abs_mean.update({key: res.abs().mean().item()})
            res_std.update({key: res.std().item()})
            truth_abs_mean.update({key: truth.abs().mean().item()})
            truth_std.update({key: truth.std().item()})
        return {
            "log_price_RMSE": log_price_rmse,
            "log_return_RMSE": log_return_rmse,
            "log_return_res_abs_mean": res_abs_mean,
            "log_return_res_std": res_std,
            "log_return_coverage": pred_coverage,
            "truth_abs_mean": truth_abs_mean,
            "truth_std": truth_std
        }

    def observation_diff_plot(self, ax_: plt.Axes):
        dates = [datetime.datetime.fromtimestamp(x).date() for x in self.dataset.dates]
        obs, truth, abs_mean, std = self.observation_diff_series
        ax_.plot(
            dates[1:-1], truth.abs().detach().numpy(),
            label=fr"truth $r$, $\bar r=${abs_mean['truth']: .4f}, $\sigma_r$={std['truth']: .4f}",
            alpha=0.6
        )
        ax_.plot(
            dates[1:-1], obs.abs().detach().numpy(),
            label=fr"res $\epsilon$, $\bar \epsilon=${abs_mean['res']: .4f}, $\sigma_\epsilon=${std['res']: .4f}",
            alpha=0.6
        )
        ax_.legend()
        ax_.grid()
        ax_.set_ylabel("Log Return")

        self.plot_data_split(ax_)

    def log_return_cumsum_plot(self, ax_: plt.Axes):
        obs = torch.stack([x[0] for x in self.observation_prediction_history[1:]]).cumsum(0)
        truth = torch.stack([x for x in self.observations[1:]]).cumsum(0)
        dates = [datetime.datetime.fromtimestamp(x).date() for x in self.dataset.dates]
        ax_.plot(dates[1: -1], truth.detach().numpy(), label="truth", linewidth=2)
        ax_.plot(dates[1:-1], obs.detach().numpy(), label="EKF", linewidth=2)

        ax_.legend()
        ax_.grid()
        ax_.set_ylabel("Normalized Log Return")

        self.plot_data_split(ax_=ax_)

    def volatility_error_report(self):
        data_type = [x for x in self.data_type]
        vol = torch.stack([torch.exp(x[0][0] / 2) for x in self.smoothed_posterior_history])
        HV = torch.Tensor(self.dataset.data[("OBSERVATIONS", "HV60")][:-1])

        error = {}
        for key in ["train", "valid", "test"]:
            pred = self.data_split(vol, data_type, key)
            hv = self.data_split(HV, data_type, key)
            error.update({key: torch.pow(torch.nn.MSELoss()(pred, hv), 0.5).item()})
        return error

    def volatility_plot(self, ax_: plt.Axes):
        dates = [datetime.datetime.fromtimestamp(x).date() for x in self.dataset.dates]
        vol = torch.stack([torch.exp(x[0][0] / 2) for x in self.smoothed_posterior_history])
        HV = self.dataset.data[("OBSERVATIONS", "HV60")]

        ax_.plot(HV, label="HV60")
        ax_.plot(dates[:-1], vol.detach().numpy(), label="Implied Vol")
        ax_.legend()
        ax_.grid()
        ax_.set_ylabel("Volatility")

        self.plot_data_split(ax_=ax_)

    def final_report(self):
        result = {}
        result.update(self.observation_error_report())
        result.update({"vol_inference_error": self.volatility_error_report()})
        return result

    def final_plot(self, **kwargs) -> plt.Figure:
        fig_, ax_ = plt.subplots(4, 1, figsize=(15, 14))
        ax_[0].set_title(f"Evaluation Results: {'Baseline' if not self.NN_ON else self.NN_Type} Lag-{self.lag} NLL {'VI' if self.VI else ''} {'PINN' if self.PINN else ''}")
        self.observation_plot(ax_[0], 2)
        self.observation_diff_plot(ax_[1])
        self.log_return_cumsum_plot(ax_[2])
        self.volatility_plot(ax_[3])
        if kwargs.get("save_path"):
            plt.savefig(kwargs.get("save_path"), bbox_inches='tight')
        return fig_


if __name__ == "__main__":
    Lag = 0
    NN_Type = "GRU"
    VI = False
    PINN = True
    NN_ON = True

    Model = Evaluator(lag=Lag, year=2020)

    Model.evaluate(nn_type=NN_Type, vi=VI, pinn=PINN, nn_on=NN_ON)

    Model.final_plot().show()

    print(Model.final_report())
