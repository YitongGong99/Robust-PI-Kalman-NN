# -*- coding: utf-8 -*-
import numpy
import pandas
import os
import torch
from torch import nn
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import tqdm
import itertools
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

NUM_TRADING_DAYS = 262
TIME_ZONE = datetime.timezone.utc
# DEVICE = "cpu"
ETA = 1e-6
PATH = "./"

__all__ = [
    "dataset_pt",
    "Variational_ParamBank",
    "EKF",
    "Exo_Var_NN",
    "evaluation_plot"
]


def raw_data(**kwargs) -> pandas.DataFrame:
    df_ = pandas.read_csv(kwargs.get("file_path", os.path.join(PATH, "data", "aluminium_raw_inputs.csv")))
    df_ = df_.ffill()
    df_.columns = [eval(x) if x.__contains__(",") else tuple([x]) for x in df_.columns]
    df_.columns = [("FEATURES", x[0].split(".")[-1], x[1]) if x[0].__contains__(".") else (x[0], "", "") for x in
                   df_.columns]
    df_.columns = pandas.MultiIndex.from_tuples(df_.columns)
    df_ = df_.rename(columns={x: x[0] if x[1] == x[2] else x for x in df_.columns})
    df_["date"] = pandas.to_datetime(df_["date"])
    for col_ in df_[("FEATURES", "SENTIMENT")].columns:
        if col_.__contains__("pmi"):
            df_[("FEATURES", "SENTIMENT", col_)] -= 50
        if (col_.__contains__("index") and col_ != "vdax_index") or col_ == "china_300" or col_ == "bdi":
            df_[("FEATURES", "SENTIMENT", col_)] = numpy.log(df_[("FEATURES", "SENTIMENT", col_)])

    df_[("FEATURES", "EXCHANGE_CURRENCY", "us_dollar_index")] -= 100
    df_[('FEATURES', 'SENTIMENT', 'us_treasuries_2y')] /= 100
    df_[('FEATURES', 'SENTIMENT', 'us_treasuries_10y')] /= 100

    df_["log_price"] = numpy.log(df_["al_lme_prices"])
    df_["log_return"] = df_["log_price"].diff()
    df_["HV10"] = df_["log_return"].rolling(10).std() * NUM_TRADING_DAYS ** 0.5
    df_["HV20"] = df_["log_return"].rolling(20).std() * NUM_TRADING_DAYS ** 0.5
    df_["HV30"] = df_["log_return"].rolling(30).std() * NUM_TRADING_DAYS ** 0.5
    df_["HV60"] = df_["log_return"].rolling(60).std() * NUM_TRADING_DAYS ** 0.5
    df_ = df_.set_index("date")
    return df_


class dataset_pt(torch.utils.data.Dataset):
    __raw_data = raw_data()

    def __init__(
            self,
            fls_lag: int = 0,
            forecast_horizon: int = 1,
            pca_components: int = None,
            order_diff: int = 0,
            start: datetime.datetime = datetime.datetime(2000, 1, 1),
            end: datetime.datetime = datetime.datetime(2025, 5, 30),
            train_split: datetime.datetime = datetime.datetime(2021, 12, 31),
            valid_split: datetime.datetime = datetime.datetime(2022, 12, 31),
    ):
        self.__data = self.__raw_data.copy()
        self.order_diff = order_diff
        self.start = start
        self.end = end
        self.train_split = train_split
        self.valid_split = valid_split
        self.fls_lag = fls_lag
        self.forecast_horizon = forecast_horizon

        self.__risk_free_rate = self.__data[('FEATURES', 'SENTIMENT', 'us_treasuries_2y')]

        if self.order_diff:
            self.__data["FEATURES"] = self.__data["FEATURES"].diff(self.order_diff)
        self.__data = self.__data.dropna()
        self.__num_exo_features = self.__data["FEATURES"].shape[1]

        self.data = self.__data[(self.__data.index >= self.start) & (self.__data.index <= self.end)]
        self.data = self.data.drop(columns=["FEATURES"])
        self.data.columns = pandas.MultiIndex.from_tuples([("OBSERVATIONS", x[0]) for x in self.data.columns])
        self.data[("OBSERVATIONS", "us_treasuries_2y")] = self.__risk_free_rate.copy()
        self.pca, components_ = self.__exo_var_pca_analysis(n_components=pca_components)
        self.data = pandas.concat([self.data, components_], axis=1)

        self.dates: torch.LongTensor = torch.LongTensor(
            [x // 1e9 for x in pandas.to_datetime(self.data.index, utc=True).astype("int64")])
        self.exo_var: torch.Tensor = torch.Tensor(
            self.data.loc[:, self.data.columns.get_level_values(0) == "EXO_PCA"].to_numpy())
        self.price: torch.Tensor = torch.Tensor(self.data[[("OBSERVATIONS", "al_lme_prices")]].to_numpy())
        self.log_price: torch.Tensor = torch.Tensor(self.data[[("OBSERVATIONS", "log_price")]].to_numpy())
        self.log_return: torch.Tensor = torch.Tensor(self.data[[("OBSERVATIONS", "log_return")]].to_numpy())
        self.risk_free_rate: torch.Tensor = torch.Tensor(
            self.data[[('OBSERVATIONS', 'us_treasuries_2y')]].to_numpy())
        self.start = self.data.index.min()

    def HV(self, window: int = 30) -> torch.Tensor:
        return torch.Tensor(self.data[f"HV{window}"].to_numpy())

    def __exo_var_pca_analysis(self, n_components: int = None):

        # Split the data
        Train_df = self.__data[(self.__data.index >= self.start) & (self.__data.index <= self.train_split)]
        Valid_df = self.__data[(self.__data.index > self.train_split) & (self.__data.index <= self.valid_split)]
        Test_df = self.__data[(self.__data.index > self.valid_split) & (self.__data.index <= self.end)]

        # Z-score normalization
        scaler = StandardScaler()
        Z_Train = scaler.fit_transform(Train_df["FEATURES"])
        Z_Valid = scaler.transform(Valid_df["FEATURES"])
        Z_Test = scaler.transform(Test_df["FEATURES"])

        # PCA
        pca = PCA(n_components=n_components)
        pca.fit(Z_Train)
        PC_Train = pca.transform(Z_Train)
        PC_Valid = pca.transform(Z_Valid)
        PC_Test = pca.transform(Z_Test)

        k = PC_Train.shape[1]
        result = pandas.DataFrame(
            numpy.concatenate([PC_Train, PC_Valid, PC_Test]),
            index=[x for x in Train_df.index] + [x for x in Valid_df.index] + [x for x in Test_df.index],
            columns=pandas.MultiIndex.from_tuples([("EXO_PCA", f"PC{ind + 1}") for ind in range(k)])
        )
        for ind in range(k, self.__num_exo_features):
            result[("EXO_PCA", f"PC{ind + 1}")] = 0
        return pca, result

    def __len__(self):
        return len(self.data) - self.forecast_horizon - 1

    def __getitem__(self, idx_):
        data_set = "train"
        if datetime.datetime.fromtimestamp(self.dates[idx_].item()).date() > self.train_split.date():
            data_set = "valid"
        if datetime.datetime.fromtimestamp(self.dates[idx_].item()).date() > self.valid_split.date():
            data_set = "test"
        return (
            {
                "order": idx_ + 1,
                "exo_var": self.exo_var[idx_],  # exo_var after pca
                "log_return": self.log_return[idx_ + 1],  # log return
                "data_type": data_set,
                "risk_free_rate": self.risk_free_rate[idx_ + 1],
                "multistep_truth": self.log_return[idx_ + 2 - self.fls_lag: idx_ + 1 + self.forecast_horizon + 1]
            }
        )


class Variational_ParamBank(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.param_names = []
        self.shapes = []
        self.sizes = []
        for name, param in model.named_parameters():
            self.param_names.append(name)
            self.shapes.append(param.shape)
            self.sizes.append(param.numel())
        self.total_size = sum(self.sizes)
        self.mu = torch.nn.Parameter(torch.randn(self.total_size) * 0.1)
        self.raw_L = torch.nn.Parameter(
            torch.diag(nn.functional.softplus(torch.randn(self.total_size))) * 0.01
        )

    def sample(self, vi: bool = True) -> dict:
        if not vi:
            sample_ = self.mu
        else:
            eps = torch.randn(self.total_size)
            L = torch.tril(self.raw_L)
            sample_ = self.mu + L @ eps
        param_dict = dict()
        idx_ = 0
        for name, size, shape in zip(self.param_names, self.sizes, self.shapes):
            param_dict[name] = sample_[idx_:idx_ + size].reshape(shape)
            idx_ += size
        return param_dict


class Exo_Var_NN(torch.nn.Module):
    net: torch.nn.Module

    def __init__(self, nn_type: str, target: str, out_mode="linear"):
        super().__init__()
        self.__out_mode = out_mode
        self.nn_type = nn_type.upper()
        hidden_dim: int
        if self.nn_type not in ["FCNN", "LSTM", "GRU", "RNN", "LINEAR"]:
            raise TypeError(f"Unknown neural network name: {self.nn_type}")

        self.target = target.upper()
        if self.target not in ["GARCH", "CARMA"]:
            raise ValueError(f"Unknown target name: {self.target}, should be in ['CARMA', 'GARCH']")

        if self.nn_type == "FCNN":
            hidden_dim = 26
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_features=45, out_features=hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Softplus(),

                torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Softplus(),
            )

        if self.nn_type == "LSTM":
            hidden_dim = 9
            self.net = torch.nn.LSTM(input_size=45, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        if self.nn_type == "GRU":
            hidden_dim = 12
            self.net = torch.nn.GRU(input_size=45, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        if self.nn_type == "RNN":
            hidden_dim = 27
            self.net = torch.nn.RNN(input_size=45, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        self.__out_dim = 1 if self.target == "GARCH" else 2
        self.out_layer = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=self.__out_dim,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nn_type in ["LSTM", "GRU", "RNN"]:
            y = self.out_layer(self.net(x)[0])
        else:
            y = self.out_layer(self.net(x))
        return y


class EKF(torch.nn.Module):
    def __init__(self, w0: torch.Tensor, P0: torch.Tensor):
        """

        :param w0: Mean of prior state
        :param P0: Covariance of prior state
        """
        super(EKF, self).__init__()

        self.dt = 1 / NUM_TRADING_DAYS

        self.b0 = torch.nn.Parameter(torch.randn(1)[0])
        self.b1 = torch.nn.Parameter(torch.randn(1)[0])
        self.a1 = torch.nn.Parameter(torch.randn(1)[0])
        self.kappa = torch.nn.Parameter(torch.randn(1)[0] * 0.01)
        self.theta = torch.nn.Parameter(torch.randn(1)[0] * 0.01)
        self.xi = torch.nn.Parameter(torch.randn(1)[0] * 0.01)
        self.rho = torch.nn.Parameter(torch.randn(1)[0] * 0.01)

        self.jacobian_history: list[dict[str, torch.Tensor]] = [None]
        self.state_prediction_history: list[tuple[torch.Tensor, torch.Tensor]] = [None]
        self.state_posterior_history: list[tuple[torch.Tensor, torch.Tensor]] = [(w0, P0)]
        self.observation_prediction_history: list[tuple[torch.Tensor, torch.Tensor]] = [None]
        self.kalman_gain_history: list[torch.Tensor] = [None]
        self.smoothing_gain_history: list[torch.Tensor] = []
        self.smoothed_posterior_history: list[tuple[torch.Tensor, torch.Tensor]] = [(w0, P0)]
        self.R_history: list[torch.Tensor] = [None]
        self.S_history: list[torch.Tensor] = [None]

        self.observations: list[torch.Tensor] = [None]
        self.data_type: list[str] = ["train"]
        self.exo_impacts = []
        self.multistep_prediction_history = []

        self.observation_prediction_error: list[torch.Tensor] = [torch.zeros(1)]
        self.remain_error: list[torch.Tensor] = []
        self.multistep_prediction_error = []
        self.risk_premium: list[torch.Tensor] = []
        self.dW: list[torch.Tensor] = [torch.tensor(.0)]
        self.d_log_relative_risk_density: list[torch.Tensor] = [torch.ones(1)]

        self.elbo_b_history = []
        self.elbo_c_history = []
        self.elbo_d_history = []

    def clear_history(self, w0: torch.Tensor, P0: torch.Tensor):
        self.jacobian_history: list[dict[str, torch.Tensor]] = [None]
        self.state_prediction_history: list[tuple[torch.Tensor, torch.Tensor]] = [None]
        self.state_posterior_history: list[tuple[torch.Tensor, torch.Tensor]] = [(w0, P0)]
        self.observation_prediction_history: list[tuple[torch.Tensor, torch.Tensor]] = [None]
        self.kalman_gain_history: list[torch.Tensor] = [None]
        self.smoothing_gain_history: list[torch.Tensor] = []
        self.smoothed_posterior_history: list[tuple[torch.Tensor, torch.Tensor]] = [(w0, P0)]
        self.R_history = [None]
        self.S_history = [None]

        self.observations: list[torch.Tensor] = [None]
        self.data_type: list[str] = ["train"]
        self.exo_impacts = []

        self.observation_prediction_error = [torch.zeros(1)[0]]
        self.remain_error: list[torch.Tensor] = []
        self.multistep_prediction_error = []
        self.risk_premium = []
        self.dW = [torch.tensor(.0)]
        self.d_log_relative_risk_density = [torch.zeros(1)]

        self.elbo_b_history = []
        self.elbo_c_history = []
        self.elbo_d_history = []

    # Covariance matrices, p(w_n|w_{n-1}), p(x_n|w_n)
    def R(self, state):
        sigma = torch.exp(state.flatten()[0] / 2)
        xi = nn.functional.softplus(self.xi)
        rho = nn.functional.tanh(self.rho)

        R = torch.stack(
            [
                torch.stack([xi ** 2, torch.tensor(.0), sigma * xi * rho]),
                torch.zeros(3),
                torch.stack([sigma * xi * rho, torch.tensor(.0), sigma ** 2]),
            ]
        ) * self.dt
        return R + torch.diag(torch.ones_like(state)) * ETA

    # Covariance matrices, p(x_n|w_n)
    def S(self, state):
        sigma = torch.exp(state[0] / 2)
        return torch.eye(1) * (sigma ** 2 * self.dt + ETA)

    # State transition equation, w_n = f(w_{n-1}, 0)
    def transition_equation(self, state: torch.Tensor, exo_impacts: dict[str, torch.Tensor]):
        garch_impact = exo_impacts["GARCH"].flatten()
        carma_impact = exo_impacts["CARMA"].flatten()

        kappa = nn.functional.softplus(self.kappa)
        theta = nn.functional.softplus(self.theta + garch_impact[0])
        xi = nn.functional.softplus(self.xi)
        term = kappa * theta * torch.clip(torch.exp(- state[0]), min=0, max=1)
        if torch.isnan(term):
            term = torch.tensor(0.)
        psi = torch.stack(
            [
                term + xi ** 2 / 2 - kappa,
                state[2] + carma_impact[0],
                - self.a1 * state[2] + carma_impact[1],
            ]
        )

        return state + psi * self.dt

    # Emission equation, x_n = g(w_n, 0)
    def emission_equation(self, state):
        phi = torch.stack([torch.tensor(.0), self.b0, self.b1])
        return torch.dot(phi, state) * self.dt

    # Jacobian, Psi
    def psi(self, state: torch.Tensor, exo_impacts: dict[str, torch.Tensor]):
        garch_impact = exo_impacts["GARCH"].flatten()

        kappa = nn.functional.softplus(self.kappa)
        theta = nn.functional.softplus(self.theta + garch_impact[0])

        term = kappa * theta * torch.clip(torch.exp(- state[0]), min=0, max=1)
        if torch.isnan(term):
            term = torch.tensor(0.)
        return torch.stack(
            [
                torch.stack([1 - term, torch.tensor(.0),
                             torch.tensor(.0)]),
                torch.Tensor([0, 1, self.dt]),
                torch.stack([torch.tensor(.0), torch.tensor(.0), 1 - self.a1 * self.dt]),
            ]
        ).reshape(state.shape[0], -1)

    # Jacobian, Phi
    def phi(self, state: torch.Tensor = None):
        return torch.stack([torch.tensor(.0), self.b0, self.b1]).reshape(1, -1) * self.dt

    # State Prediction, p( w_n | F_{n-1} )
    def state_prediction(self, w_prior, P_prior, exo_impacts: dict[str, torch.Tensor], psi: torch.Tensor):
        w_pred = self.transition_equation(state=w_prior, exo_impacts=exo_impacts)
        P_pred = psi @ P_prior @ psi.T + self.R(w_prior)
        P_pred = (P_pred + P_pred.T) / 2 + torch.eye(P_pred.shape[0]) * ETA
        return w_pred, P_pred

    # Observation Prediction, p( x_n | F_{n-1} )
    def observation_prediction(self, w_pred, P_pred, phi):
        x_pred = self.emission_equation(w_pred).flatten()
        Q = phi.reshape(1, -1) @ P_pred @ phi.reshape(-1, 1) + self.S(w_pred)
        Q = (Q + Q.T) / 2 + torch.eye(Q.shape[0]) * ETA  # To ensure symmetry
        return x_pred, Q

    @staticmethod
    # Kalman Gain, K
    def kalman_gain(P_pred, Q_pred, phi):
        K = P_pred @ phi.reshape(-1, 1) @ torch.linalg.inv(Q_pred)
        return K.reshape(P_pred.shape[0], -1)

    @staticmethod
    # Smoothing Gain, J
    def smoothing_gain(P_prior, P_pred, psi):
        J = P_prior @ psi.T @ torch.linalg.inv(P_pred)
        return J

    # Update Smoothed Posterior, p( w_n | F_{n+L} )
    def fixed_lag_smoothing(self, lag: int):
        w, P = self.state_posterior_history[-1]
        for idx in range(1, lag + 1):
            J = self.smoothing_gain_history[- idx]
            w_pred, P_pred = self.state_prediction_history[- idx]
            w_post, P_post = self.state_posterior_history[-1 - idx]
            w = w_post + J @ (w - w_pred)
            P = P_post + J @ (P - P_pred) @ J.T
            P = (P + P.T) / 2 + torch.diag(torch.ones_like(P)) * ETA
        self.smoothed_posterior_history[-1 - lag] = (w, P)

    # Multistep Prediction from Smoothed Posterior, p( w_{n+L+1} | F_{n+L} ) and p( x_{n+L+1} | F_{n+L} )
    def multistep_prediction(self, lag: int):
        state_pred = []
        observation_pred = []
        w_pred, P_pred = self.state_posterior_history[-1 - lag]
        exo_impacts = self.exo_impacts[-1 - lag]
        for step in range(1, lag + 1 + 1):
            psi = self.psi(w_pred, exo_impacts)
            w_pred, P_pred = self.state_prediction(w_pred, P_pred, exo_impacts, psi)
            phi = self.phi(w_pred)
            x_pred, Q_pred = self.observation_prediction(w_pred, P_pred, phi)
            state_pred.append((w_pred, P_pred))
            observation_pred.append((x_pred, Q_pred))
        return state_pred, observation_pred

    @staticmethod
    def __theta(phi: torch.Tensor, w_post: torch.Tensor, rQ: torch.Tensor):
        rQ = rQ.flatten()[0]
        coef = torch.Tensor([1, 0, 0])
        theta = (phi @ w_post + 0.5 * torch.exp(torch.dot(coef, w_post)) - rQ) / torch.exp(
            0.5 * torch.dot(coef, w_post))
        return theta

    def __dW(self, observation: torch.Tensor, w_post: torch.Tensor, phi):
        coef = torch.Tensor([1, 0, 0])
        dW = (observation - phi @ w_post * self.dt) / torch.exp(0.5 * torch.dot(coef, w_post))
        return dW

    @staticmethod
    def elbo_b(
            w_prior: torch.Tensor,
            w_pred: torch.Tensor,
            w_smoothed_prior: torch.Tensor, P_smoothed_prior: torch.Tensor,
            w_smoothed_post: torch.Tensor, P_smoothed_post: torch.Tensor,
            psi: torch.Tensor, R: torch.Tensor, J: torch.Tensor,
    ):
        u = w_pred - psi @ w_prior
        U = R

        E_delta = w_smoothed_post - psi @ w_smoothed_prior - u
        Cov_delta = P_smoothed_post + psi @ P_smoothed_prior @ psi.T - J @ P_smoothed_post @ psi.T - psi @ P_smoothed_post @ J.T
        return - 0.5 * (
                torch.trace(torch.linalg.inv(U) @ Cov_delta) + E_delta.T @ torch.linalg.inv(U) @ E_delta + torch.logdet(
            2 * torch.pi * U)
        )

    @staticmethod
    def elbo_c(
            w_pred: torch.Tensor,
            x_pred: torch.Tensor,
            observation: torch.Tensor,
            w_smoothed_post: torch.Tensor, P_smoothed_post: torch.Tensor,
            phi: torch.Tensor, S: torch.Tensor,
    ):
        v = x_pred - phi @ w_pred
        V = S

        E_gamma = observation - phi @ w_smoothed_post - v
        Cov_gamma = phi @ P_smoothed_post @ phi.T
        return -0.5 * (
            torch.trace(torch.linalg.inv(V) @ Cov_gamma) + E_gamma.T @ torch.linalg.inv(V) @ E_gamma + torch.logdet(2 * torch.pi * V)
        )

    @staticmethod
    def elbo_d(P_smoothed_post: torch.Tensor):
        return 0.5 * (torch.logdet(2 * torch.pi * P_smoothed_post) + P_smoothed_post.shape[0])

    def forward(self, observation: torch.Tensor, multistep_truth: torch.Tensor, exo_impacts: dict[str, torch.Tensor], risk_free_rate: torch.Tensor,
                lag: int, data_type: str, prior: torch.Tensor = None):
        if prior is None:
            w_prior, P_prior = self.state_posterior_history[-1]
        else:
            w_prior, P_prior = prior

        # w_prior, P_prior = w_prior.detach(), P_prior.detach()

        self.data_type.append(data_type)
        self.exo_impacts.append(exo_impacts)

        # Prediction Step
        # Making State Prediction
        psi = self.psi(w_prior, exo_impacts)
        R = self.R(w_prior)
        self.R_history.append(R)
        w_pred, P_pred = self.state_prediction(w_prior=w_prior, P_prior=P_prior, exo_impacts=exo_impacts, psi=psi)
        self.state_prediction_history.append((w_pred, P_pred))

        # Making Observation Prediction
        phi = self.phi(w_pred)
        S = self.S(w_pred)
        self.S_history.append(S)
        x_pred, Q_pred = self.observation_prediction(w_pred=w_pred, P_pred=P_pred, phi=phi)
        self.observation_prediction_history.append((x_pred, Q_pred))

        # Record Jacobian
        self.jacobian_history.append({"psi": psi, "phi": phi})

        # Calculate Kalman Gain and Smoothing Gain
        K = self.kalman_gain(P_pred=P_pred, Q_pred=Q_pred, phi=phi)
        self.kalman_gain_history.append(K)
        J = self.smoothing_gain(P_prior=P_prior, P_pred=P_pred, psi=psi)
        self.smoothing_gain_history.append(J)

        # Update Step
        w_post = w_pred + K @ (observation - x_pred)
        I = torch.diag(torch.ones_like(w_prior))
        P_post = (I - K @ phi) @ P_pred @ (I - K @ phi).T + K @ Q_pred @ K.T
        P_post = (P_post + P_post.T) / 2 + torch.diag(torch.ones_like(P_post)) * ETA  # To ensure symmetry
        self.state_posterior_history.append((w_post, P_post))
        self.smoothed_posterior_history.append((w_post, P_post))
        diff = observation - x_pred
        self.remain_error.append(diff)
        self.observation_prediction_error.append(
            diff.T @ torch.linalg.inv(Q_pred) @ diff + torch.logdet(2 * torch.pi * Q_pred)
        )
        self.observations.append(observation)

        # Smooth the Previous States
        if len(self.state_posterior_history) > lag:
            self.fixed_lag_smoothing(lag=lag)

            theta = self.__theta(phi=phi, w_post=self.state_posterior_history[-1 - lag][0], rQ=risk_free_rate)
            self.risk_premium.append(theta)
            dW = self.__dW(observation=observation, w_post=self.state_posterior_history[-1 - lag][0], phi=phi)
            self.dW.append(dW)
            dl = -1/2 * theta ** 2 * self.dt - theta * dW
            self.d_log_relative_risk_density.append(dl)
            self.elbo_d_history.append(
                self.elbo_d(P_smoothed_post=self.smoothed_posterior_history[-1 - lag][1])
            )

        if len(self.state_posterior_history) > (lag + 1):
            # Multistep Prediction from n-L to n+1
            multi_pred = self.multistep_prediction(lag=lag)
            self.multistep_prediction_history.append(multi_pred)
            error = []
            for truth, (x, Q) in zip(multistep_truth.flatten(), multi_pred[1]):
                diff = truth - x
                error.append(
                    diff.T @ torch.linalg.inv(Q) @ diff + torch.logdet(2 * torch.pi * Q)
                )
            self.multistep_prediction_error.append(torch.stack(error).mean())

            w_prior, _ = self.state_posterior_history[-1 - lag - 1]
            w_pred, _ = self.state_prediction_history[-1 - lag]
            x_pred, _ = self.observation_prediction_history[-1 - lag]
            obs_lag = self.observations[-1 - lag]
            w_smoothed_prior, P_smoothed_prior = self.state_posterior_history[-1 - lag - 1]
            w_smoothed_post, P_smoothed_post = self.smoothed_posterior_history[-1 - lag]
            psi = self.jacobian_history[-1 - lag]["psi"]
            phi = self.jacobian_history[-1 - lag]["phi"]
            R = self.R_history[-1 - lag]
            S = self.S_history[-1 - lag]
            J = self.smoothing_gain_history[-1 - lag]
            self.elbo_b_history.append(
                self.elbo_b(
                    w_prior=w_prior,
                    w_pred=w_pred,
                    w_smoothed_prior=w_smoothed_prior, P_smoothed_prior=P_smoothed_prior,
                    w_smoothed_post=w_smoothed_post, P_smoothed_post=P_smoothed_post,
                    psi=psi, R=R, J=J
                )
            )
            self.elbo_c_history.append(
                self.elbo_c(
                    w_pred=w_pred,
                    x_pred=x_pred,
                    observation=obs_lag,
                    w_smoothed_post=w_smoothed_post,
                    P_smoothed_post=P_smoothed_post,
                    phi=phi, S=S
                )
            )


def pinn_loss(ekf: EKF, ):
    dw = torch.stack(ekf.dW[1:])
    theta = torch.stack(ekf.risk_premium)
    delta_log_L = -0.5 * torch.pow(theta, 2) * ekf.dt - theta * dw
    log_L = delta_log_L.cumsum(0)
    return torch.pow(torch.exp(log_L).mean() - 1, 2)


class Loss_Func(torch.nn.Module):
    def __init__(self, ekf: EKF, weight=None):
        super().__init__()

        if weight is None:
            weight = {
                "elbo": 1,
                "pinn": 1,
                "nll": 1
            }
        self.model = ekf

    def PINN_Loss(self):
        dW = torch.stack(self.model.dW[1:])
        theta = torch.stack(self.model.risk_premium)
        delta_log_L = -0.5 * torch.pow(theta, 2) * self.model.dt - theta * dW
        log_L = delta_log_L.cumsum(0)

        log_L_train = torch.stack([l for idx, l in enumerate(log_L) if self.model.data_type[idx + 1] == "train"])
        log_L_valid = torch.stack([l for idx, l in enumerate(log_L) if self.model.data_type[idx + 1] == "valid"])
        log_L_test = torch.stack([l for idx, l in enumerate(log_L) if self.model.data_type[idx + 1] == "test"])

        return {
            "train": torch.pow(torch.exp(log_L_train).mean() - 1, 2) * len(log_L_train) if len(log_L_train) > 0 else torch.tensor(.0),
            "valid": torch.pow(torch.exp(log_L_valid).mean() - 1, 2) * len(log_L_valid) if len(log_L_valid) > 0 else torch.tensor(.0),
            "test": torch.pow(torch.exp(log_L_test).mean() - 1, 2) * len(log_L_test) if len(log_L_test) > 0 else torch.tensor(.0),
        }

    def NLL_Loss(self):
        error = torch.stack(self.model.observation_prediction_error)
        error_train = torch.stack([e for idx, e in enumerate(error[1:]) if self.model.data_type[idx + 1] == "train"])
        error_valid = torch.stack([e for idx, e in enumerate(error[1:]) if self.model.data_type[idx + 1] == "valid"])
        error_test = torch.stack([e for idx, e in enumerate(error[1:]) if self.model.data_type[idx + 1] == "test"])
        return {
            "train": error_train.sum() if len(error_train) > 0 else torch.tensor(.0),
            "valid": error_valid.sum() if len(error_valid) > 0 else torch.tensor(.0),
            "test": error_test.sum() if len(error_test) > 0 else torch.tensor(.0),
        }

    def ELBO(self):
        raise NotImplementedError

    def forward(self):
        return self.NLL_Loss()


def model_evaluation(
        ekf: EKF,
        garch_net: Exo_Var_NN,
        carma_net: Exo_Var_NN,
        vp_ekf: Variational_ParamBank,
        vp_garch: Variational_ParamBank,
        vp_carma: Variational_ParamBank,
        dataset: dataset_pt,
        lag: int,
        w0: torch.Tensor,
        P0: torch.Tensor,
        NN_on = True,
        vi=True,
        train=True,
        **kwargs
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    if train:
        ekf.train()
        garch_net.train()
        carma_net.train()
        vp_ekf.train()
        vp_garch.train()
        vp_carma.train()
    else:
        ekf.eval()
        garch_net.eval()
        carma_net.eval()
        vp_ekf.eval()
        vp_garch.eval()
        vp_carma.eval()

    ekf.clear_history(w0, P0)

    mc_sample_ekf = vp_ekf.sample(vi=vi)
    mc_sample_garch = vp_garch.sample(vi=vi)
    mc_sample_carma = vp_carma.sample(vi=vi)

    garch_impacts = torch.func.functional_call(garch_net, mc_sample_garch, args=(dataset.exo_var,)) * (1 if NN_on else 0)
    carma_impacts = torch.func.functional_call(carma_net, mc_sample_carma, args=(dataset.exo_var,)) * (1 if NN_on else 0)
    pbar = tqdm.tqdm(dataloader, desc=kwargs.get("prefix"))
    for batch_ in pbar:
        torch.func.functional_call(
            module=ekf,
            parameter_and_buffer_dicts=mc_sample_ekf,
            args=(),
            kwargs={
                "observation": batch_["log_return"].flatten(),
                "lag": lag,
                "exo_impacts": {
                    "GARCH": garch_impacts[batch_["order"] - 1].flatten(),
                    "CARMA": carma_impacts[batch_["order"] - 1].flatten(),
                },
                "risk_free_rate": batch_["risk_free_rate"].flatten(),
                "data_type": batch_["data_type"][0],
            },
        )
        W = ekf.state_posterior_history[-1]
        if torch.isnan(W[0][0]):
            print("There is NaN")
            break
        pbar.set_postfix({"State": W[0].detach()})
    ...


def evaluation_plot(model: EKF, dataset: dataset_pt, lag: int, **kwargs):
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    Len = len(model.state_posterior_history) - 1
    date = [datetime.datetime.fromtimestamp(x).date() for x in dataset.dates[1: Len + 1]]

    obs_pred = torch.stack([x[0] for x in model.observation_prediction_history[1:]]).detach().numpy()
    ax[0].plot(date, dataset.log_return[1: Len + 1].cpu().numpy(), alpha=0.5, label="Truth")
    ax[0].plot(date, obs_pred, label="Prediction")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(date, dataset.log_return[1: Len + 1].cumsum(0).cpu().numpy(), alpha=0.5, label="Cumulative Truth")
    ax[1].plot(date, obs_pred.cumsum(), label="Cumulative Prediction")
    ax[1].grid(True)
    ax[1].legend()

    state_post = torch.stack([torch.exp(x[0][0] / 2) for x in model.smoothed_posterior_history[1:]]).detach().numpy()
    ax[2].plot(date, dataset.data[("OBSERVATIONS", "HV60")][1: Len + 1], alpha=0.5, label="HV60")
    ax[2].plot(date, state_post, label="Inferred Volatility")
    theta = kwargs.get('theta', torch.tensor(.0))
    # ax[2].plot(date, nn.functional.softplus(theta + torch.stack([x["GARCH"] for x in model.exo_impacts])).flatten(
    # ).detach())
    ax[2].grid(True)
    ax[2].legend()

    ax[3].plot(date, torch.stack(model.remain_error).detach(), label="Remain Noise")
    ax[3].grid(True)
    ax[3].legend()

    ax[0].set_title(label=kwargs.get("title", "Model Evaluation"))

    for ax_ in ax:
        ax_.axvspan(dataset.start, dataset.train_split, color="green", alpha=0.2, label="Train")
        ax_.axvspan(dataset.train_split, dataset.valid_split, color="red", alpha=0.2, label="Validation")
        ax_.axvspan(dataset.valid_split, dataset.end, color="blue", alpha=0.2, label="Test")
    if kwargs.get("path", None):
        fig.savefig(kwargs.get("path"), bbox_inches="tight")
    if kwargs.get("show"):
        plt.show()


def model_train(
        ekf: EKF,
        garch_net: Exo_Var_NN,
        carma_net: Exo_Var_NN,
        vp_ekf: Variational_ParamBank,
        vp_garch: Variational_ParamBank,
        vp_carma: Variational_ParamBank,
        dataset: dataset_pt,
        lag: int,
        w0: torch.Tensor,
        P0: torch.Tensor,
        loss_func,
        epochs: int=100,
        lr: float|int= 1e-1,
        NN_on=True,
        vi=True,
        k_mc: int=1,
        **kwargs
):
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        itertools.chain(vp_ekf.parameters(), vp_carma.parameters(), vp_garch.parameters()), lr=lr
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    for epoch in range(epochs):
        loss = .0
        optimizer.zero_grad()
        for mc in range(k_mc):
            model_evaluation(
                ekf=ekf,
                garch_net=garch_net,
                carma_net=carma_net,
                vp_ekf=vp_ekf,
                vp_garch=vp_garch,
                vp_carma=vp_carma,
                dataset=dataset,
                lag=lag,
                w0=w0,
                P0=P0,
                NN_on=NN_on,
                vi=vi,
                train=True,
                prefix=f"Epoch: {epoch + 1} / {epochs}, MC sample: {mc + 1}/{k_mc}"
            )
            loss += loss_func(ekf)()["train"] / k_mc
            print(f"Epoch {epoch + 1} --- loss: {loss.item(): .6f}")
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


if __name__ == '__main__':
    torch.manual_seed(0)
    VI = False
    NN_Type = "GRU"
    Prior = (torch.Tensor([torch.log(torch.tensor(0.04)), 0, 0]), torch.eye(3) * 0.01)

    EKF_Net = EKF(*Prior)
    GARCH_Net = Exo_Var_NN(nn_type=NN_Type, target="GARCH")
    CARMA_Net = Exo_Var_NN(nn_type=NN_Type, target="CARMA")

    VP_EKF = Variational_ParamBank(EKF_Net)
    VP_GARCH = Variational_ParamBank(GARCH_Net)
    VP_CARMA = Variational_ParamBank(CARMA_Net)

    Lag = 10
    K_MC = 3 if VI else 1
    Year = 2019
    Data = dataset_pt(
        start=datetime.datetime(Year, 1, 1),
        end=datetime.datetime(Year + 4, 12, 31),
        train_split=datetime.datetime(Year + 2, 12, 31),
        valid_split=datetime.datetime(Year + 3, 12, 31),
        order_diff=1,
        pca_components=10,
        fls_lag=Lag,
        forecast_horizon=1
    )

    # model_evaluation(
    #     ekf=EKF_Net,
    #     garch_net=GARCH_Net,
    #     carma_net=CARMA_Net,
    #     vp_ekf=VP_EKF,
    #     vp_garch=VP_GARCH,
    #     vp_carma=VP_CARMA,
    #     dataset=Data,
    #     lag=Lag,
    #     w0=Prior[0],
    #     P0=Prior[1],
    #     vi=False,
    #     train=False,
    #     prefix=f"Final Evaluation"
    # )
    # evaluation_plot(EKF_Net, Data, lag=Lag, title=f"Final Evaluation")

    # model_train(
    #     ekf=EKF_Net,
    #     garch_net=GARCH_Net,
    #     carma_net=CARMA_Net,
    #     vp_ekf=VP_EKF,
    #     vp_carma=VP_CARMA,
    #     vp_garch=VP_GARCH,
    #     dataset=Data,
    #     lag=Lag,
    #     w0=Prior[0],
    #     P0=Prior[1],
    #     loss_func=Loss_Func
    # )
    #
    Path = os.path.join(
        "experiment_output",
        f"{NN_Type}_{Data.start.strftime('%Y%m%d')}-{Data.end.strftime('%Y%m%d')}_VI-{VI}_Lag-{Lag}_PINN-{True}_MSE-{True}"
    )
    os.makedirs(Path, exist_ok=True)

    Optimizer = torch.optim.Adam(
        itertools.chain(VP_EKF.parameters(), VP_GARCH.parameters(), VP_CARMA.parameters()),
        lr=1e-1
    )
    Lr_scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=10, gamma=0.9)
    Epochs = 100

    best_loss = torch.inf

    for epoch in range(Epochs):
        loss = .0
        early_stop_loss = .0
        Optimizer.zero_grad()
        for k_mc in range(K_MC):
            EKF_Sample = VP_EKF.sample(vi=VI)
            GARCH_Sample = VP_GARCH.sample(vi=VI)
            CARMA_Sample = VP_CARMA.sample(vi=VI)

            model_evaluation(
                ekf=EKF_Net,
                garch_net=GARCH_Net,
                carma_net=CARMA_Net,
                vp_ekf=VP_EKF,
                vp_garch=VP_GARCH,
                vp_carma=VP_CARMA,
                dataset=Data,
                lag=Lag,
                w0=torch.Tensor([2 * torch.log(torch.tensor(Data.data[("OBSERVATIONS", "HV60")][0])), 0.0, 0.0]),
                # w0=Prior[0],
                P0=Prior[1],
                vi=VI,
                train=True,
                prefix=f"Epoch: {epoch + 1}/{Epochs}, MC Sample: {k_mc + 1}/{K_MC}"
            )

            # mse_loss = nn.MSELoss(reduction="sum")(
            #   torch.stack([x[0] for x in EKF_Net.observation_prediction_history[1:]]).cumsum(0),
            #   torch.stack([x for x in EKF_Net.observations[1:]]).cumsum(0)
            # )
            # loss += (100 * pinn_loss(EKF_Net) + 1 * torch.stack(
            #     [x.flatten() for x in EKF_Net.observation_prediction_error]).sum()) / K_MC
            # loss += (100 * pinn_loss(EKF_Net) + 1 * mse_loss) / K_MC
            Loss = Loss_Func(EKF_Net)
            MSE_loss_train = Loss.NLL_Loss()["train"]
            PINN_loss_train = Loss.PINN_Loss()["train"]
            MSE_loss_valid = Loss.NLL_Loss()["valid"]
            PINN_loss_valid = Loss.PINN_Loss()["valid"]

            loss += 0 * PINN_loss_train / K_MC + 1 * MSE_loss_train / K_MC

            early_stop_loss += 0 * (PINN_loss_train + PINN_loss_valid) / K_MC + 1 * (MSE_loss_train + MSE_loss_valid) / K_MC

        evaluation_plot(EKF_Net, Data, lag=Lag, title=f"Epoch {epoch + 1}/{Epochs}")
        Save = False
        if early_stop_loss < best_loss:
            torch.save(VP_EKF.state_dict(), os.path.join(Path, "VP_EKF.pth"))
            torch.save(VP_GARCH.state_dict(), os.path.join(Path, "VP_GARCH.pth"))
            torch.save(VP_CARMA.state_dict(), os.path.join(Path, "VP_CARMA.pth"))
            with torch.no_grad():
                best_loss = early_stop_loss
            Save = True
        print(f"Epoch {epoch + 1}/{Epochs}, Train Loss: {loss.item():.6f}, Validation Loss: {(MSE_loss_valid + PINN_loss_valid).item(): .6f}, Total Loss: {early_stop_loss}, saved: {Save}")

        if torch.isnan(loss):
            break
        loss.backward()
        Optimizer.step()
        Lr_scheduler.step()

    VP_EKF.load_state_dict(torch.load(os.path.join(Path, "VP_EKF.pth")))
    VP_GARCH.load_state_dict(torch.load(os.path.join(Path, "VP_GARCH.pth")))
    VP_CARMA.load_state_dict(torch.load(os.path.join(Path, "VP_CARMA.pth")))