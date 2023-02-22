import random

import slim
import torch
from neuromancer.constraint import variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from neuromancer.dataset import read_file, get_sequence_dataloaders
from neuromancer.estimators import TimeDelayEstimator, RNNEstimator
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer

random.seed(127)


class GRUEstimator(TimeDelayEstimator):
    def __init__(
        self,
        data_dims,
        nsteps=1,
        window_size=1,
        bias=False,
        linear_map=slim.Linear,
        nonlin=nn.LeakyReLU,
        hsizes=[64],
        input_keys=["Yp"],
        linargs=dict(),
        name="RNN_estim",
    ):
        """
        see base class for arguments
        """
        super().__init__(
            data_dims,
            nsteps=nsteps,
            window_size=window_size,
            input_keys=input_keys,
            name=name,
        )
        self.in_features = self.sequence_dims_sum
        self.net = nn.GRU(
            input_size=self.in_features,
            hidden_size=hsizes[0],
            num_layers=1,
            bias=bias,
            batch_first=True,
            dropout=0.075,
            bidirectional=False,
        )
        self.last_layer = slim.Linear(hsizes[0], self.out_features, **linargs)
        self.nonlin = nonlin()

    def forward(self, data):
        features = torch.cat(
            [
                data[k][:, self.nsteps - self.window_size : self.nsteps, :]
                for k in self.input_keys
            ],
            dim=2,
        )
        output = {
            name: tensor
            for tensor, name in zip(
                [self.net(features), self.reg_error()], self.output_keys
            )
        }
        return output


def load_data():
    data = read_file(
        "/Users/tudoroancea/Developer/racing_team/brains/src/brains_python/work/datasets/data/ve_dataset/combined.csv"
    )
    return data


def main_neuromancer():
    nsteps = 1  # sequence length: each sequence provided to the network is 250 time steps long, i.e. 2.5 seconds
    batch_size = 1  # batch size: each batch contains 1 sequence
    device = "cpu"
    # prepare data =============================================================
    raw_data = load_data()
    random.shuffle(raw_data)
    (
        (train_data, dev_data, test_data),
        (train_loop, dev_loop, test_loop),
        dims,
    ) = get_sequence_dataloaders(
        data=raw_data,
        nsteps=nsteps,
        moving_horizon=True,
        batch_size=batch_size,
    )
    sequences = next(iter(train_data))
    print({k: v.shape for k, v in sequences.items() if isinstance(v, torch.Tensor)})
    print("dims: ", dims)
    # build model ==============================================================
    nx = dims["X"][1]
    ny = dims["Y"][1]
    # estim = GRUEstimator(
    #     {**train_data.dataset.dims, "x0": (nx,)},
    #     linear_map=slim.Linear,
    #     input_keys=["Yp"],
    # )
    estim = RNNEstimator(
        {**train_data.dataset.dims, "x0": (nx,)},
        linear_map=slim.Linear,
        input_keys=["Yp"],
        nonlin=nn.LeakyReLU,
        hsizes=[64],
    )
    batch = train_data.dataset.get_full_batch()
    estimation = estim(batch)
    print(
        "batch.shape={}, estimation.shape={}".format(
            batch["Yp"].shape, estimation[estim.output_keys[0]].shape
        )
    )

    # formulate problem (loss and constraints) ================================
    x_estim = variable(estim.output_keys[0])
    x_true = variable("Xf")
    ref_loss = (x_estim == x_true) ^ 2
    ref_loss.name = "ref_loss"

    loss = PenaltyLoss([ref_loss], [])
    problem = Problem([estim], loss)

    # plt.figure()
    # problem.plot_graph("gruve_graph.png")

    # solve problem ============================================================
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
    # trainer
    cl_trainer = Trainer(
        problem,
        train_data,
        dev_data,
        test_data,
        optimizer,
        epochs=10,
        patience=100,
        train_metric="nstep_train_loss",
        dev_metric="nstep_dev_loss",
        test_metric="nstep_test_loss",
        eval_metric="nstep_dev_loss",
        warmup=10,
    )
    best_model_cl = cl_trainer.train()
    best_outputs = cl_trainer.test(best_model_cl)
    print("best_outputs: ", best_outputs)


class GRUVE(nn.Module):
    def __init__(self, nx, ny, nh=64, nl=1, dropout=0.075, nonlin=nn.LeakyReLU):
        super(GRUVE, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nh = nh
        self.nl = nl
        self.dropout = dropout
        self.nonlin = nonlin()
        self.net = nn.GRU(
            input_size=self.ny,
            hidden_size=self.nh,
            num_layers=self.nl,
            bias=True,
            batch_first=True,
            # dropout=self.dropout,
            bidirectional=False,
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.last_layer = nn.Linear(self.nh, self.nx)

    def forward(self, input):
        input = self.net(input)[0]
        input = self.nonlin(input)
        input = self.dropout_layer(input)
        input = self.last_layer(input)
        input = self.nonlin(input)
        return input


def main():
    device = "cpu"
    dtype = torch.float32
    to_tensor = lambda x: torch.tensor(x, dtype=dtype, device=device)
    # load data ================================================================
    raw_data = load_data()  # list of dicts {X: ..., Y: ...}
    # extract sequences of length 250 from each dict in raw_data and append them to a single list
    moving_horizon = True
    X_sequences = []
    Y_sequences = []
    for d in raw_data[:1]:
        if moving_horizon:
            for i in range(len(d["X"]) - 250):
                X_sequences.append(d["X"][i : i + 250])
                Y_sequences.append(d["Y"][i : i + 250])
        else:
            for i in range(0, len(d["X"]), 250):
                X_sequences.append(d["X"][i : i + 250])
                Y_sequences.append(d["Y"][i : i + 250])

    # split data into train, validation and test sets
    idx = list(range(len(X_sequences)))
    random.shuffle(idx)
    train_idx = idx[: int(0.8 * len(idx))]
    val_idx = idx[int(0.8 * len(idx)) : int(0.9 * len(idx))]
    test_idx = idx[int(0.9 * len(idx)) :]
    train_X = torch.stack([to_tensor(X_sequences[i]) for i in train_idx])
    train_Y = torch.stack([to_tensor(Y_sequences[i]) for i in train_idx])
    val_X = torch.stack([to_tensor(X_sequences[i]) for i in val_idx])
    val_Y = torch.stack([to_tensor(Y_sequences[i]) for i in val_idx])
    test_X = torch.stack([to_tensor(X_sequences[i]) for i in test_idx])
    test_Y = torch.stack([to_tensor(Y_sequences[i]) for i in test_idx])

    # create datasets and dataloaders ==============================================================
    train_data = data.TensorDataset(train_X, train_Y)
    val_data = data.TensorDataset(val_X, val_Y)
    test_data = data.TensorDataset(test_X, test_Y)
    train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size=1, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=1, shuffle=True)

    # build model ==============================================================
    nx = train_X.shape[2]
    ny = train_Y.shape[2]
    estim = GRUVE(nx, ny, nh=64, nl=1, dropout=0.075, nonlin=nn.LeakyReLU)
    estim = estim.to(device)
    batch_X = train_X[:1]
    batch_Y = train_Y[:1]
    estimation = estim(batch_Y)
    print(
        "batch_X.shape={}, batch_Y.shape={}, estimation.shape={}".format(
            batch_X.shape, batch_Y.shape, estimation.shape
        )
    )

    # train model with classical torch minibatching ==============================================================
    optimizer = torch.optim.AdamW(estim.parameters(), lr=0.0005)
    epochs = 100
    loss = nn.MSELoss()
    loss_value = torch.tensor([0.0])
    best_loss_value = torch.tensor([torch.inf])
    for epoch in range(epochs):
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            optimizer.zero_grad()
            estimation = estim(batch_Y)
            loss_value = loss(estimation[:, 100:, :], batch_X[:, 100:, :])
            loss_value.backward()
            optimizer.step()
        if loss_value < best_loss_value:
            best_loss_value = loss_value
            # save model
            torch.save(estim.state_dict(), "best_model.pt")
            # if epoch % 100 == 0:
        print(
            "epoch {}: current RMSE={}, best RMSE={}".format(
                epoch, loss_value.item() ** 0.5, best_loss_value.item() ** 0.5
            )
        )

    print(
        "final RMSE: ",
        loss_value.item() ** 0.5,
        "best RMSE: ",
        best_loss_value.item() ** 0.5,
    )
    # save model ==============================================================
    torch.save(estim.state_dict(), "final_model.pt")


if __name__ == "__main__":
    # main_neuromancer()
    main()
