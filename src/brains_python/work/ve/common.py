import numpy as np


def stats_errs(x, y):
    """returns average error, standard deviation of errors, max error, RMSE"""
    errs = x - y
    npmax = np.max(np.abs(errs))
    return (
        np.mean(errs),
        np.std(errs),
        np.max(np.abs(errs)),
        np.sqrt(np.mean(errs**2)),
    )


def print_stats_errs(x, y, name, relative=False):
    mean, std, max_err, rmse = stats_errs(x, y)
    # if relative:
    #     mean /= np.mean(y)
    #     std /= np.mean(y)
    #     max_err /= np.mean(y)
    #     rmse /= np.mean(y)
    # print(
    #     f"error statistics for {name}: mean = {mean}, std = {std}, max = {max_err}, RMSE = {rmse}"
    # )

    print(f"{name}: RMSE = {rmse:.3f}, percentage={100*rmse/np.max(np.abs(y)):.3f}")
