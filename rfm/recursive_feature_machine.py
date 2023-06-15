try:
    from eigenpro2.models import KernelModel
    from eigenpro3.utils import accuracy
except ModuleNotFoundError:
    print(
        "`eigenpro2` not installed.. using torch.linalg.solve for training kernel model"
    )
import torch, numpy as np
from .kernels import laplacian_M, euclidean_distances_M
from tqdm import tqdm
import hickle


class RecursiveFeatureMachine(torch.nn.Module):
    def __init__(
        self, device=torch.device("cpu"), mem_gb=32, diag=False, centering=False, M_init_scheme='identity'
    ):
        super().__init__()
        self.M = None
        self.M_init_scheme = M_init_scheme # specifies how to initialize M
        self.model = None
        self.diag = diag  # if True, Mahalanobis matrix M will be diagonal
        self.centering = centering  # if True, update_M will center the gradients before taking an outer product
        self.device = device
        self.mem_gb = mem_gb

    def get_data(self, data_loader, batches=None):
        X, y = [], []
        cnt = 1
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # inputs = inputs.view(-1, inputs.shape[-1])
            # labels = labels.view(-1, 1)
            X.append(inputs)
            y.append(labels)
            if cnt >= batches:
                break
            cnt += 1
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        return X, y

    def update_M(self):
        raise NotImplementedError("Must implement this method in a subclass")

    def fit_predictor(self, centers, targets, **kwargs):
        self.centers = centers
        if self.M is None:
            if self.diag:
                self.M = torch.ones(centers.shape[-1], device=self.device)
            else:
                d = centers.shape[-1]
                if self.M_init_scheme == 'identity':
                    self.M = torch.eye(d, device=self.device)
                elif self.M_init_scheme == 'random':
                    self.M = torch.normal(mean=0.0, std=1/np.sqrt(d), size=(d, d), device=self.device)
                else:
                    raise ValueError(f"{self.M_init_scheme} is not valid. Options include `identity` and `random`.")
        if (len(centers) > 20_000) or self.fit_using_eigenpro:
            self.weights = self.fit_predictor_eigenpro_old(centers, targets, **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets)

    def fit_predictor_lstsq(self, centers, targets):
        return torch.linalg.solve(
            self.kernel(centers, centers)
            + 1e-3 * torch.eye(len(centers), device=centers.device),
            targets,
        )

    def fit_predictor_eigenpro_old(self, centers, targets, **kwargs):
        n_classes = 1 if targets.dim() == 1 else targets.shape[-1]
        self.model = KernelModel(self.kernel, centers, n_classes)
        _ = self.model.fit(centers, targets, mem_gb=self.mem_gb, **kwargs)
        return self.model.weight

    def fit_predictor_eigenpro(self, centers, targets, test_loader, **kwargs):
        """Uses new eigenpro3 to train kernel."""
        n_classes = 1 if targets.dim() == 1 else targets.shape[-1]
        self.model = KernelModel(
            n_classes, centers, self.kernel, X=centers, y=targets, devices=[self.device]
        )
        _ = self.model.fit(
            self.model.train_loaders, test_loader, score_fn=accuracy, epochs=10
        )
        return self.model.weights

    def predict(self, samples):
        return self.kernel(samples, self.centers) @ self.weights

    def fit(
        self,
        train_loader,
        test_loader,
        iters=3,
        name=None,
        reg=1e-3,
        method="lstsq",
        train_acc=False,
        loader=True,
        classif=True,
        callbacks=[],
        **kwargs,
    ):
        # if method=='eigenpro':
        #     raise NotImplementedError(
        #         "EigenPro method is not yet supported. "+
        #         "Please try again with `method='lstlq'`")
        self.fit_using_eigenpro = method.lower() == "eigenpro"
        # self.fit_using_eigenpro = False

        train_mses, test_mses = [], []

        if loader:
            print("Loaders provided")
            X_train, y_train = self.get_data(train_loader, batches=1)
            X_test, y_test = self.get_data(test_loader, batches=1)
        else:
            X_train, y_train = train_loader
            X_test, y_test = test_loader
        # X_train, y_train = train_loader
        # X_test, y_test = self.get_data(test_loader, batches=5)
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        for i in range(iters):
            self.fit_predictor(X_train, y_train, x_val=X_test, y_val=y_test, **kwargs)

            if classif:
                train_acc = self.score(X_train, y_train, metric="accuracy")
                print(f"Round {i}, Train Acc: {train_acc:.2f}%", end="\t")
                test_acc = self.score(X_test, y_test, metric="accuracy")
                print(f"Test Acc: {test_acc:.2f}%", end="\t")

            train_mse = self.score(X_train, y_train, metric="mse")
            print(f"Train MSE: {train_mse:.4f}", end="\t")
            test_mse = self.score(X_test, y_test, metric="mse")
            print(f"Test MSE: {test_mse:.4f}", end="\t")
            train_mses.append(train_mse.item())
            test_mses.append(test_mse.item())

            # see how close M is to identity
            identity_close = torch.linalg.norm(self.M - torch.eye(self.M.shape[0], device=self.device))
            print(f"Distance to Identity: {identity_close:.4f}", end="\t")

            # run the callbacks
            for callback in callbacks:
                label, result = callback(self)
                print(f"{label}: {result:.4f}", end="\t")
            print()

            self.update_M(X_train)

            if name is not None:
                hickle.dump(M, f"saved_Ms/M_{name}_{i}.h")

        self.fit_predictor(X_train, y_train, x_val=X_test, y_val=y_test, **kwargs)
        final_mse = self.score(X_test, y_test, metric="mse")
        print(f"Final MSE: {final_mse:.4f}")
        if classif:
            final_test_acc = self.score(X_test, y_test, metric="accuracy")
            print(f"Final Test Acc: {final_test_acc:.2f}%")

        return train_mses, test_mses

    def score(self, samples, targets, metric="mse"):
        preds = self.predict(samples)
        if metric == "accuracy":
            # TODO: replace with torchmetrics
            return (1.0 * (targets == (preds >= 0.5))).mean() * 100.0
        elif metric == "mse":
            return (targets - preds).pow(2).mean()


class LaplaceRFM(RecursiveFeatureMachine):
    def __init__(self, bandwidth=1.0, **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel = lambda x, z: laplacian_M(
            x, z, self.M, self.bandwidth
        )  # must take 3 arguments (x, z, M)

    def update_M(self, samples):
        K = self.kernel(samples, self.centers)

        dist = euclidean_distances_M(samples, self.centers, self.M, squared=False)
        dist = torch.where(
            dist < 1e-10, torch.zeros(1, device=dist.device).float(), dist
        )

        K = K / dist
        K[K == float("Inf")] = 0.0

        p, d = self.centers.shape
        p, c = self.weights.shape
        n, d = samples.shape

        samples_term = (K @ self.weights).reshape(n, c, 1)  # (n, p)  # (p, c)

        if self.diag:
            centers_term = (
                K  # (n, p)
                @ (
                    self.weights.view(p, c, 1) * (self.centers * self.M).view(p, 1, d)
                ).reshape(
                    p, c * d
                )  # (p, cd)
            ).view(
                n, c, d
            )  # (n, c, d)

            samples_term = samples_term * (samples * self.M).reshape(n, 1, d)

        else:
            G = (
                K  # (n, p)
                @ (
                    self.weights.view(p, c, 1) * (self.centers @ self.M).view(p, 1, d)
                ).reshape(
                    p, c * d
                )  # (p, cd)
            ).view(
                n, c, d
            )  # (n, c, d)

            samples_term = samples_term * (samples @ self.M).reshape(n, 1, d)

        G = (G - samples_term) / self.bandwidth  # (n, c, d)

        if self.centering:
            G = G - G.mean(0)  # (n, c, d)

        if self.diag:
            torch.einsum("ncd, ncd -> d", G, G) / len(samples)
        else:
            self.M = torch.einsum("ncd, ncD -> dD", G, G) / len(samples)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # define target function
    def fstar(X):
        return torch.cat([(X[:, 0] > 0)[:, None], (X[:, 1] < 0.1)[:, None]], axis=1)

    # create low rank data
    n = 4000
    d = 100
    np.random.seed(0)
    X_train = torch.from_numpy(np.random.normal(scale=0.5, size=(n, d)))
    X_test = torch.from_numpy(np.random.normal(scale=0.5, size=(n, d)))

    y_train = fstar(X_train).double()
    y_test = fstar(X_test).double()

    model = LaplaceRFM(bandwidth=1.0, diag=False, centering=False)
    model.fit(
        (X_train, y_train), (X_test, y_test), loader=False, iters=5, classif=False
    )
