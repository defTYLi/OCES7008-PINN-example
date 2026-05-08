import math
import torch
import torch.nn as nn


# =========================================================
# 1. Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# 2. Simple MLP: input (x, y), output psi(x, y)
# =========================================================
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32, out_dim=1, num_hidden=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================================================
# 3. Autograd helper
# =========================================================
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]


# =========================================================
# 4. Exact solution and forcing term
#    psi_true = sin(pi x) sin(pi y)
#    -laplacian(psi_true) = 2*pi^2*sin(pi x)sin(pi y)
# =========================================================
def psi_true(x, y):
    return torch.sin(math.pi * x) * torch.sin(math.pi * y)

def forcing(x, y):
    return 2.0 * (math.pi ** 2) * torch.sin(math.pi * x) * torch.sin(math.pi * y)


# =========================================================
# 5. PDE loss
#    PDE: -(psi_xx + psi_yy) = f(x, y)
# =========================================================
def pde_loss(model, n_interior=2000):
    x = torch.rand(n_interior, 1, device=device, requires_grad=True)
    y = torch.rand(n_interior, 1, device=device, requires_grad=True)

    xy = torch.cat([x, y], dim=1)
    psi = model(xy)

    psi_x = grad(psi, x)
    psi_y = grad(psi, y)
    psi_xx = grad(psi_x, x)
    psi_yy = grad(psi_y, y)

    residual = -(psi_xx + psi_yy) - forcing(x, y)
    return torch.mean(residual**2)


# =========================================================
# 6. Boundary loss
#    psi = 0 on all boundaries
# =========================================================
def boundary_loss(model, n_boundary=500):
    # x = 0, x = 1, y random
    y1 = torch.rand(n_boundary, 1, device=device)
    x0 = torch.zeros_like(y1, device=device)
    x1 = torch.ones_like(y1, device=device)

    # y = 0, y = 1, x random
    x2 = torch.rand(n_boundary, 1, device=device)
    y0 = torch.zeros_like(x2, device=device)
    y1_top = torch.ones_like(x2, device=device)

    xy_left   = torch.cat([x0, y1], dim=1)
    xy_right  = torch.cat([x1, y1], dim=1)
    xy_bottom = torch.cat([x2, y0], dim=1)
    xy_top    = torch.cat([x2, y1_top], dim=1)

    psi_left   = model(xy_left)
    psi_right  = model(xy_right)
    psi_bottom = model(xy_bottom)
    psi_top    = model(xy_top)

    loss = (
        torch.mean(psi_left**2)
        + torch.mean(psi_right**2)
        + torch.mean(psi_bottom**2)
        + torch.mean(psi_top**2)
    )
    return loss


# =========================================================
# 7. Optional evaluation loss against exact solution
# =========================================================
@torch.no_grad()
def eval_mse(model, n_eval=100):
    x = torch.linspace(0, 1, n_eval, device=device)
    y = torch.linspace(0, 1, n_eval, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    pred = model(xy).reshape(n_eval, n_eval)
    true = psi_true(X, Y)

    mse = torch.mean((pred - true) ** 2).item()
    return mse


# =========================================================
# 8. Training
# =========================================================
def train(model, epochs=5000, lr=1e-3, lambda_bc=1.0, print_every=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        loss_pde = pde_loss(model)
        loss_bc = boundary_loss(model)
        loss = loss_pde + lambda_bc * loss_bc

        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            mse = eval_mse(model, n_eval=80)
            print(
                f"Epoch {epoch:5d} | "
                f"Total: {loss.item():.6e} | "
                f"PDE: {loss_pde.item():.6e} | "
                f"BC: {loss_bc.item():.6e} | "
                f"Eval MSE: {mse:.6e}"
            )


# =========================================================
# 9. Run
# =========================================================
model = MLP().to(device)
train(model, epochs=5000, lr=1e-3, lambda_bc=1.0, print_every=500)