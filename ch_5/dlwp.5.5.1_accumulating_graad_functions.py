from pprint import pprint
import torch

print()
pprint("5.5.1 Computing the gradient automatically")

print()
pprint("ACCUMLATING GRAD FUNCTIONS")
def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            # This could be done at any point in the loop prior to calling loss.backward().
            params.grad.zero_()

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()

        with torch.no_grad():
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params

print()
pprint("5.2.2 Gathering some data")
pprint("From dlwpt/p1ch5/1_parameter_estimation.ipynb")
pprint("t_c values are temperatures in celsius")
pprint("t_u measurements in unknown units")

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
# Normalize t_u
t_un = 0.1 * t_u

print()
pprint("Training loop")
params = training_loop(
            n_epochs = 5000,
            learning_rate = 1e-2,
            params = torch.tensor([1.0, 0.0], requires_grad=True),
            t_u = t_un,
            t_c = t_c)

print()
pprint(f"params: {params}")

