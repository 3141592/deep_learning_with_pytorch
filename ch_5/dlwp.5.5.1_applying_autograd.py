from pprint import pprint
import torch

print()
pprint("5.5.1 Computing the gradient automatically")


pprint("APPLYING AUTOGRAD")
def model(t_u, w, b):
  print()
  print(f"w * t_u + b: {w * t_u + b}")
  return w * t_u + b

def loss_fn(t_p, t_c):
  print(f"t_c: {t_c}")
  print(f"t_p: {t_p}")
  squared_diffs = (t_p - t_c)**2
  return squared_diffs.mean()

params = torch.tensor([1.0, 0.0], requires_grad=True)
print()
pprint(f"params: {params}")

pprint("USING THE GRAD ATTRIBUTE")
pprint(f"params.grad: {params.grad}")

print()
pprint("5.2.2 Gathering some data")
pprint("From dlwpt/p1ch5/1_parameter_estimation.ipynb")
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

loss = loss_fn(model(t_u, *params), t_c)
loss.backward()

print()
pprint(f"params.grad: {params.grad}")

