import torch
import numpy as np
from icecream import ic

def check_autodiff_convergence_order(func: callable, inputs: list, base_eps=1e-06):
    delta_inputs = [torch.rand_like(input) for input in inputs]
    delta_inputs_norm = sum([torch.sum(delta_input ** 2) for delta_input in delta_inputs])
    delta_inputs_norm = torch.sqrt(delta_inputs_norm)
    delta_inputs = [delta_input / delta_inputs_norm for delta_input in delta_inputs]
    errors = []
    epsilons = [base_eps / 2 ** i for i in range(-5, 6)]
    coeffs = None
    for eps in epsilons:
        inputs0 = [input.detach().clone() + eps * delta_input for (input, delta_input) in zip(inputs, delta_inputs)]
        inputs1 = [input.detach().clone() - eps * delta_input for (input, delta_input) in zip(inputs, delta_inputs)]
        inputs0 = [input.requires_grad_() for input in inputs0]
        inputs1 = [input.requires_grad_() for input in inputs1]
        outputs0 = func(*inputs0)
        if isinstance(outputs0, torch.Tensor):
            outputs0 = [outputs0]
        if coeffs is None:
            coeffs = [torch.rand_like(output) for output in outputs0]
        loss0 = sum([(coeff * output).sum() for (coeff, output) in zip(coeffs, outputs0)])
        loss0.backward()
        outputs1 = func(*inputs1)
        if isinstance(outputs1, torch.Tensor):
            outputs1 = [outputs1]
        loss1 = sum([(coeff * output).sum() for (coeff, output) in zip(coeffs, outputs1)])
        loss1.backward()
        grads0 = [input.grad for input in inputs0]
        grads1 = [input.grad for input in inputs1]
        grad_error = (loss0 - loss1) / eps - sum([torch.sum(delta_input * (grad0 + grad1)) for (delta_input, grad0, grad1) in zip(delta_inputs, grads0, grads1)])
        errors.append(abs(grad_error.item()))
    print(errors)
    for i in range(1, len(errors)):
        print(f'convergence_order: {np.log(errors[i - 1] / errors[i]).item() / np.log(2)}')

def check_autodiff(func: callable, inputs: list, eps=1e-06):
    """
    Check the gradient of a function using finite differences.
    """
    delta_inputs = [torch.rand_like(input) for input in inputs]
    delta_inputs_norm = sum([torch.sum(delta_input ** 2) for delta_input in delta_inputs])
    delta_inputs_norm = torch.sqrt(delta_inputs_norm)
    delta_inputs = [delta_input / delta_inputs_norm for delta_input in delta_inputs]
    inputs0 = [input.detach().clone() + eps * delta_input for (input, delta_input) in zip(inputs, delta_inputs)]
    inputs1 = [input.detach().clone() - eps * delta_input for (input, delta_input) in zip(inputs, delta_inputs)]
    inputs0 = [input.requires_grad_() for input in inputs0]
    inputs1 = [input.requires_grad_() for input in inputs1]
    outputs0 = func(*inputs0)
    if isinstance(outputs0, torch.Tensor):
        outputs0 = [outputs0]
    coeffs = [torch.rand_like(output) for output in outputs0]
    loss0 = sum([(coeff * output).sum() for (coeff, output) in zip(coeffs, outputs0)])
    loss0.backward()
    outputs1 = func(*inputs1)
    if isinstance(outputs1, torch.Tensor):
        outputs1 = [outputs1]
    loss1 = sum([(coeff * output).sum() for (coeff, output) in zip(coeffs, outputs1)])
    loss1.backward()
    grads0 = [input.grad for input in inputs0]
    grads1 = [input.grad for input in inputs1]
    grad_error = (loss0 - loss1) / eps - sum([torch.sum(delta_input * (grad0 + grad1)) for (delta_input, grad0, grad1) in zip(delta_inputs, grads0, grads1)])
    print(f'\x1b[91mgrad_error: {grad_error.item()}\x1b[0m')
    print('finite_diff:', (loss0 - loss1).item() / eps)
    print('analytical:', sum([torch.sum(delta_input * (grad0 + grad1)) for (delta_input, grad0, grad1) in zip(delta_inputs, grads0, grads1)]).item())
    return grad_error
import warp as wp

def check_grad(energy: callable, grad: callable, x: torch.Tensor, delta: torch.Tensor, eps=1e-08):
    """
    Check the gradient of a function using finite differences.
    """
    delta_norm = torch.sqrt(torch.sum(delta ** 2))
    delta = delta / delta_norm
    x0 = x.detach().clone() + eps * delta
    x1 = x.detach().clone() - eps * delta
    e0 = energy(x0)
    e1 = energy(x1)
    grad0 = grad(x0)
    grad1 = grad(x1)
    grad_error = (e0 - e1) / eps - torch.sum(delta * (grad0 + grad1))
    print(f'\x1b[91mgrad_error: {grad_error.item()}\x1b[0m')
    print('finite_diff:', (e0 - e1).item() / eps)
    print('analytical:', torch.sum(delta * (grad0 + grad1)).item())
    return grad_error

def check_jacobian(grad: callable, hess_mul: callable, x: torch.Tensor, delta: torch.Tensor, eps=1e-08):
    """
    Check the jacobian of a function using finite differences.
    """
    delta_norm = torch.sqrt(torch.sum(delta ** 2))
    delta = delta / delta_norm
    x0 = x.detach().clone() + eps * delta
    x1 = x.detach().clone() - eps * delta
    grad0 = grad(x0)
    grad1 = grad(x1)
    hess0 = hess_mul(x0, delta)
    hess1 = hess_mul(x1, delta)
    jacobian_error = ((grad0 - grad1) / eps - (hess0 + hess1)).abs().mean()
    print(f'\x1b[91mjacobian_error: {jacobian_error.item()}\x1b[0m')
    print('finite_diff:', (grad0 - grad1).abs().mean().item() / eps)
    print('analytical:', (hess0 + hess1).abs().mean().item())
    return jacobian_error

def check_jacobian2(grad: callable, hess_mul: callable, x, eps=1e-06):
    hess_cols = []
    hess_fd_cols = []
    for i in range(x.numel()):
        delta = torch.zeros_like(x)
        delta.view(-1)[i] = 1
        hess_col = hess_mul(x, delta)
        grad0 = grad(x + eps * delta)
        grad1 = grad(x - eps * delta)
        hess_finite_diff = (grad0 - grad1) / (2 * eps)
        hess_cols.append(hess_col.view(-1))
        hess_fd_cols.append(hess_finite_diff.view(-1))
    hess_fd = torch.stack(hess_fd_cols, dim=1)
    hess = torch.stack(hess_cols, dim=1)
    print(hess_fd.cpu().numpy())
    print()
    print(hess.cpu().numpy())
    print()
    print((hess - hess_fd).abs().cpu().numpy())