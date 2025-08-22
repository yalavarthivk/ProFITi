import math
import torch
import torch.nn.functional as F
import pdb


class Shiesh:
    """
    Shiesh activation function, which combines the inverse hyperbolic sine (arcsinh)
    and sinh functions to create a smooth activation function with a learnable slope.

    Parameters:
    -----------
    t : float, optional, default=1.0
        The time or scaling factor, used to scale the slope of the function.

    a : float, optional, default=1.0
        A learnable parameter that scales the input of the activation function.
    """

    def __init__(self, t: float = 1.0, a: float = 1.0):
        """
        Initializes the Shiesh activation function with the given parameters.

        Arguments:
        ----------
        t : float, optional, default=1.0
            The time or scaling factor used to calculate the slope of the activation.

        a : float, optional, default=1.0
            The parameter for the input of the activation function.
        """
        self.t = t
        self.a = a
        self.slope = math.exp(a * t)
        self.log_slope = a * t

    def shiesh_fn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Shiesh activation function for the given input `x`.

        The Shiesh function is defined as:
        f(x) = arcsinh(slope * sinh(a * x)) / a

        Arguments:
        ----------
        x : torch.Tensor
            Input tensor to which the activation is applied.

        Returns:
        --------
        torch.Tensor
            Output tensor after applying the Shiesh activation.
        """
        shiesh_x = torch.arcsinh(self.slope * torch.sinh(self.a * x)) / self.a
        return shiesh_x

    def shiesh_log_jac(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-Jacobian determinant for a transformation in a numerically
        stable way, with debugging checks for NaN/Inf in both forward and backward.
        """

        # Forward computations with checks
        if x.isinf().any() or x.isnan().any():
            pdb.set_trace()
            print("[DEBUG: FORWARD] Input x has NaN/Inf")
            print("  min:", x.min().item(), "max:", x.max().item())
            bad_idx = torch.nonzero(torch.isnan(x) | torch.isinf(x), as_tuple=True)
            print("  offending values:", x[bad_idx])

        cosh_ax = torch.cosh(self.a * x)
        if torch.isnan(cosh_ax).any() or torch.isinf(cosh_ax).any():
            pdb.set_trace()
            print("[DEBUG: FORWARD] cosh_ax has NaN/Inf")
            print("  min:", cosh_ax.min().item(), "max:", cosh_ax.max().item())
            bad_idx = torch.nonzero(
                torch.isnan(cosh_ax) | torch.isinf(cosh_ax), as_tuple=True
            )
            print("  offending values:", cosh_ax[bad_idx])
        # check_tensor("cosh_ax", cosh_ax)

        sinh_ax = torch.sinh(self.a * x)
        if torch.isnan(sinh_ax).any() or torch.isinf(sinh_ax).any():
            pdb.set_trace()
            print("[DEBUG: FORWARD] sinh_ax has NaN/Inf")
            print("  min:", sinh_ax.min().item(), "max:", sinh_ax.max().item())
            bad_idx = torch.nonzero(
                torch.isnan(sinh_ax) | torch.isinf(sinh_ax), as_tuple=True
            )
            print("  offending values:", sinh_ax[bad_idx])
        # check_tensor("sinh_ax", sinh_ax)

        log_numerator = self.a * self.t + torch.log(cosh_ax + 1e-12)
        if torch.isnan(log_numerator).any() or torch.isinf(log_numerator).any():
            pdb.set_trace()
            print("[DEBUG: FORWARD] log_numerator has NaN/Inf")
            print(
                "  min:", log_numerator.min().item(), "max:", log_numerator.max().item()
            )
            bad_idx = torch.nonzero(
                torch.isnan(log_numerator) | torch.isinf(log_numerator), as_tuple=True
            )
            print("  offending values:", log_numerator[bad_idx])
        # check_tensor("log_numerator", log_numerator)

        val = (self.slope * sinh_ax) ** 2
        if torch.isnan(val).any() or torch.isinf(val).any():
            pdb.set_trace()
            print("[DEBUG: FORWARD] val has NaN/Inf")
            print("  min:", val.min().item(), "max:", val.max().item())
            bad_idx = torch.nonzero(torch.isnan(val) | torch.isinf(val), as_tuple=True)
            print("  offending values:", val[bad_idx])
        # check_tensor("val = (slope * sinh_ax)^2", val)

        log_denominator = 0.5 * torch.log1p(val + 1e-12)
        if torch.isnan(log_denominator).any() or torch.isinf(log_denominator).any():
            pdb.set_trace()
            print("[DEBUG: FORWARD] log_denominator has NaN/Inf")
            print(
                "  min:",
                log_denominator.min().item(),
                "max:",
                log_denominator.max().item(),
            )
            bad_idx = torch.nonzero(
                torch.isnan(log_denominator) | torch.isinf(log_denominator),
                as_tuple=True,
            )
            print("  offending values:", log_denominator[bad_idx])
        # check_tensor("log_denominator", log_denominator)

        out = log_numerator - log_denominator
        # check_tensor("output", out)

        return out

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the Shiesh activation function with a special handling
        for large values of `x`.

        For values of `x` where |x| > 5, the activation simplifies to a linear
        transformation of the form `f(x) = x + sign(x)` to avoid extreme computational
        values that might occur for large inputs.

        For smaller a value, threshold can be increased.

        Arguments:
        ----------
        x : torch.Tensor
            Input tensor.

        Returns:
        --------
        tuple
            - activation_out (torch.Tensor): The result of the activation function.
            - activation_log_jac (torch.Tensor): Logarithmic Jacobian for the input tensor.
        """
        large_inds = torch.abs(x) > 5
        # Mask out large values before applying shiesh_log_jac
        safe_x = torch.where(large_inds, torch.zeros_like(x), x)

        activation_out = torch.where(
            large_inds, x + self.t * torch.sign(x), self.shiesh_fn(safe_x)
        )
        activation_log_jac = torch.where(
            large_inds,
            0.0,  # Logarithmic Jacobian is zero for the linear part
            self.shiesh_log_jac(safe_x),
        )
        return activation_out, activation_log_jac

    def __call__(self, x: torch.Tensor) -> tuple:
        """
        Makes the class callable, forwarding to the forward method.

        Arguments:
        ----------
        x : torch.Tensor
            Input tensor.

        Returns:
        --------
        tuple
            shiesh_fn output and log determinent of the jacobian;
            the Output from the forward method.
        """
        return self.forward(x)
