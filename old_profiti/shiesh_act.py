import math
import torch
import torch.nn.functional as F


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
        self.log_cosh = LogCosh()

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
        Computes the log-Jacobian determinant for a transformation
        in a numerically stable and efficient way.
        """
        ax = self.a * x

        # log(cosh(ax)) is safe directly since |ax| <= 5
        log_cosh_ax = torch.log(torch.cosh(ax))

        # sinh(ax) is also stable in this range
        sinh_ax = torch.sinh(ax)

        # log(1 + slope^2 * sinh^2(ax))
        log_denominator = 0.5 * torch.log1p((self.slope * sinh_ax) ** 2)

        # numerator = e^{at}*cosh(ax)
        log_numerator = self.log_slope + log_cosh_ax

        return log_numerator - log_denominator

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
