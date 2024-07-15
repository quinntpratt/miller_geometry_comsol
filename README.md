## COMSOL 2DFW with Miller geometry
Python program to create inputs for a two dimensional full-wave COMSOL simulation based on Miller flux surfaces.
COMSOL simulations require the density (ne) and the magnetic field components (Bx, By, Bz). 

The user describes the plasma with, 
- Miller parameters (``R0, delta, kappa``)
- Toroidal field function (``I = R*B_tor``)
- Saftey factor profile (``q``)
- Density profile (``ne``)

The user must also specify a rectangular domain to create the high-resolution grid of COMSOL inputs. 

The resulting flux surfaces (and density profile) are not necessarily solutions to the Grad-Shafranov equation. 

## Example plots
This plot illustrates the flux surfaces and calculated poloidal magnetic field given the profiles input on the right side of the figure.
The red rectangle indicates an example COMSOL domain.
<p align="center">
  <img width="750" alt="image" src="https://github.com/user-attachments/assets/0877e774-5582-4556-91e5-955bc1f60b62">
</p>

This plot illustrates the mapped density profile as well as calculations of the midplane plasma cutoff frequencies.
In this case we use a simple piecewise-continuous linear density profile.
<p align="center">
  <img width="750" alt="image" src="https://github.com/user-attachments/assets/95bb51ea-1c75-420f-81c1-a0db7be162f4">
</p>
