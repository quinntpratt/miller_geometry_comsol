#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:55:06 2024

@author: Quinn Pratt

Description
-----------
This program creates an input file for 2D full-wave simulations using the 
COMSOL RF module.
The Miller Geometry [1] is used to create non-trivial flux surfaces.

The magnetic field is specified by the toroidal field function, I = RB_\phi, 
along with the saftey factor, q.
 
The density is specified with a radial profile and is mapped onto the 2D 
flux surfaces.

The resulting magnetic equilibrium is NOT guarenteed to be a solution to the 
Grad-Shafronov equation.

The magnetic field components and the density are written to a file.

References
----------
[1] Physics of Plasmas 5, 973 (1998); https://doi.org/10.1063/1.872666
[2] Y. Hu, "Notes on tokamak equilibrium"
    https://youjunhu.github.io/research_notes/tokamak_equilibrium_htlatex/tokamak_equilibrium.html#x1-48001r182

All rights reserved
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from scipy.interpolate import LinearNDInterpolator
import copy

# Change the path and filename,
write_file = False
path = "/Users/quinnpratt/Documents/COMSOL60/simplified_dbs_emw_v2/"
fname = "comsol_dbs_emw_input.csv"
f_rf = 72.5 # [GHz]

class MillerGeom:
    def __init__(self, r, theta, R0=1.7, delta=0.17, kappa=1.35):
        """
        Initialize a MillerGeom object.
        Local axisymmetric, up/down symmetric, flux surface(s) parameterized by
        R0, delta, and kappa.
        Radial profiles of R0, delta, and kappa are also supported.

        Parameters
        ----------
        r : 1d np.ndarray
            Minor radius coordinate \psi.
        theta : 1d np.ndarray
            Poloidal angle coordinate \theta.
        R0 : float or 1d np.ndarray, optional
            Flux surface center. The default is 1.7.
        delta : float or 1d np.ndarray, optional
            Triangularity. The default is 0.16.
        kappa : float or 1d np.ndarray, optional
            Elongation. The default is 1.35.

        """
        self.r = r
        self.theta = theta
        self.R0=R0
        self.kappa=kappa
        self.delta=delta
        mkw = dict(R0=R0, kappa=kappa, delta=delta)
        # make sure they're numpy arrays of len(r)
        for k,v in mkw.items():
            v = np.atleast_1d(v)
            if len(v) == 1:
                v = v[0]*np.ones(len(r))
            else:
                if len(v) != len(r):
                    print(f"ERROR: length of Miller param. {k} array must match len(r)")
                    return
            setattr(self, k, v)
        
        # Additional shape management: (nth, nr)
        r, R0, delta, kappa = np.atleast_2d(r, R0, delta, kappa ) # (1, nr)
        theta = np.atleast_2d(theta).T # (nth, 1)
        
        # Compute flux surfaces: results are (nth, nr)
        x = np.arcsin(delta)
        self.Rs = R0 + r*np.cos(theta + x*np.sin(theta))
        self.Zs = kappa*r*np.sin(theta)
        
    def gradient(self):
        """
        Compute the gradient components, dR/dr, dR/d\theta, dZ/dr, dZ/d\theta.
        All outputs are (theta, r) shape 2d np.arrays.
        
        Because we are using the Miller parameterizations for R, Z - the use 
        of np.gradient can be replaced by analytic expressions in the future.
        
        Returns
        -------
        dRdt : 2d np.array of shape (theta, r)
                d(Rs)/d\theta
        dRdr : 2d np.array of shape (theta, r)
                d(Rs)/dr
        dZdt : 2d np.array of shape (theta, r)
                d(Zs)/d\theta
        dZdr : 2d np.array of shape (theta, r)
                d(Zs)/dr
        """
        self.dRdt, self.dRdr = np.gradient(self.Rs, self.theta, self.r)
        self.dZdt, self.dZdr = np.gradient(self.Zs, self.theta, self.r)
        return self.dRdt, self.dRdr, self.dZdt, self.dZdr
    
    def calc_metric_terms(self):
        """
        Compute useful metric terms (used in other methods).

        Returns
        -------
        J : 2d np.array of shape (theta, r)
            Jacobian of R(r,theta) ; Z(r,theta) surfaces.
        gr_gt : 2d np.array of shape (theta, r)
            $\nabla r \cdot \nabla \theta$
        gr2 : 2d np.array of shape (theta, r)
            $|\nabla r|^2$.

        """
        # Returns useful metric terms,
        R = self.Rs
        dRdt, dRdr, dZdt, dZdr = self.gradient()
        # 1. The Jacobian determinant,
        self.J = R*( dRdt*dZdr - dRdr*dZdt )
        # 2. grad(r) \cdot grad(\theta),
        self.gr_gt = -R**2/self.J**2 * ( dZdt*dZdr + dRdt*dRdr )
        # 3. |grad(r)|**2
        self.gr2 = R**2/self.J**2 * ( dZdt**2 + dRdt**2 )
        return self.J, self.gr_gt, self.gr2
    
    def calc_psiprime(self, I, q):
        """
        Calculate the (radial) derivative of the poloidal magnetic flux.
        NOTE: this method assumes the MillerGeom object covers theta in [0,2pi) 

        Parameters
        ----------
        I : 1d np.array with len(r)
            Toroidal field flux function, I = R*B_phi.
        q : 1d np.array with len(r)
            Saftey factor.

        Returns
        -------
        psiprime : 1d np.array with len(r)
            Radial derivative of the poloidal magnetic flux.

        """
        
        self.calc_metric_terms()
        self.psiprime = -1/(2*np.pi)*I/q* np.trapz(self.J/self.Rs**2, self.theta, axis=0)
        return self.psiprime
    
    def calc_Bp(self, I, q):
        """
        Calculate the poloidal magnetic field.

        Parameters
        ----------
        I : 1d np.array with len(r)
            Toroidal field flux function, I = R*B_phi.
        q : 1d np.array with len(r)
            Saftey factor.

        Returns
        -------
        Bp : list of 2d np.array with shapes (theta, r)
            Poloidal magnetic field vector: [Bp_R, Bp_Z] components.

        """
        
        psiprime = self.calc_psiprime(I, q)
        
        Z_comp = self.gr_gt * self.dRdt + self.gr2*self.dRdr
        R_comp = self.gr_gt * self.dZdt + self.gr2*self.dZdr
        
        Bp_R = psiprime/self.Rs * -1*R_comp
        Bp_Z = psiprime/self.Rs * Z_comp
        self.Bp = [Bp_R, Bp_Z]
        
        return self.Bp
        

# %% Flux Surface Shape
# This section calculates the flux surface shapes.

# Definition of the full [0, 2pi) grid,
Nth, Nr = 201, 101
r = np.linspace(0.4, 0.7, Nr) # midplane half-diameter coordinate.
theta = np.linspace(0, 2*np.pi, Nth) # poloidal angle coordinate.

# Flux-surface kwargs can be scalars or functions of 'r',
fs_kw = dict(R0=1.7, delta=0.16, kappa=1.35)
# Linear functions of 'r', 
fs_kw = dict(R0=-0.14*r + 1.78,
             delta=1.26*r - 0.41,
             kappa=1.12*r + 0.84,)

# Create the MillerGeom object,
MG = MillerGeom(r, theta, **fs_kw)
Rs, Zs = MG.Rs, MG.Zs # Flux surfaces.

# Plot the flux surfaces and the Miller params,
fig = plt.figure("Miller Flux Surfaces",(10, 6))
gs = fig.add_gridspec(5,2,)
ax = fig.add_subplot(gs[:,0])
axs = []
for i in range(5):
    axs += [fig.add_subplot(gs[i,1])]
skip = 10
ax.plot(Rs[:,::skip], Zs[:,::skip], "k", lw=0.5)
ax.set_aspect("equal")
ax.set_ylabel("Z [m]")
ax.set_xlabel("R [m]")

for i, k in enumerate(fs_kw.keys()):
    axs[i].plot(r, getattr(MG,k))
    axs[i].set_ylabel(k)
axs[-1].set_xlabel("r")
    
# %% Magnetic field
# This section calculates the magnetic field provided I and q.

# The toroidal field function -- must me len(r)
I = -3.08*np.ones(len(r)) # I(psi) = R*B_tor [m*T]
Btor_mag = I/Rs
# Saftey factor,
q = 2.8*np.ones(len(r)) # constant
q = -17.2*r + 5.1 # linear
q = -65.5*r**2 + 48.5*r - 11.16 # quadratic

# Calculate the poloidal magnetic field,
Bp = MG.calc_Bp(I, q)
Bp_R = Bp[0] # [T]
Bp_Z = Bp[1] # [T] 
Bp_mag = np.sqrt(Bp_R**2 + Bp_Z**2) # [T]

# Plot the poloidal magnetic field,
c = ax.contourf(Rs, Zs, Bp_mag)
fig.colorbar(c, ax=ax)
ax.quiver(Rs[::skip,::skip], Zs[::skip,::skip],
          Bp_R[::skip,::skip], Bp_Z[::skip,::skip])
# Also show the I(psi) and q(psi)
axs[3].plot(r, I)
axs[3].set_ylabel(r"$I = RB_\phi$")
axs[4].plot(r, q)
axs[4].set_ylabel(r"q")
fig.tight_layout()
# Additional plot of the toroidal and poloidal field at the midplane
# for comparison with geqdsk file.
fig, axs = plt.subplots(2,1,sharex=True)
axs[0].plot(r, Btor_mag[0,:])
axs[1].plot(r, Bp_mag[0,:])
for a in axs:
    a.axvline(0.46, ls='--',color="r")
axs[0].set_ylabel(r"$B_\phi$ [T] (midplane)")
axs[1].set_ylabel(r"$B_p$ [T] (midplane)")
axs[1].set_xlabel("r (rmin)")

# %% Density profile,
# Define a density profile vs. rmin, then map it onto the flux surfaces.

r_sep = 0.618 # Edge r_min (~separatrix)
ne_sep = 1E-6 # Density value at r=r_sep (will also fill SOL)
ne_core = 12.0 # Density value at min(r)
# Piecewise-continuous linear density profile,
m = (ne_sep - ne_core)/r_sep
ne = np.piecewise(r, [r <= r_sep, r > r_sep ],
                  [lambda r:  m*(r - r_sep) + ne_sep, ne_sep],
                  )
# Easy way to use broadcasting to map ne onto flux surfaces,
ne_fs = ne + 0*Rs 

# Compute fundamental frequencies,
B_mag = np.sqrt(Bp_mag**2 + Btor_mag**2)[0,:] # midplane [T]
fce = 28*B_mag # [GHz]
fpe = 8.98E3*np.sqrt(ne*1E13)/1E9 # [GHz]
frh = 0.5*(np.abs(fce) + np.sqrt(fce**2 + 4*fpe**2)) # [GHz]

# Plot the density and the cutoff frequencies,
fig = plt.figure("Miller Flux Surfaces - Density",(10, 6))
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[:,0])
axn, axf = fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,1])
ax1.contourf(Rs, Zs, ne_fs)
ax1.plot(Rs[:,::skip], Zs[:,::skip], "k", lw=0.5)
ax1.set_aspect("equal")
ax1.set_ylabel("Z [m]")
ax1.set_xlabel("R [m]")
axn.plot(r, ne)
axn.set_ylabel("Density [E19 1/m^3]")
axn.set_xlabel("r")
axf.plot(r, fpe, 'g-')
axf.plot(r, frh, 'k-')
axf.plot(r, fce, 'b--')
axf.plot(r, 2*fce, 'b--')
axf.set_ylabel("Freq. [GHz]")
axf.set_xlabel("r")
axf.axhline(f_rf, color='r',ls="-")

# %% COMSOL Domain setup.
# Define a box in the poloidal plane for COMSOL simulations.

# Lower-left corner,
Rc, Zc = 2.14, -0.11 # [m]
# Width and height of domain,
w, h = 0.195, 0.3075 # [m]

# Create a higher-resolution grid,
NRi, NZi = 256, 300
Ri = np.linspace(Rc, Rc + w, NRi)
Zi = np.linspace(Zc, Zc + h, NZi)
RRi, ZZi = np.meshgrid(Ri, Zi)
# Ravel into points for 2D interpolation,
points = np.array([Rs.flatten(),Zs.flatten()]).T # must be (NRi*NZi, 2)
# Linearly interpolate quantities onto the higher-res. grid.
# NOTE: regions out of the flux surfaces are filled with zeros/ne_sep.
nei = LinearNDInterpolator(points, ne_fs.flatten(), fill_value=ne_sep)(RRi, ZZi)
Bx =  LinearNDInterpolator(points, Bp_R.flatten(), fill_value=0.0)(RRi, ZZi)
By =  LinearNDInterpolator(points, Bp_Z.flatten(), fill_value=0.0)(RRi, ZZi)
Bz =  LinearNDInterpolator(points, Btor_mag.flatten(), fill_value=0.0)(RRi, ZZi)
B = np.sqrt(Bx**2 + By**2 + Bz**2)

# Determine the edge pitch angle,
pitch = np.arctan(By/Bz)[0,-1] * 180/np.pi
print(f"Edge pitch angle: {pitch:.3f} [deg]")

# Plot the COMSOL domain,
for a in [ax, ax1]:
    rect = Rectangle((Rc, Zc),w, h, color='r', alpha=0.5)
    a.add_artist(rect)
# Sanitize the magnetic field for plotting,
Bsafe = copy.deepcopy(B)
Bsafe[B==0] = np.nan

fig, axs = plt.subplots(2,3,num="Interpolated fields", figsize=(8, 5))
axs = axs.flatten()
pc = axs[0].pcolormesh(RRi, ZZi, nei)
fig.colorbar(pc, ax=axs[0], label="Density [E19 1/m^3]")
pc = axs[1].pcolormesh(RRi, ZZi, Bsafe)
fig.colorbar(pc, ax=axs[1], label="Total B [T]")
norm = Normalize(-1, 1)
pc = axs[3].pcolormesh(RRi, ZZi, Bx/Bsafe, norm=norm)
axs[4].pcolormesh(RRi, ZZi, By/Bsafe, norm=norm)
axs[5].pcolormesh(RRi, ZZi, Bz/Bsafe, norm=norm)
fig.colorbar(pc, ax=axs[5])
for a, txt in zip(axs, ["ne", "B_tot","","bx","by","bz"]):
    a.set_title(txt)
    a.set_aspect("equal")
fig.tight_layout()
axs[2].remove()

# %% Write COMSOL input file.
if write_file:

    data = np.array([RRi.flatten(),ZZi.flatten(), nei.flatten(), 
                     Bx.flatten(), By.flatten(), Bz.flatten()]).T
    print(f"INFO: writing data to {path+fname}")
    np.savetxt(path + fname, 
               data,
               header="R,Z,ne,Bx,By,Bz",delimiter=",",comments="%",
               )
