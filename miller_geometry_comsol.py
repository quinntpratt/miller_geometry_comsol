#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:55:06 2024

@author: Quinn Pratt

Description
-----------
This program creates input files for simulations of Doppler Back-Scattering (DBS).
Input files for 2D full-wave simulations using COMSOL-RF can be created.
Input files for 3D beam-tracing using the Scotty code [1] can also be created.

The Miller Geometry [2] is used to create non-trivial model flux surfaces.

The magnetic field is specified by the toroidal field function, I = R*B_\phi and
the saftey factor, q.
 
The plasma density is specified with a radial profile, ne(r) and is mapped onto
the 2D flux surfaces.

The resulting magnetic equilibrium is NOT guarenteed to be a solution to the 
Grad-Shafronov equation.

The magnetic field components and the density are written to a (.csv) file.
The file consists of a regularly-spaced R,Z grid of the form, 
    R, Z, ne, Bx, By, Bz

Turbulent fluctuations, if included, are written to a similar .csv file, 
    R, Z, dn
    
Scotty input files can also be created.
The poloidal magnetic flux is used (by Scotty) to map the density profile onto
the R, Z magnetic equilibrium. Two files are created when write_scotty_input=True, 
    - ne.dat
    - topfile.json

References
----------
[1] Valerian H Hall-Chen et al 2022 Plasma Phys. Control. Fusion 64 095002
    https://doi.org/10.1088/1361-6587/ac57a1
[2] Physics of Plasmas 5, 973 (1998); https://doi.org/10.1063/1.872666
[3] Y. Hu, "Notes on tokamak equilibrium"
    https://youjunhu.github.io/research_notes/tokamak_equilibrium_htlatex/tokamak_equilibrium.html#x1-48001r182

All rights reserved
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import cumulative_trapezoid
import copy
import os

# Optional: change the paths and filenames of outputs,
write_comsol_input = False
comsol_files_path = os.getcwd() + "/"
comsol_input_filename = "comsol_dbs_emw_input.csv"

include_turb = True
write_comsol_turb_input = False
comsol_input_turb_filename = "comsol_dbs_emw_input_turb.csv"

write_scotty_input = False
scotty_files_path = os.getcwd() + "/"
# The scotty files are: 'topfile.json' and 'ne.dat'

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
        
    def gradient(self,analytic=True):
        """
        Compute the gradient components, dR/dr, dR/d\theta, dZ/dr, dZ/d\theta.
        All outputs are (theta, r) shape 2d np.arrays.
        
        Parameters
        ----------
        analytic : bool, optional
            Compute gradient using analytic formulas.
            If False, uses np.gradient()
                
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
        # Compute the derivative of the params,
        if analytic:
            dR0_dr = np.gradient(self.R0, self.r)
            ddelta_dr = np.gradient(self.delta, self.r)
            dkappa_dr = np.gradient(self.kappa, self.r)
            # Analytic derivative of x(r) = arcsin(delta(r))
            # can become NaN for delta(r) >= 1.
            dx_dr = ddelta_dr/np.sqrt(1 - self.delta**2)
            # Make each (1, r) for broadcasting against \theta dimension,
            dR0_dr, dx_dr, dkappa_dr = np.atleast_2d(dR0_dr, dx_dr, dkappa_dr)
            # Get other params and make (1,r) for broadcasting,
            r, R0, delta, kappa = np.atleast_2d(self.r, self.R0, self.delta, self.kappa ) # (1, nr)
            theta = np.atleast_2d(self.theta).T # (nth, 1)
            x = np.arcsin(delta)
            
            # Analytic gradients based on the Miller formulas,
            self.dRdt = r*( -np.sin(theta + x*np.sin(theta))*(1 + x*np.cos(theta)) )
            self.dRdr = dR0_dr + np.cos(theta + x*np.sin(theta)) - r*np.sin(theta + x*np.sin(theta))*dx_dr*np.sin(theta)
            self.dZdt = kappa*r*np.cos(theta)
            self.dZdr = np.sin(theta)*( kappa + r*dkappa_dr )
        else:
            # Numerical gradient based on np.gradient()            
            self.dRdt, self.dRdr = np.gradient(self.Rs, self.theta, self.r)
            self.dZdt, self.dZdr = np.gradient(self.Zs, self.theta, self.r)
        return self.dRdt, self.dRdr, self.dZdt, self.dZdr
    
    def calc_metric_terms(self, debug_plot=False):
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
        
        if debug_plot:
            plt.figure("MillerGeom.calc_metric_terms")
            plt.plot(self.r,self.J[0,:], label="Jacobian")
            plt.plot(self.r,self.gr_gt[0,:], label=r"$\nabla r \cdot \nabla \theta$")
            plt.plot(self.r,self.gr2[0,:], label=r"$|\nabla r|^2$")
            plt.xlabel("r_min")
            plt.title(r"Metric Terms at $\theta = 0$")
            plt.legend()
        
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
# Definition of the full \theta \in [0, 2pi) grid,
Nth, Nr = 512, 64
r = np.linspace(0.4, 0.7, Nr) # midplane half-diameter coordinate.
theta = np.linspace(-np.pi, np.pi, Nth) # poloidal angle coordinate.

# Flux-surface kwargs can be scalars or functions of 'r',
fs_kw = dict(R0=1.7, delta=0.16, kappa=1.35)
# Linear functions of 'r', 
fs_kw = dict(R0=-0.14*r + 1.78,
             delta=1.26*r - 0.41,
             kappa=1.12*r + 0.84,)

# Create the MillerGeom object,
MG = MillerGeom(r, theta, **fs_kw)
# Extract the flux surfaces,
Rs, Zs = MG.Rs, MG.Zs 
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
# The toroidal field function -- must be len(r)
I = -3.08*np.ones(len(r)) # I(psi) = R*B_tor [m*T]
Btor_mag = I/Rs
# Saftey factor,
# q = 2.8*np.ones(len(r)) # constant
# q = -17.2*r + 5.1 # linear
q = -65.5*r**2 + 48.5*r - 11.16 # quadratic

# Calculate the poloidal magnetic field,
Bp = MG.calc_Bp(I, q)
Bp_R = Bp[0] # [T]
Bp_Z = Bp[1] # [T] 
Bp_mag = np.sqrt(Bp_R**2 + Bp_Z**2) # [T]

# Plot the poloidal magnetic field,
c = ax.contourf(Rs, Zs, Bp_mag)
fig.colorbar(c, ax=ax,label=r"$B_p$ [T]")
ax.quiver(Rs[::skip,::skip], Zs[::skip,::skip],
          Bp_R[::skip,::skip], Bp_Z[::skip,::skip])
# Also show the I(psi) and q(psi)
axs[3].plot(r, I)
axs[3].set_ylabel(r"$I = RB_\phi$")
axs[4].plot(r, q)
axs[4].set_ylabel(r"q")
fig.tight_layout()
# Additional plot of the toroidal and poloidal field at the midplane
# for comparison with GEQDSK files.
fig, axs = plt.subplots(2,1,sharex=True)
axs[0].plot(r, Btor_mag[0,:])
axs[1].plot(r, Bp_mag[0,:])
axs[0].set_ylabel(r"$B_\phi$ [T] (midplane)")
axs[1].set_ylabel(r"$B_p$ [T] (midplane)")
axs[1].set_xlabel("r (rmin)")

# %% Density profile,
# Define a density profile vs. rmin, then map it onto the flux surfaces.
# Although technically the LCFS should be the last point in our 'r' grid (above)
# it is sometimes useful to have close field-lines extend slightly outward.
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
B_mag = np.sqrt(Bp_mag**2 + Btor_mag**2)[Nth//2,:] # midplane [T]
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
fig.tight_layout()

# %% Poloidal magnetic flux,
# The Scotty code uses the pol. magnetic flux to map the density profile onto the 
# magnetic equilibrium. It's okay if the poloidal magnetic flux is not totally
# accurate. It only needs to create a 1:1 map onto the density profile.

# To obtain the poloidal magnetic flux we will integrate its (radial) derivative
# which is calculated internally by the MillerGeometry class.
dpsi_dr = MG.psiprime # shape is (r)
# Formally, we would need to integrate from the magnetic axis.
# for now we will just set an arbitrary offset value,
offset=0.1
psi = cumulative_trapezoid(dpsi_dr, r, initial=0) + offset

# The normalized poloidal flux (psi_norm) is assumed to be the square root
# of the poloidal flux normalized to the separatrix value,
psi_norm = np.sqrt(psi/psi[np.argmin(abs(r - r_sep))])
# The Scotty topfile.json expects the poloidal flux to be the square of the 
# abscissa accompanying the ne profile.
psi_RZ = psi_norm**2 + 0*Rs # map onto Z, R

# %% Turbulence,
# NOTE: for high k_theta, high poloidal resolution is needed - consider increasing
# Nth above.

# Additional toggles,
use_local_values = True
turb_plot_debug = True

if include_turb:
    # Poloidal wavenumber,
    k_theta = 0.3 # [rad/cm]
    # Differential poloidal arc-length, d\ell/d\theta converts k_theta in
    # [rad/cm] into a poloidal mode number 'm',
    dl_dtheta = np.sqrt( MG.dRdt**2 + MG.dZdt**2 ) # [Nt, Nr] array    
    m = k_theta*100 * dl_dtheta # multiply by 100 to convert to [rad/m]
    
    # Turbulence model,
    dn_n = 0.005 # 0.5% over background.
    r_turb = 0.5 # [m]
    dr_turb = 0.01 # [m]
    radial_func = np.exp(-0.5*(np.atleast_2d(r) - r_turb)**2/dr_turb**2) # [1, Nr]
    
    turb_ind = np.argmin(abs(r - r_turb))
    if use_local_values:
        # uses the dl/d\theta at r = r_turb.
        # dl/d\theta can shear significantly on the HFS, leading to weird artifacts.
        m = np.atleast_2d(m[:,turb_ind]).T # [Nth, 1]
        amplitude = dn_n*ne[turb_ind]
    else:
        amplitude = dn_n*ne_fs
        
    turb = amplitude * np.cos(m*np.atleast_2d(theta).T) * radial_func # [Nt, Nr]

    # Overplot as contours on the flux surface,
    v = dn_n*ne[turb_ind]
    levels = [-v, -0.5*v, -0.25*v, 0.25*v, 0.5*v, v]
    ax1.contour(Rs, Zs, turb, levels=levels, cmap="bwr", norm=Normalize(-v, v))
    
    if turb_plot_debug:
        figturb, (axt1, axt2) = plt.subplots(2,1,sharex=True,num="turb_plot_debug")
        axt1.plot(theta/np.pi, dl_dtheta[:,turb_ind])
        axt1.set_ylabel(r"$dl/d\theta$")
        axt2.plot(theta/np.pi, turb[:,turb_ind])
        axt2.set_ylabel(r"$\delta n$ [E19 1/$m^3$]")
        axt2.set_xlabel(r"$\theta/\pi$")


# %% COMSOL Domain setup.
# Define a box in the poloidal plane for COMSOL simulations.

# Lower-left corner,
Rc, Zc = 2.14, -0.11 # [m]
# Width and height of domain,
w, h = 0.195, 0.3075 # [m]

# Create a higher-resolution grid,
NRi, NZi = 256, 256
Ri = np.linspace(Rc, Rc + w, NRi)
Zi = np.linspace(Zc, Zc + h, NZi)
RRi, ZZi = np.meshgrid(Ri, Zi)
# Ravel into points for 2D interpolation,
points = np.array([Rs.flatten(),Zs.flatten()]).T # must be (NRi*NZi, 2)
# Linearly interpolate quantities onto the higher-res. grid.
# NOTE: regions out of the flux surfaces are filled with zeros/ne_sep.
nei = LinearNDInterpolator(points, ne_fs.flatten(), fill_value=ne_sep)(RRi, ZZi)

psi_RZi = LinearNDInterpolator(points, psi_RZ.flatten(), fill_value=0.0)(RRi, ZZi)
Bx =  LinearNDInterpolator(points, Bp_R.flatten(), fill_value=0.0)(RRi, ZZi)
By =  LinearNDInterpolator(points, Bp_Z.flatten(), fill_value=0.0)(RRi, ZZi)
Bz =  LinearNDInterpolator(points, Btor_mag.flatten(), fill_value=0.0)(RRi, ZZi)
B = np.sqrt(Bx**2 + By**2 + Bz**2)

if include_turb:
    dn_RZi = LinearNDInterpolator(points, turb.flatten(), fill_value=0.0)(RRi, ZZi)


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
if include_turb:
    pc = axs[2].pcolormesh(RRi, ZZi, dn_RZi, cmap="bwr")
    fig.colorbar(pc, ax=axs[2], label=r"$\delta$n [E19 1/m^3]")
else:
    axs[2].remove()

norm = Normalize(-1, 1)
pc = axs[3].pcolormesh(RRi, ZZi, Bx/Bsafe, norm=norm)
axs[4].pcolormesh(RRi, ZZi, By/Bsafe, norm=norm)
axs[5].pcolormesh(RRi, ZZi, Bz/Bsafe, norm=norm)
fig.colorbar(pc, ax=axs[5])
for a, txt in zip(axs, ["ne", "B_tot","","bx","by","bz"]):
    a.set_title(txt)
    a.set_aspect("equal")
fig.tight_layout()

# %% Write input files for other codes.
if write_comsol_input:
    data = np.array([RRi.flatten(),ZZi.flatten(), nei.flatten(), 
                     Bx.flatten(), By.flatten(), Bz.flatten()]).T
    print(f"INFO: writing data to {comsol_files_path+comsol_input_filename}")
    np.savetxt(comsol_files_path + comsol_input_filename, 
               data,
               header="R,Z,ne,Bx,By,Bz",delimiter=",",comments="%",
               )

if write_comsol_turb_input and include_turb:
    data = np.array([RRi.flatten(),ZZi.flatten(), dn_RZi.flatten()]).T
    print(f"INFO: writing turb data to {comsol_files_path+comsol_input_turb_filename}")
    np.savetxt(comsol_files_path + comsol_input_turb_filename, 
               data,
               header="R,Z,dn",delimiter=",",comments="%",
               )

if write_scotty_input:
    print(f"INFO: writing Scotty input files to {scotty_files_path}")
    # The Scotty beam tracing code requires an ne.dat file and a topfile.
    # 1. The ne.dat file contains the density profile information,
    # Write the file,
    fmt = '%.8e'
    with open(scotty_files_path+"ne.dat", "w+") as f:
        f.write(str(len(psi_norm)) + '\n')  # first line is the number of points
        _tmp = np.vstack([psi_norm, ne]).T
        np.savetxt(f, _tmp, fmt=fmt)
    # 2. The topfile.json contains the magnetic field information,
    import json
    topfile_file = open(scotty_files_path+'topfile.json', 'w')
    topfile_dict = {'R': Ri.tolist(), 
                    'Z': Zi.tolist(),
                    'Br': Bx.flatten().tolist(), 
                    'Bz': By.flatten().tolist(), 
                    'Bt': Bz.flatten().tolist(), 
                    'pol_flux': psi_RZi.flatten().tolist()}
    json = json.dumps(topfile_dict, indent=6)
    topfile_file.write(json)
    topfile_file.close()
print("* done")