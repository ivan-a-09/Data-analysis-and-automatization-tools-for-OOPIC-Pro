# Data-analysis-and-automatization-tools-for-OOPIC-Pro

This repository contains the tools developed to automatize the data collection from  OOPIC Pro v2.1, as well as the functions developed to perform data analysis on them.


## OOPIC_Pro_diagnostics_dump_automatization.py

This file uses the modules `pywinauto` and `time` to make use of the GUI of *OOPIC Pro* in order to obtain and save data from the diagnostic windows in a customizable way.
The main parameters in this code are:
* `iter4dump`: represent the number of iterations to run the simulation between saves.
* `maxiter`: represents the total number number of iterations before finishing the simulation.
* `diag_list`: contains in a list format the names of the diverse diagnostics that will be saved
  
This functionality of this code is saving every open diagnostics stated at `diag_list` and saves it in the same folder as the original input file which is being executed, while renaming it with the iteration number (the data file is stored with the form: `{diagnostics_name}_i{iteration}`). In order to avoid the undesired behaviour of the program, it's worth checking that the diagnostics windows from OOPIC Pro are identified with the exact name stated at `diag_list`.


## Data_analysis_functions.py

This file is meant to be used as a local module, containing all functions developed for the data handling and analysis. Due to the diversity of the functions included, several functions from different modules need to be imported in order to eneable the execution of all functions. The list of all modules whose functions are imported is:
`numpy`, `h5py`, `matplotlib.pyplot`, `os`, `scipy.fft`, `sys`, `scipy.sparse`, `numba`, `matplox`, `matplotlib.animation`, `scipy.ndimage`, `scipy.linalg`

The functions are divided into the different sections. A summary of the main functions can be found below:

### Data collection

This section collects the functions that eneable different ways of importing data and propperly prepare it for future analysis.

<details> <summary><code>read_h5_state(path_to_file): dict</code></summary> <p> Loads particle position and velocity data from an HDF5 file.<br> <b>path_to_file</b>: Path to the .h5 file containing simulation state data.<br> <b>dict</b>: Dictionary with keys per time step, each containing 'r' (2×N) and 'v' (3×N) NumPy arrays for positions and velocities. </p> </details>
<details> <summary><code>get_density(r_data, Nx, Ny, Lx, Ly): ndarray</code></summary> <p> Computes 2D particle density histogram.<br> <b>r_data</b>: 2×N array of positions.<br> <b>Nx, Ny</b>: Number of bins in x and y directions.<br> <b>Lx, Ly</b>: Domain lengths in x and y.<br> <b>ndarray</b>: (Nx, Ny) array of particle counts per bin. </p> </details>
<details> <summary><code>read_h5_E(path_to_file_E): (ndarray, tuple)</code></summary> <p> Loads electric field data from an HDF5 file.<br> <b>path_to_file_E</b>: Path to the .h5 file containing electric field data.<br> <b>ndarray</b>: (T, Nx, Ny) array of electric field data.<br> <b>tuple</b>: Domain info as (Nx, Ny, Lx, Ly). </p> </details>
<details> <summary><code>Get_phi_from_E(E_data, Lx, Ly): ndarray</code></summary> <p> Computes scalar potential from electric field using FFT-based method.<br> <b>E_data</b>: (2, Nx, Ny) array of E-field components.<br> <b>Lx, Ly</b>: Domain lengths.<br> <b>ndarray</b>: (Nx, Ny) array of scalar potential phi. </p> </details>
<details> <summary><code>Get_phi_from_E_FDM(E_data, Lx, Ly): ndarray</code></summary> <p> Computes scalar potential from electric field using finite difference method.<br> <b>E_data</b>: (2, Nx, Ny) array of E-field components.<br> <b>Lx, Ly</b>: Domain lengths.<br> <b>ndarray</b>: (Nx, Ny) array of scalar potential phi. </p> </details>
<details> <summary><code>get_vmod(r_data, v_data, Nx, Ny, Lx, Ly): ndarray</code></summary> <p> Computes spatial map of average velocity magnitude per grid cell.<br> <b>r_data</b>: 2×N array of positions.<br> <b>v_data</b>: 2×N array of velocity components.<br> <b>Nx, Ny</b>: Grid size.<br> <b>Lx, Ly</b>: Domain lengths.<br> <b>ndarray</b>: (Nx, Ny) array of average velocity magnitudes. </p> </details>
<details> <summary><code>get_v(r_data, v_data, Nx, Ny, Lx, Ly): (ndarray, ndarray)</code></summary> <p> Computes spatial map of average velocity vector components.<br> <b>r_data</b>: 2×N array of positions.<br> <b>v_data</b>: 2×N array of velocity components.<br> <b>Nx, Ny</b>: Grid size.<br> <b>Lx, Ly</b>: Domain lengths.<br> <b>ndarray, ndarray</b>: Two (Nx, Ny) arrays for x and y components of mean velocity. </p> </details>

### Fourier analysis

This section collects the functions required to analyze using fourier transformations circles of constant radius in order to study the perturbations developed.

<details> <summary><code>Initialise_Variable(init_option, args, Nx, Ny, Lx, Ly): ndarray</code></summary> <p> Initializes a 2D variable grid either from a parametric shape or file data.<br> <b>init_option</b>: Initialization type (0 = disk-based, 1 = file-based).<br> <b>args</b>: Arguments tuple. If 0: (eps, n_modes, Rx, Ry). If 1: (directory, filename).<br> <b>Nx, Ny</b>: Grid resolution.<br> <b>Lx, Ly</b>: Domain dimensions.<br> <b>ndarray</b>: Initialized (Nx, Ny) array. </p> </details>
<details> <summary><code>f_disk(x, y, Rx, Ry, eps, n_modes, theta, Lx, Ly): int</code></summary> <p> Defines a perturbed elliptical disk shape for grid initialization.<br> <b>x, y</b>: Coordinates.<br> <b>Rx, Ry</b>: Radii in x and y.<br> <b>eps</b>: Perturbation amplitude.<br> <b>n_modes</b>: Number of modes in perturbation.<br> <b>theta</b>: Angular coordinate.<br> <b>Lx, Ly</b>: Domain size.<br> <b>int</b>: 1 if point is inside perturbed disk, else 0. </p> </details>
<details> <summary><code>read_txt_file(data_file): tuple</code></summary> <p> Reads structured variable data from a text file.<br> <b>data_file</b>: Path to text file.<br> <b>tuple</b>: Lists of m, n indices, x, y coordinates, and variable values. </p> </details>
<details> <summary><code>Get_Variable_straight(Variable, N_r, N_theta, R_max): ndarray</code></summary> <p> Converts a 2D cartesian variable field into polar coordinate sampling.<br> <b>Variable</b>: 2D input array.<br> <b>N_r</b>: Number of radial divisions.<br> <b>N_theta</b>: Number of angular divisions.<br> <b>R_max</b>: Maximum radius of domain.<br> <b>ndarray</b>: (N_r, N_theta) array in polar form. </p> </details>
<details> <summary><code>Get_fourier_variable(Variable_straight, N_r): ndarray</code></summary> <p> Computes the FFT of the variable along each radius in polar coordinates.<br> <b>Variable_straight</b>: (N_r, N_theta) array in polar form.<br> <b>N_r</b>: Number of radial divisions.<br> <b>ndarray</b>: (N_r, N_modes) array of Fourier amplitudes. </p> </details>
<details> <summary><code>Get_plot_variable(Variable, variable_name, Lx, Ly, All_Variable_min, All_Variable_max): tuple</code></summary> <p> Generates a heatmap plot of a variable in cartesian coordinates.<br> <b>Variable</b>: 2D variable array.<br> <b>variable_name</b>: Label for the variable.<br> <b>Lx, Ly</b>: Domain dimensions.<br> <b>All_Variable_min, All_Variable_max</b>: Color scale limits.<br> <b>tuple</b>: Matplotlib figure, image, and iteration text handle. </p> </details>
<details> <summary><code>Get_plot_polar(Variable_straight, variable_name, R_max, All_Variable_straight_min, All_Variable_straight_max): tuple</code></summary> <p> Plots the variable as a function of radius and angle in polar form.<br> <b>Variable_straight</b>: (N_r, N_theta) array in polar form.<br> <b>variable_name</b>: Label for the variable.<br> <b>R_max</b>: Maximum radius.<br> <b>All_Variable_straight_min, All_Variable_straight_max</b>: Color scale limits.<br> <b>tuple</b>: Matplotlib figure, image, and iteration text handle. </p> </details>
<details> <summary><code>Get_plot_fft(fourier_variable, n_limit, variable_name, R_max): tuple</code></summary> <p> Plots the Fourier spectrum along each radius.<br> <b>fourier_variable</b>: FFT data (N_r, N_modes).<br> <b>n_limit</b>: Max number of modes to display.<br> <b>variable_name</b>: Name of the variable.<br> <b>R_max</b>: Maximum radius.<br> <b>tuple</b>: Matplotlib figure, image, and iteration text handle. </p> </details>
<details> <summary><code>deep_getsizeof(obj, seen=None): int</code></summary> <p> Recursively computes the total memory usage of a Python object including its contents.<br> <b>obj</b>: Any Python object.<br> <b>seen</b>: Set of visited object IDs to avoid circular references.<br> <b>int</b>: Total memory size in bytes. </p> </details>


### Electric potential calculation using multigrid solver

This section implements a multigrid solver to compute solutions to the Poisson equation using a non-recursive V-cycle approach. This is the same approach (with probably a suboptimal implementation) that OOPIC Pro uses to obtain the potential if required by the diagnostic selection. It includes JIT-accelerated computational kernels via numba for efficient relaxation, restriction, and prolongation operations.


<details>
<summary>
  <code>relax(phi, rhs, dx, dy, w=1.9, n_sweeps=3): None</code>
  </summary>
<p>
Performs Red-Black Gauss-Seidel relaxation with over-relaxation. Updates the solution array <code>phi</code> in-place by smoothing using multiple sweeps. It first updates the "red" points ((i+j) even), then the "black" points ((i+j) odd).

<b>Parameters:</b><br>
- <code>phi</code>: 2D ndarray, current solution estimate.<br>
- <code>rhs</code>: 2D ndarray, right-hand side of the Poisson equation.<br>
- <code>dx</code>, <code>dy</code>: float, grid spacing in x and y.<br>
- <code>w</code>: float, over-relaxation factor (default 1.9).<br>
- <code>n_sweeps</code>: int, number of relaxation sweeps (default 3).<br>

No return value, modifies <code>phi</code> in-place.
</p>
</details>

### restrict(res): ndarray
<details>
<summary>Restricts residual to a coarser grid using full-weighting stencil.</summary>
<p>
Reduces the resolution of the residual array <code>res</code> to a coarser grid by applying a weighted average over a 3x3 stencil. Boundaries use simple averaging if full stencil is unavailable.

<b>Parameters:</b><br>
- <code>res</code>: 2D ndarray, residual on fine grid.<br>

<b>Returns:</b><br>
- 2D ndarray of restricted residual on the coarser grid.
</p>
</details>

---

### prolongate(error, target_shape): ndarray
<details>
<summary>Prolongates an error from coarse to fine grid by bilinear interpolation.</summary>
<p>
Interpolates the coarse grid <code>error</code> to a finer grid of shape <code>target_shape</code> using bilinear interpolation with explicit loops to enable JIT compilation.

<b>Parameters:</b><br>
- <code>error</code>: 2D ndarray, error on coarse grid.<br>
- <code>target_shape</code>: tuple (int, int), desired shape of fine grid.<br>

<b>Returns:</b><br>
- 2D ndarray of interpolated error on fine grid.
</p>
</details>

---

### vcycle(phi, rhs, dx, dy): ndarray
<details>
<summary>Performs one iterative V-cycle of the multigrid solver.</summary>
<p>
Constructs a multigrid hierarchy by restriction down to coarsest grid (smallest dimension ≤ 7), performs smoothing, then prolongates corrections upward.

<b>Parameters:</b><br>
- <code>phi</code>: 2D ndarray, initial solution estimate.<br>
- <code>rhs</code>: 2D ndarray, right-hand side.<br>
- <code>dx</code>, <code>dy</code>: float, grid spacing.<br>

<b>Returns:</b><br>
- 2D ndarray, updated solution after one V-cycle.
</p>
</details>

---

### multigrid_solver(phi, rhs, dx, dy, n_vcycles=5): ndarray
<details>
<summary>Performs multiple V-cycles to iteratively solve the Poisson problem.</summary>
<p>
Repeatedly applies the <code>vcycle</code> function to progressively improve the solution estimate <code>phi</code>.

<b>Parameters:</b><br>
- <code>phi</code>: 2D ndarray, initial guess.<br>
- <code>rhs</code>: 2D ndarray, right-hand side.<br>
- <code>dx</code>, <code>dy</code>: float, grid spacing.<br>
- <code>n_vcycles</code>: int, number of V-cycles (default 5).<br>

<b>Returns:</b><br>
- 2D ndarray, refined solution after all cycles.
</p>
</details>

