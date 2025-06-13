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

<details>
<summary><code>restrict(res): ndarray</code></summary>
<p>
Restricts residual to a coarser grid using full-weighting stencil. Reduces the resolution of the residual array <code>res</code> to a coarser grid by applying a weighted average over a 3x3 stencil. Boundaries use simple averaging if full stencil is unavailable.

<b>Parameters:</b><br>
- <code>res</code>: 2D ndarray, residual on fine grid.<br>

<b>Returns:</b><br>
- 2D ndarray of restricted residual on the coarser grid.
</p>
</details>

<details>
<summary><code>prolongate(error, target_shape): ndarray</code></summary>
<p>
Prolongates an error from coarse to fine grid by bilinear interpolation. Interpolates the coarse grid <code>error</code> to a finer grid of shape <code>target_shape</code> using bilinear interpolation with explicit loops to enable JIT compilation.

<b>Parameters:</b><br>
- <code>error</code>: 2D ndarray, error on coarse grid.<br>
- <code>target_shape</code>: tuple (int, int), desired shape of fine grid.<br>

<b>Returns:</b><br>
- 2D ndarray of interpolated error on fine grid.
</p>
</details>

<details>
<summary><code>vcycle(phi, rhs, dx, dy): ndarray</code></summary>
<p>
Performs one iterative V-cycle of the multigrid solver. Constructs a multigrid hierarchy by restriction down to coarsest grid (smallest dimension ≤ 7), performs smoothing, then prolongates corrections upward.

<b>Parameters:</b><br>
- <code>phi</code>: 2D ndarray, initial solution estimate.<br>
- <code>rhs</code>: 2D ndarray, right-hand side.<br>
- <code>dx</code>, <code>dy</code>: float, grid spacing.<br>

<b>Returns:</b><br>
- 2D ndarray, updated solution after one V-cycle.
</p>
</details>

<details>
<summary><code>multigrid_solver(phi, rhs, dx, dy, n_vcycles=5): ndarray</code></summary>
<p>
Performs multiple V-cycles to iteratively solve the Poisson problem. Repeatedly applies the <code>vcycle</code> function to progressively improve the solution estimate <code>phi</code>.

<b>Parameters:</b><br>
- <code>phi</code>: 2D ndarray, initial guess.<br>
- <code>rhs</code>: 2D ndarray, right-hand side.<br>
- <code>dx</code>, <code>dy</code>: float, grid spacing.<br>
- <code>n_vcycles</code>: int, number of V-cycles (default 5).<br>

<b>Returns:</b><br>
- 2D ndarray, refined solution after all cycles.
</p>
</details>

### Amplitude time evolution

This section includes the functions used for studying the time evolution of the amplitude of all perturbation modes present in the given data.

<details> <summary><code>get_amplitude_evolution(Data_history, nframes, iter_per_step, m_list, r_0, r_width_average, variable_name, R_max, N_r, Data_ranges): tuple</code></summary> <p> Computes the log amplitude evolution over time for selected Fourier modes.<br> <b>Data_history</b>: List of dictionaries containing FFT data.<br> <b>nframes</b>: Number of frames (time steps).<br> <b>iter_per_step</b>: Iterations per frame.<br> <b>m_list</b>: List of mode numbers to extract.<br> <b>r_0</b>: Target radius for extraction.<br> <b>r_width_average</b>: Averaging half-width in radial index.<br> <b>variable_name</b>: Name of the variable.<br> <b>R_max</b>: Maximum radius.<br> <b>N_r</b>: Number of radial points.<br> <b>Data_ranges</b>: Dictionary of variable data ranges for scaling.<br> <b>tuple</b>: (2D array of log amplitudes, array of iteration numbers). </p></details> <details> <summary><code>write_amplitude_evo(directory, variable_name, iterations, log_amplitude_list): None</code></summary> <p> Writes the amplitude evolution data to a text file.<br> <b>directory</b>: Base directory to store the file.<br> <b>variable_name</b>: Variable name (used as subdirectory).<br> <b>iterations</b>: Array of iteration numbers.<br> <b>log_amplitude_list</b>: 2D array of log amplitudes.<br> <b>None</b>: Writes file and prints confirmation. </p></details> <details> <summary><code>read_amplitude_evo(directory, variable_name): tuple</code></summary> <p> Reads the amplitude evolution data from a text file.<br> <b>directory</b>: Base directory containing the file.<br> <b>variable_name</b>: Variable name (subdirectory).<br> <b>tuple</b>: (2D array of log amplitudes, list of iterations). </p></details> <details> <summary><code>get_amplitude_evolution_plot(log_amplitude_list, iterations, modes_list, iter_range, normalization_flag, variable_name, r_0, r_width_average, R_max, N_r, dt): Figure</code></summary> <p> Generates a plot of the log amplitude evolution for selected modes.<br> <b>log_amplitude_list</b>: 2D array of log amplitudes.<br> <b>iterations</b>: Array of iteration numbers.<br> <b>modes_list</b>: List or int: modes to plot or number of top modes.<br> <b>iter_range</b>: (start, end) indices for the range to plot.<br> <b>normalization_flag</b>: Whether to normalize to the initial value in range.<br> <b>variable_name</b>: Name of the variable.<br> <b>r_0</b>: Target radius.<br> <b>r_width_average</b>: Averaging width in radial index.<br> <b>R_max</b>: Maximum radius.<br> <b>N_r</b>: Number of radial points.<br> <b>dt</b>: Time step size.<br> <b>Figure</b>: Matplotlib figure object with the plot. </p></details>

### Shear layer study

This section includes the functions developed in order to visualize the time evolution of the shear layer (arbitrary shear layer profile).

<details> <summary><code>get_shear_layer_plot(avg_Variable_straight_frame, std_Variable_straight_frame, R_max, upper_lim, bottom_lim): tuple</code></summary> <p> Plots the shear layer profile with mean and standard deviation.<br> <b>avg_Variable_straight_frame</b>: Mean velocity profile.<br> <b>std_Variable_straight_frame</b>: Standard deviation of the profile.<br> <b>R_max</b>: Maximum radius.<br> <b>upper_lim</b>: Upper y-axis limit.<br> <b>bottom_lim</b>: Lower y-axis limit.<br> <b>tuple</b>: (Figure, mean line, std fill, iteration text). </p></details> <details> <summary><code>get_shear_layer_anim(Variable_straight, R_max, nframes, iter_per_step, frame_initial): FuncAnimation</code></summary> <p> Generates an animation of the shear layer evolution over time.<br> <b>Variable_straight</b>: List of (N_r, N_theta) arrays over time.<br> <b>R_max</b>: Maximum radius.<br> <b>nframes</b>: Number of frames.<br> <b>iter_per_step</b>: Iterations per frame.<br> <b>frame_initial</b>: Initial frame index.<br> <b>FuncAnimation</b>: Matplotlib animation object. </p></details>

### Instability criteria for an arbitrary profile

This section gathers the functions used to caracterize and study the arbitrary shear layer profile obtained in the previous section. It includes a custom function to solve the previously analytical procedure for simpler approximation, generalizing thus to arbitrary shear layer profiles. It eneables the user to collect the growth rates of the different perturbation modes, store and read the processed data and visualizing using plots and animations.

<details> <summary><code>get_instability_data(Ue_profile_raw, r_disk, R_max, m_values, dy): ndarray</code></summary> <p> Computes the maximum growth rates for a given velocity profile using Rayleigh’s criterion.<br> <b>Ue_profile_raw</b>: Raw edge velocity profile.<br> <b>r_disk</b>: Reference disk radius.<br> <b>R_max</b>: Maximum radius.<br> <b>m_values</b>: Array of mode numbers to evaluate.<br> <b>dy</b>: Grid spacing in y.<br> <b>ndarray</b>: Array of maximum growth rates for each mode. </p></details> <details> <summary><code>write_instability_data(directory, variable_name, iterations, instability_data): None</code></summary> <p> Writes the computed instability data to a text file.<br> <b>directory</b>: Base directory.<br> <b>variable_name</b>: Name for subdirectory and file.<br> <b>iterations</b>: Array of iteration numbers.<br> <b>instability_data</b>: 2D array of growth rates.<br> <b>None</b>: Writes file and prints confirmation. </p></details> <details> <summary><code>read_instability_data(directory, variable_name): tuple</code></summary> <p> Reads the instability data from a text file.<br> <b>directory</b>: Base directory.<br> <b>variable_name</b>: Subdirectory and file name.<br> <b>tuple</b>: (2D array of growth rates, list of iterations). </p></details> <details> <summary><code>get_instability_plot(max_growth_rates, m_values, upper_lim, bottom_lim): tuple</code></summary> <p> Plots the maximum growth rate as a function of mode number.<br> <b>max_growth_rates</b>: Array of growth rates.<br> <b>m_values</b>: Array of mode numbers.<br> <b>upper_lim</b>: Upper y-axis limit.<br> <b>bottom_lim</b>: Lower y-axis limit.<br> <b>tuple</b>: (Figure, line plot, iteration text, max mode text). </p></details> <details> <summary><code>get_instability_anim(instability_data, m_values, nframes, iter_per_step, frame_initial): FuncAnimation</code></summary> <p> Creates an animation of the instability growth rate profile over time.<br> <b>instability_data</b>: 2D array of growth rates over time.<br> <b>m_values</b>: Array of mode numbers.<br> <b>nframes</b>: Number of frames.<br> <b>iter_per_step</b>: Iterations per frame.<br> <b>frame_initial</b>: Initial frame index.<br> <b>FuncAnimation</b>: Matplotlib animation object. </p></details>


## Example_of_use.py

This script is presented as an example of the use of most of the functions. It includes the process to get most of the results presented in the study report and poster from the raw data obtained by means of the `OOPIC_Pro_diagnostics_dump_automatization.py` script. It's divided in different sections:
 - <b>Data collection</b>: This section's main purpose is defining the parameters of the simulation and loading the raw data previously obtained. Then, it formats all the data into a single dictionary that can be stored in a Pickle file to avoid re-processing. Moreover, it includes a graphical visualization section that stores animations for all the variables studied in cartesian coordinates, polar coordinates and the fourier domain.
 - <b>B study plot</b>: This section carries out the analysis related to extracting the mode amplitude evolution from the formatted data of the previous section. This is done for all cases stated (in this study, different simulations for different values of the magnetic field magnitude).
 - <b>Get shear layer profile from drift velocity</b>: This section extracts the shear layer profile from the drift velocity magnitude. The drift velocity has to be one of the diagnostics stored in order to enable this functionality.
 - <b>Get instability crieteria for arbitrary Ue profile</b>: This section implements a way to solve Rayleigh equation for an arbitrary profile of the shear layer. The data obtained is comparable to the one of section "B study plot", being it the expected result for a pure Kelvin-Helmholtz instability.

## Disclaimers
 - Please note that some functionalities may need the user to adjust some parameters or variables to properly produce results with correct physical meaning.
 - The execution time for some functions can be considerably high due to the size of the simulation raw data, even though some optimization has been realized.
 - The size of some stored data can be large, understanding of the data studied and this scripts is highly recomended before executing them.
