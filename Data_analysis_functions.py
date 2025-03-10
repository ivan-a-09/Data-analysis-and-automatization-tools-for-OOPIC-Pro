import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.fft import fft2, ifft2, fftfreq
import sys
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit
import matplotx
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from scipy.linalg import eigvals





# =============================================================================
# DATA COLLECTION
# =============================================================================


def read_h5_state(path_to_file):
    with h5py.File(path_to_file, 'r') as h5file:
        key_list = list(h5file.keys())[3:]

        data_info = {}
        for key in key_list:
            group = h5file[key]
            pGrp_key_list = list(group.keys())

            pGrp_len_list = [len(group[pGrp_key]['ptcls'][:]) for pGrp_key in pGrp_key_list]
            n_ptcl = sum(pGrp_len_list)

            data_info[key] = {'n_ptcl': n_ptcl, 'pGrp_len_list': np.array(pGrp_len_list)}

        data = {}
        for key in key_list:
            n_ptcl = data_info[key]['n_ptcl']
            pGrp_key_list = list(h5file[key])

            r = np.zeros((2, n_ptcl))
            v = np.zeros((3, n_ptcl))

            pGrp_len_list = data_info[key]['pGrp_len_list']
            index_ranges = np.cumsum(pGrp_len_list)

            for i, pGrp_key in enumerate(pGrp_key_list):
                start_idx = 0 if i == 0 else index_ranges[i - 1]
                end_idx = index_ranges[i]

                particles = h5file[key][pGrp_key]['ptcls'][:].T  # Transpose once
                r[:, start_idx:end_idx] = particles[:2, :]
                v[:, start_idx:end_idx] = particles[2:5, :]

            data[key] = {'r': r, 'v': v}

    return data


def get_density(r_data, Nx, Ny, Lx, Ly):
    x_pos = r_data[0, :]
    y_pos = r_data[1, :]
    n = np.histogram2d(x_pos, y_pos, bins=[Nx, Ny], range=[[0, Lx], [0, Ly]])[0]
    return n




def read_h5_E(path_to_file_E):
    with h5py.File(path_to_file_E, 'r') as h5file:
        key_list = list(h5file.keys())[:]
        try:
            key_list.remove('x1array')
            key_list.remove('x2array')
        except:
            print('####################################')
            print('WARNING: data structure not expected')
            print('####################################')
            
        x1array = h5file['x1array'][:]
        x2array = h5file['x2array'][:]
        Nx = len(x1array)
        Lx = x1array[-1]
        Ny = len(x2array)
        Ly = x2array[-1]
        domain = Nx, Ny, Lx, Ly
            
        data_E = np.zeros((len(key_list),Nx,Ny))
        
        for i in range(len(key_list)):
            data_E[i,:,:] = h5file[key_list[i]][:][:,:,0]
            
        
    return data_E, domain




def Get_phi_from_E(E_data,Lx,Ly):
    E_x = E_data[0][:,:]
    E_y = E_data[1][:,:]
    Nx, Ny = E_x.shape
    delta_x = Lx/(Nx-1)
    delta_y = Ly/(Ny-1)
    div_E = (np.gradient(E_x, axis=1) / delta_x + np.gradient(E_y, axis=0) / delta_y)
    
    kx = 2 * np.pi * fftfreq(Nx, delta_x)
    ky = 2 * np.pi * fftfreq(Ny, delta_y)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1
    
    phi = (ifft2(fft2(-div_E) / K2)).real
    phi -= phi[0, 0]

    return phi



def Get_phi_from_E_FDM(E_data, Lx, Ly):
    E_x = E_data[0][:,:]
    E_y = E_data[1][:,:]
    Nx, Ny = E_x.shape
    delta_x = Lx / (Nx - 1)
    delta_y = Ly / (Ny - 1)

    # Compute divergence of E using central differences
    div_E = np.zeros((Nx, Ny))
    div_E[1:-1, :] += (E_x[1:-1, :] - E_x[:-2, :]) / delta_x
    div_E[:, 1:-1] += (E_y[:, 1:-1] - E_y[:, :-2]) / delta_y

    # Reshape div_E into a 1D vector (flattened column-major order)
    b = -div_E.flatten()

    # Construct the finite difference Laplacian matrix
    N = Nx * Ny
    diag = np.ones(N) * (-2 / delta_x**2 - 2 / delta_y**2)
    off_x = np.ones(N - 1) / delta_x**2
    off_y = np.ones(N - Nx) / delta_y**2

    # Remove coupling at boundaries
    for i in range(1, Nx):
        off_x[i * Ny - 1] = 0  # Remove horizontal connections at edges

    # Construct sparse matrix
    A = sp.diags(
        [off_y, off_x, diag, off_x, off_y],
        [-Nx, -1, 0, 1, Nx],
        shape=(N, N),
        format="csr"
    )

    # Apply Dirichlet BCs (zero potential at edges)
    for i in range(Nx):
        for j in range(Ny):
            if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                index = i * Ny + j
                A[index, :] = 0
                A[index, index] = 1  # Set diagonal to 1
                b[index] = 0  # Force phi to be zero

    # Solve linear system A * phi = b
    phi = spla.spsolve(A, b)

    # Reshape solution back to 2D grid
    phi = phi.reshape((Nx, Ny))

    return phi



def get_vmod(r_data, v_data, Nx, Ny, Lx, Ly):
    v_x = v_data[0,:]
    v_y = v_data[1,:]
    x_bins = np.linspace(0, Lx, Nx+1)
    y_bins = np.linspace(0, Ly, Ny+1)
    
    # 1. For each data point, find the corresponding bin index along x and y.
    #    np.digitize returns indices in 1...len(bins); subtract 1 to get 0-indexed bins.
    bin_x = np.digitize(r_data[0, :], x_bins) - 1
    bin_y = np.digitize(r_data[1, :], y_bins) - 1
    
    # 2. Combine the bin indices into a single index for each grid cell.
    #    For each point, its unique bin index is: index = bin_x * Ny + bin_y.
    bin_indices = bin_x * Ny + bin_y
    
    # 3. Use np.bincount to compute the count and sums per bin.
    #    Make sure the length is Nx*Ny so that each bin has an entry.
    counts = np.bincount(bin_indices, minlength=Nx*Ny)
    sum_vx = np.bincount(bin_indices, weights=v_x, minlength=Nx*Ny)
    sum_vy = np.bincount(bin_indices, weights=v_y, minlength=Nx*Ny)
    
    # 4. Compute the means safely. Avoid division by zero.
    mean_vx = np.zeros(Nx*Ny)
    mean_vy = np.zeros(Nx*Ny)
    nonzero = counts > 0
    mean_vx[nonzero] = sum_vx[nonzero] / counts[nonzero]
    mean_vy[nonzero] = sum_vy[nonzero] / counts[nonzero]
    
    # 5. Compute the magnitude per bin.
    vmod_flat = np.sqrt(mean_vx**2 + mean_vy**2)
    
    # 6. Reshape the flat result back into an (Nx, Ny) array.
    vmod = vmod_flat.reshape(Nx, Ny)
    return vmod


def get_v(r_data, v_data, Nx, Ny, Lx, Ly):
    v_x = v_data[0,:]
    v_y = v_data[1,:]
    x_bins = np.linspace(0, Lx, Nx+1)
    y_bins = np.linspace(0, Ly, Ny+1)
    
    # 1. For each data point, find the corresponding bin index along x and y.
    #    np.digitize returns indices in 1...len(bins); subtract 1 to get 0-indexed bins.
    bin_x = np.digitize(r_data[0, :], x_bins) - 1
    bin_y = np.digitize(r_data[1, :], y_bins) - 1
    bin_x = np.clip(bin_x, 0, Nx - 1)
    bin_y = np.clip(bin_y, 0, Ny - 1)
    
    # 2. Combine the bin indices into a single index for each grid cell.
    #    For each point, its unique bin index is: index = bin_x * Ny + bin_y.
    bin_indices = bin_y * Ny + bin_x
    
    # 3. Use np.bincount to compute the count and sums per bin.
    #    Make sure the length is Nx*Ny so that each bin has an entry.
    counts = np.bincount(bin_indices, minlength=Nx*Ny)
    sum_vx = np.bincount(bin_indices, weights=v_x, minlength=Nx*Ny)
    sum_vy = np.bincount(bin_indices, weights=v_y, minlength=Nx*Ny)
    
    # 4. Compute the means safely. Avoid division by zero.
    mean_vx = np.zeros(Nx*Ny)
    mean_vy = np.zeros(Nx*Ny)
    nonzero = counts > 0
    mean_vx[nonzero] = sum_vx[nonzero] / counts[nonzero]
    mean_vy[nonzero] = sum_vy[nonzero] / counts[nonzero]
    
    # 5. Reshape the flat result back into an (Nx, Ny) array.
    mean_vx_reshaped = mean_vx.reshape(Nx, Ny)
    mean_vy_reshaped = mean_vy.reshape(Nx, Ny)
    return mean_vx_reshaped , mean_vy_reshaped


# =============================================================================
# =============================================================================


# =============================================================================
# FOURIER ANALYSIS
# =============================================================================


def Initialise_Variable(init_option, args, Nx,Ny,Lx,Ly):
    Variable = np.zeros((Nx,Ny))
    
    if init_option == 0:
        eps, n_modes, Rx, Ry = args
        for i in range(Nx):
            for j in range(Ny):
                x = i*Lx/Nx
                y = j*Ly/Ny
                theta = np.arctan2((y-Ly/2), (x-Lx/2 + 1e-15))
                Variable[i,j] = f_disk(x, y, Rx, Ry, eps, n_modes, theta)
                
    elif init_option == 1:
        directory, filename = args
        data_file = os.path.join(directory,filename)
        m_data, n_data, x_data, y_data, Variable_data = read_txt_file(data_file)
        Variable = np.zeros((int(np.sqrt(len(Variable_data))),int(np.sqrt(len(Variable_data)))))
        #Coordinates = np.zeros((int(np.sqrt(len(Variable_data))),int(np.sqrt(len(Variable_data))),2))
        for i in range(len(m_data)):
            Variable[m_data[i],n_data[i]] = Variable_data[i]
            #Coordinates[m_data[i],n_data[i],:] = x_data[i], y_data[i]
        
    return Variable


def f_disk(x,y,Rx,Ry,eps,n_modes,theta,Lx,Ly):
    perturbation = np.sum(eps*np.sin(theta*n_modes))
    if ((x-Lx/2)/Rx)**2 + ((y-Ly/2)/Ry)**2 <= (1+perturbation)**2:
        return (1)
    else:
        return 0
    

def read_txt_file(data_file):
    with open(data_file, 'r',encoding='cp1252') as file:
        header_skipped = False
        m_data = []
        n_data = []
        x_data = []
        y_data = []
        Variable_data = []
        
        for line in file:
            line = line.strip()
            
            # Skip the header and metadata section
            if not header_skipped:
                if line.startswith("* m, n, x, y,"):
                    header_skipped = True  # Start reading data after this line
                continue
            
            # Stop processing if the footer is reached
            if line.startswith("*"):
                break
            
            # Process the data lines
            if line:
                # Split the line into columns and convert to appropriate data types
                row = line.split()
                m, n = int(row[0]), int(row[1])
                x, y, Variable = float(row[2]), float(row[3]), float(row[4])
                m_data.append(m)
                n_data.append(n)
                x_data.append(x)
                y_data.append(y)
                Variable_data.append(Variable)
            
    return m_data, n_data, x_data, y_data, Variable_data




def Get_Variable_straight(Variable,N_r,N_theta,R_max):
    Nx, Ny = Variable.shape
    R_list = np.linspace(R_max * 0.001, R_max, N_r)
    Theta_list = np.linspace(0, 2 * np.pi, N_theta)
    R_grid, Theta_grid = np.meshgrid(R_list, Theta_list, indexing='ij')
    x_coords = np.round((R_grid * np.cos(Theta_grid) / R_max + 1)*(Nx-1) / 2).astype(int)
    y_coords = np.round((R_grid * np.sin(Theta_grid) / R_max + 1)*(Ny-1) / 2).astype(int)
    Variable_straight = Variable[x_coords, y_coords]
    return Variable_straight



def Get_fourier_variable(Variable_straight,N_r):
    fourier_variable = np.zeros((N_r, len(Variable_straight)//2+1))
    for i in range(N_r):
        fourier_variable[i,:] = np.abs(np.fft.rfft(Variable_straight[i,:] - np.mean(Variable_straight[i,:] ), norm="ortho"))
    return fourier_variable

def Get_plot_variable(Variable,variable_name,Lx,Ly,All_Variable_min,All_Variable_max):
    ### PLOT OF THE STUDIED VARIABLE
    fig_variable, ax = plt.subplots(figsize=(10,8))
    plt.title(f'Variable studied: {variable_name}')
    plot_variable = ax.imshow(Variable, origin="lower",extent=[0, Lx, 0, Ly], vmin=All_Variable_min, vmax=All_Variable_max)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.colorbar(plot_variable)
    iter_text = ax.text(0.05, 0.95, "iter = ", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes , ha="left")
    return fig_variable, plot_variable, iter_text


def Get_plot_polar(Variable_straight,variable_name,R_max,All_Variable_straight_min,All_Variable_straight_max):
    ### PLOT OF THE VARIABLE IN POLAR COORDINATES
    fig_polar, ax = plt.subplots(figsize=(10,8))
    plt.title(f'Transformation to polar coordinates of {variable_name}')
    plot_polar = ax.imshow(Variable_straight, origin="lower", extent=[0.001*R_max,R_max, 0,2*np.pi], aspect='auto', vmin=All_Variable_straight_min, vmax=All_Variable_straight_max)
    ax.set_yticks([0,np.pi/2,np.pi, 3*np.pi/2, 2*np.pi],[r'$0$', r'$\dfrac{\pi}{2}$', r'$\pi$', r'$\dfrac{3 \pi}{2}$', r'$2 \pi$'])
    ax.set_xlabel('r (cm)')
    ax.set_ylabel(r'$\theta$ (rad)')
    plt.colorbar(plot_polar)
    iter_text = ax.text(0.05, 0.95, "iter = ", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes , ha="left")
    return fig_polar, plot_polar, iter_text


def Get_plot_fft(fourier_variable,n_limit,variable_name,R_max):
    ### PLOT OF THE FOURIER TRANSFORM FOR EACH RADIUS
    fig_fft, ax = plt.subplots(figsize=(10,8))
    plt.title('Fourier transform of every circumference (r=constant)')
    plot_fft = ax.imshow(fourier_variable[:,:n_limit+1], aspect='auto', interpolation='none', origin="lower", extent=[0.001*R_max,R_max, 0, n_limit], vmin=0,vmax=1)
    ax.set_xlabel('r (cm)')
    ax.set_ylabel(fr'$\mathcal{{F}} \  \left( {variable_name}|_{{r = constant}} \right)$')
    ax.set_yticks(np.array(range(0,n_limit))+0.5, [str(n) for n in range(0,n_limit)])
    plt.colorbar(plot_fft)
    iter_text = ax.text(0.05, 0.95, "iter = ", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes , ha="left")
    return fig_fft, plot_fft, iter_text


def deep_getsizeof(obj, seen=None):
    """Recursively find the total memory footprint of an object and its contents."""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Prevent infinite recursion for circular references
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(deep_getsizeof(i, seen) for i in obj)

    return size


# =============================================================================
# =============================================================================




# =============================================================================
# PHI CALCULATION USING MULTIGRID SOLVER (Modified)
# =============================================================================

# ---------------------------------------------------------
# JIT-compiled numerical kernels
# ---------------------------------------------------------

@njit
def relax(phi, rhs, dx, dy, w=1.9, n_sweeps=3):
    """
    Performs Red-Black Gauss-Seidel relaxation with over-relaxation.
    This version updates all interior points by first updating the "red" points
    (where (i+j) is even) then the "black" points (where (i+j) is odd).
    Updates are done in-place.
    """
    dx2 = dx**2
    dy2 = dy**2
    factor = 1.0 / (2.0/dx2 + 2.0/dy2)
    ny, nx = phi.shape
    for sweep in range(n_sweeps):
        # Red update: (i+j)%2 == 0
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if (i + j) % 2 == 0:
                    phi[i, j] = (1 - w)*phi[i, j] + w * factor * (
                        (phi[i+1, j] + phi[i-1, j]) / dx2 +
                        (phi[i, j+1] + phi[i, j-1]) / dy2 - rhs[i, j])
        # Black update: (i+j)%2 == 1
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if (i + j) % 2 == 1:
                    phi[i, j] = (1 - w)*phi[i, j] + w * factor * (
                        (phi[i+1, j] + phi[i-1, j]) / dx2 +
                        (phi[i, j+1] + phi[i, j-1]) / dy2 - rhs[i, j])

@njit
def restrict(res):
    """
    Restricts a residual to a coarser grid using full-weighting.
    For interior coarse points the stencil is:
    
      coarse[i,j] = 1/16 * [4*res(2i,2j) +
                   2*(res(2i-1,2j) + res(2i+1,2j) + res(2i,2j-1) + res(2i,2j+1)) +
                   (res(2i-1,2j-1) + res(2i-1,2j+1) + res(2i+1,2j-1) + res(2i+1,2j+1))]
    
    At boundaries where the full stencil isn’t available, we fall back on a simple average.
    """
    ny, nx = res.shape
    # Compute coarse dimensions (assumes even/odd structure; adjust if needed)
    ny_coarse = (ny + 1) // 2
    nx_coarse = (nx + 1) // 2
    coarse = np.empty((ny_coarse, nx_coarse), dtype=res.dtype)
    for i in range(ny_coarse):
        for j in range(nx_coarse):
            I = 2 * i
            J = 2 * j
            # Check if we can apply full 3x3 stencil
            if I > 0 and I < ny-1 and J > 0 and J < nx-1:
                coarse[i, j] = (1/16.0)*(
                    4 * res[I, J] +
                    2 * (res[I-1, J] + res[I+1, J] + res[I, J-1] + res[I, J+1]) +
                    res[I-1, J-1] + res[I-1, J+1] + res[I+1, J-1] + res[I+1, J+1]
                )
            else:
                # For boundaries, average the available neighbors
                sum_val = 0.0
                count = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ii = I + di
                        jj = J + dj
                        if ii >= 0 and ii < ny and jj >= 0 and jj < nx:
                            sum_val += res[ii, jj]
                            count += 1
                coarse[i, j] = sum_val / count
    return coarse

@njit
def prolongate(error, target_shape):
    """
    Prolongates an error to a finer grid using bilinear interpolation.
    This version is written with explicit loops so that it can be JIT compiled.
    """
    m, n = error.shape
    M, N = target_shape
    fine_error = np.empty((M, N), dtype=error.dtype)
    # Scaling factors to map fine grid indices to coarse grid indices
    scale_i = (m - 1) / (M - 1) if M > 1 else 0.0
    scale_j = (n - 1) / (N - 1) if N > 1 else 0.0
    for i in range(M):
        for j in range(N):
            x = i * scale_i
            y = j * scale_j
            i0 = int(np.floor(x))
            j0 = int(np.floor(y))
            i1 = i0 + 1 if i0 + 1 < m else i0
            j1 = j0 + 1 if j0 + 1 < n else j0
            di = x - i0
            dj = y - j0
            fine_error[i, j] = ((1 - di) * (1 - dj) * error[i0, j0] +
                                (1 - di) * dj * error[i0, j1] +
                                di * (1 - dj) * error[i1, j0] +
                                di * dj * error[i1, j1])
    return fine_error

# ---------------------------------------------------------
# Iterative V-cycle multigrid solver (non-recursive)
# ---------------------------------------------------------

def vcycle(phi, rhs, dx, dy):
    """
    Performs one V-cycle of the multigrid method.
    
    The V-cycle is implemented iteratively:
      - In the downward phase, we restrict the residual to build coarser levels.
      - In the upward phase, we prolongate the correction back to the fine grid.
      
    Changes:
      - We stop coarsening when the grid’s smallest dimension is <= 7.
    """
    # Lists to store grid levels and associated parameters
    levels_phi = []
    levels_rhs = []
    levels_dx = []
    levels_dy = []
    
    # Downward phase: create the multigrid hierarchy
    current_phi = phi.copy()
    current_rhs = rhs.copy()
    current_dx = dx
    current_dy = dy
    # Stop coarsening when the smallest dimension is small enough
    while current_phi.shape[0] > 7 and current_phi.shape[1] > 7:
        levels_phi.append(current_phi.copy())
        levels_rhs.append(current_rhs.copy())
        levels_dx.append(current_dx)
        levels_dy.append(current_dy)
        relax(current_phi, current_rhs, current_dx, current_dy, n_sweeps=3)
        ny, nx = current_phi.shape
        # Compute the residual on the interior
        residual = np.zeros_like(current_rhs)
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                lap = ((current_phi[i-1, j] - 2*current_phi[i, j] + current_phi[i+1, j]) / current_dx**2 +
                       (current_phi[i, j-1] - 2*current_phi[i, j] + current_phi[i, j+1]) / current_dy**2)
                residual[i, j] = current_rhs[i, j] - lap
        # Restrict the residual to obtain the coarse grid RHS using full weighting
        coarse_rhs = restrict(residual)
        coarse_shape = coarse_rhs.shape
        current_phi = np.zeros(coarse_shape, dtype=phi.dtype)
        current_rhs = coarse_rhs
        current_dx *= 2
        current_dy *= 2
        
    # At the coarsest level, apply heavy smoothing
    relax(current_phi, current_rhs, current_dx, current_dy, n_sweeps=10)
    
    # Upward phase: prolongate corrections and update the solution
    for k in range(len(levels_phi)-1, -1, -1):
        fine_phi = levels_phi[k].copy()
        fine_rhs = levels_rhs[k]
        fine_dx = levels_dx[k]
        fine_dy = levels_dy[k]
        # Determine the size of the interior of the fine grid
        M = fine_phi.shape[0] - 2  # interior size in i-direction
        N = fine_phi.shape[1] - 2  # interior size in j-direction
        # Prolongate the correction to the interior
        correction = prolongate(current_phi, (M, N))
        ny, nx = fine_phi.shape
        # Update the interior of the fine grid (assuming the correction aligns with interior points)
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                fine_phi[i, j] += correction[i-1, j-1]
        relax(fine_phi, fine_rhs, fine_dx, fine_dy, n_sweeps=3)
        current_phi = fine_phi.copy()
        
    return current_phi

def multigrid_solver(phi, rhs, dx, dy, n_vcycles=5):
    """
    Performs multiple V-cycles to solve the problem.
    """
    for _ in range(n_vcycles):
        phi = vcycle(phi, rhs, dx, dy)
    return phi







# =============================================================================
# AMPLITUDE TIME EVOLUTION
# =============================================================================



def get_amplitude_evolution(Data_history, nframes, iter_per_step, m_list, r_0, r_width_average, variable_name, R_max, N_r, Data_ranges):
    Max_fft = Data_ranges[variable_name]["fft"][0]
    log_amplitude_list = np.zeros((len(m_list),nframes))
    iterations = np.arange(1, nframes+1) * iter_per_step
    for i in range(len(m_list)):
        m = m_list[i]
        radius_array  = np.linspace(R_max * 0.001, R_max, N_r)
        r_index = np.argmin(np.abs(radius_array - r_0))
        start_index = max(0, r_index - r_width_average)
        end_index = min(N_r, r_index + r_width_average+1)
        amplitude_values = [
            np.max(
            d[variable_name]["fft"][start_index:end_index,m]
            )*Max_fft for d in Data_history]
        log_amplitude_values = np.log(np.abs(amplitude_values) + 1e-10)
        log_amplitude_list[i,:] = log_amplitude_values
    return log_amplitude_list, iterations


def write_amplitude_evo(directory, variable_name, iterations, log_amplitude_list):
    with open(os.path.join(directory,variable_name,"amplitude_evo.txt"), "w") as file:    
        for i in range(len(iterations)):
            values_str = "\t".join(map(str, log_amplitude_list[:,i]))  # Convert values list to a tab-separated string
            file.write(f"{iterations[i]}\t{values_str}\n")
    print('Data stored at txt file')


def read_amplitude_evo(directory, variable_name):
    with open(os.path.join(directory,variable_name,"amplitude_evo.txt"), "r") as file:
        lines = file.readlines()[:]
    
    iterations, values = [], []
    for line in lines:
        parts = line.strip().split("\t")  # Split by tab
        iterations.append(int(parts[0]))  # First part is the iteration
        values.append([float(x) for x in parts[1:]])  # Remaining parts are values
    values = np.array(values).T
    return values, iterations



def get_amplitude_evolution_plot(log_amplitude_list, iterations, modes_list, iter_range, normalization_flag, variable_name, r_0, r_width_average, R_max, N_r, dt):
    iter_range_left, iter_range_right = iter_range
    fig_amplitude = plt.figure(figsize=(8, 5))
    if normalization_flag:
        log_amplitude_corrected_list = log_amplitude_list[:,iter_range_left:iter_range_right]- (np.meshgrid(log_amplitude_list[:,iter_range_left],np.ones(len(log_amplitude_list[0,iter_range_left:iter_range_right])))[0]).T
    elif not normalization_flag:
        log_amplitude_corrected_list = log_amplitude_list[:,iter_range_left:iter_range_right]

    if type(modes_list) == int:
        # IF MODES_LIST IS AN INT IT RETURNS THE HIGHES M MODES
        m2plot_list = log_amplitude_corrected_list[:,-1].argsort()[-modes_list:]
    elif type(modes_list) == list:
        # IF MODES_LIST IS A LIST IT RETURNS THOSE MODES
        m2plot_list = np.copy(modes_list)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(m2plot_list[i]/np.max(m2plot_list)) for i in range(len(m2plot_list))]
    for i in range(len(m2plot_list)):
        plt.plot(dt*iterations[iter_range_left:iter_range_right], log_amplitude_corrected_list[m2plot_list[i],:] , linestyle='-', label = f'm = {m2plot_list[i]}', color=colors[i])
    plt.xlabel("time (s)")
    plt.ylabel(f"log |{variable_name}|")
    matplotx.line_labels()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Log Modulus of Fourier Coefficient vs Iterations (Radius={round(r_0*100,2)}+-{r_width_average/N_r*R_max*100} cm)")
    return fig_amplitude




# =============================================================================
# SHEAR LAYER
# =============================================================================

def get_shear_layer_plot(avg_Variable_straight_frame, std_Variable_straight_frame, R_max, upper_lim, bottom_lim):
    fill_down, fill_up = avg_Variable_straight_frame - std_Variable_straight_frame, avg_Variable_straight_frame + std_Variable_straight_frame
    r4plot = np.linspace(0.001*R_max,R_max,len(avg_Variable_straight_frame))
    
    fig_shear, ax = plt.subplots(figsize=(10,8))
    plt.title(r'Shear layer profile from drift velocity (avg \pm std)')
    plot_avg = ax.plot(r4plot, avg_Variable_straight_frame, color='b')
    plot_std = ax.fill_between(r4plot, fill_down, fill_up, color='b', alpha=0.3)
    ax.set_ylim(bottom_lim, upper_lim)
    ax.set_xlabel('r (cm)')
    ax.set_ylabel(r'$U_e$ (m/s)')
    iter_text = ax.text(0.05, 0.95, "iter = ", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes , ha="left")
    return fig_shear, plot_avg, plot_std, iter_text


def get_shear_layer_anim(Variable_straight,R_max,nframes,iter_per_step,frame_initial):
    
    avg_Variable_straight = [np.mean(Variable_straight[i],axis=1)for i in range(len(Variable_straight))]
    std_Variable_straight = [np.std(Variable_straight[i],axis=1) for i in range(len(Variable_straight))]
    
    fill_down = [avg_Variable_straight[i] - std_Variable_straight[i] for i in range(len(Variable_straight))]
    fill_up = [avg_Variable_straight[i] + std_Variable_straight[i] for i in range(len(Variable_straight))]
    r4plot = np.linspace(0.001*R_max,R_max,len(avg_Variable_straight[0]))
    
    upper_lim = np.max(fill_up)
    bottom_lim = np.min(fill_down)
    
    iteration = 0
    fig_shear, plot_avg, plot_std, iter_text_shear = get_shear_layer_plot(avg_Variable_straight[iteration], std_Variable_straight[iteration], R_max, upper_lim, bottom_lim)
    
    def update_shear(frame):
        iter_text_shear.set_text(f"iter = {frame*iter_per_step + frame_initial*iter_per_step}")
        plot_avg[0].set_ydata(avg_Variable_straight[frame])
        path = plot_std.get_paths()[0]
        vertices = path.vertices.copy()
        n = len(r4plot)

        vertices[0,1] = fill_down[frame][0]
        vertices[1:n+1, 1] = fill_up[frame]
        vertices[n+1, 1] = fill_down[frame][-1]
        vertices[n+2:2*n+2, 1] = fill_down[frame][::-1]
        vertices[-1, 1] = fill_down[frame][0]
        
        path.vertices = vertices
        return plot_avg, plot_std, iter_text_shear
    
    anim_shear = animation.FuncAnimation(fig=fig_shear, func=update_shear, frames=nframes, interval=30)
    return anim_shear







# =============================================================================
# INSTABILITY CRITERIA FOR ARBITRARY PROFILE
# =============================================================================


def get_instability_data(Ue_profile_raw,r_disk, R_max, m_values, dy):
    N = len(Ue_profile_raw)
    N_interior = N - 2
    
    U_profile = gaussian_filter(Ue_profile_raw, 20)
    
    U_double_prime_profile = np.gradient(np.gradient(U_profile, dy), dy)
    
    U_vec = U_profile[1:-1]
    U_pp_vec = U_double_prime_profile[1:-1]
    
    D2 = (np.diag(np.ones(N_interior - 1), 1) -
          2 * np.diag(np.ones(N_interior), 0) +
          np.diag(np.ones(N_interior - 1), -1)) / dy**2

    I = np.eye(N_interior)

    def eigenvalue_problem(k, U_vec, U_pp_vec):
        M = D2 - k**2 * I
        L = np.einsum('i,ij->ij', U_vec, M) - np.diag(U_pp_vec)
        eigenvals = eigvals(L, M)
        return eigenvals
    
    k_values = m_values / r_disk
    max_growth_rates = np.empty_like(k_values)
    
    for i, k in enumerate(k_values):
        e_vals = eigenvalue_problem(k, U_vec, U_pp_vec)
        growth_rates = k * np.imag(e_vals)
        max_growth_rates[i] = np.max(growth_rates)
    
    return max_growth_rates


def write_instability_data(directory, variable_name, iterations, instability_data):
    with open(os.path.join(directory,variable_name,"instability_data.txt"), "w") as file:    
        for i in range(len(iterations)):
            values_str = "\t".join(map(str, instability_data[i,:]))  # Convert values list to a tab-separated string
            file.write(f"{iterations[i]}\t{values_str}\n")
    print('Data stored at txt file')
    
    
def read_instability_data(directory, variable_name):
    with open(os.path.join(directory,variable_name,"instability_data.txt"), "r") as file:
        lines = file.readlines()[:]
    
    iterations, values = [], []
    for line in lines:
        parts = line.strip().split("\t")  # Split by tab
        iterations.append(int(parts[0]))  # First part is the iteration
        values.append([float(x) for x in parts[1:]])  # Remaining parts are values
    values = np.array(values)
    return values, iterations



def get_instability_plot(max_growth_rates, m_values, upper_lim, bottom_lim):
    fig_inst, ax = plt.subplots(figsize=(10,8))
    plt.title(r'Instability profile')
    plot_inst = ax.plot(m_values, max_growth_rates)
    ax.set_ylim(bottom_lim, upper_lim)
    ax.set_xlabel('mode number (m)')
    ax.set_ylabel('max growth rate')
    iter_text = ax.text(0.05, 0.95, "iter = ", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes , ha="left")
    max_text = ax.text(0.05, 0.85, "iter = ", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes , ha="left")
    return fig_inst, plot_inst, iter_text, max_text


def get_instability_anim(instability_data,m_values,nframes,iter_per_step,frame_initial):
    
    upper_lim = np.max(instability_data)*1.05
    bottom_lim = np.min(instability_data)*0.95
    
    iteration = 0
    fig_inst, plot_inst, iter_text_inst, max_text_inst = get_instability_plot(instability_data[iteration,:], m_values, upper_lim, bottom_lim)
    
    def update_inst(frame):
        iter_text_inst.set_text(f"iter = {frame*iter_per_step + frame_initial*iter_per_step}")
        max_text_inst.set_text(f"max at m ={np.argmax(instability_data[frame])}")
        plot_inst[0].set_data(m_values,instability_data[frame])
        return plot_inst, iter_text_inst, max_text_inst
    
    anim_inst = animation.FuncAnimation(fig=fig_inst, func=update_inst, frames=nframes, interval=30)
    return anim_inst
