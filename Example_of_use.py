import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import correlate, resample
import sys
import matplotlib.animation as animation
from numba import njit
import matplotx
import pickle 
import matplotlib as mpl






os.chdir(r"")      # path to internship_functions.py folder
import internship_functions as ifc



# =============================================================================
# DATA COLLECTION
# =============================================================================



folder = r""     # path to work directory
case = r""       # folder name inside working directory
directory = os.path.join(folder,case)
directory_dump = os.path.join(directory,'dump files') # data stored in "dump files" folder

B = 0.075 # Tesla
iter_per_step = 10
time_per_iter =  3.57356e-12/B

Nx, Ny, Lx, Ly = 257, 257, 0.02, 0.02
R_max = min(Lx, Ly) / 2
N_r, N_theta = 1000, 1000

mode_plot_limit = 51
#mode_plot_limit = len(fourier_variable[0,:])




variable_names = ['n_e','n_i','phi','Ue']
for variable_name in variable_names:
    os.makedirs(os.path.join(directory,variable_name), exist_ok=True)


Data_history = []

# data_load_option = 'h5'
data_load_option = 'txt'

frame_initial = 1
nframes = 85
for i in range(frame_initial,nframes+frame_initial):
    print(f'    state {i} / {nframes+frame_initial-1}')
    
    if data_load_option == 'h5':
    
        if 'phi' in variable_names or 'Ue' in variable_names:
            filename_E = "E_i"+str(i)+".h5"
            path_to_file_E = os.path.join(directory_dump,filename_E)
            E_data, domain = ifc.read_h5_E(path_to_file_E)
            
            Nx, Ny, Lx, Ly = domain
            
            E_x , E_y = E_data[0,:,:] , E_data[1,:,:]
            Ue = np.hypot(E_x,E_y)/B
            
            if 'phi' in variable_names:
                dx , dy = Lx/(Nx-1) , Ly/(Ny-1)
                rhs = np.zeros((Nx, Ny))
                rhs[1:-1, 1:-1] = (E_x[1:-1, 1:-1] - E_x[:-2, 1:-1]) / dx + (E_y[1:-1, 1:-1] - E_y[1:-1, :-2]) / dy
                phi = np.zeros((Nx, Ny))
                phi = -ifc.multigrid_solver(phi, rhs, dx, dy, n_vcycles=3)
        
        if ('n_e' in variable_names) or ('n_i' in variable_names) or ('n_e' in variable_names):
            filename_state = "test1_i"+str(i)+".h5"
            path_to_file_state = os.path.join(directory_dump,filename_state)
            data_read = ifc.read_h5_state(path_to_file_state)
            
            r_data_e = data_read['xelectron']['r']
            r_data_i = data_read['xion0']['r']
            v_data_e = data_read['xelectron']['v']
            v_data_i = data_read['xion0']['v']
            
            n_e = ifc.get_density(r_data_e, Nx, Ny, Lx, Ly)
            n_i = ifc.get_density(r_data_i, Nx, Ny, Lx, Ly)
            
            vmod_e = ifc.get_vmod(r_data_e, v_data_e, Nx, Ny, Lx, Ly)
            vmod_i = ifc.get_vmod(r_data_i, v_data_i, Nx, Ny, Lx, Ly)
            
        # =============================================================================
        # Analyze the magnitudes to get valuable information
        # =============================================================================
        
        Data = {}
        for variable_name in variable_names:
            
            ### Perform the Fourier analysis on all the variables
            Variable_raw = globals()[variable_name]
            Variable = gaussian_filter(Variable_raw, 3)
            Variable_straight = ifc.Get_Variable_straight(Variable,N_r,N_theta,R_max)
            fourier_variable = ifc.Get_fourier_variable(Variable_straight,N_r)
            
            Data[variable_name] = {'var': Variable.astype(np.float32), 'polar': Variable_straight.astype(np.float32), 'fft': fourier_variable.astype(np.float32)}
            
        Data_history.append(Data)
            
    elif data_load_option == 'txt':
        Raw_data = {}
        for variable_name in variable_names:
            m_data, n_data, _, _, Raw_Data_flat = ifc.read_txt_file(os.path.join(directory_dump, f'{variable_name}_i{i*iter_per_step}.txt'))
            Variable_raw = np.zeros((Nx,Ny))
            Variable_raw[m_data,n_data] = Raw_Data_flat
            
            Variable = gaussian_filter(Variable_raw, 0)
            Variable_straight = ifc.Get_Variable_straight(Variable,N_r,N_theta,R_max)
            fourier_variable = ifc.Get_fourier_variable(Variable_straight,N_r)
            
            Raw_data[variable_name] = {'var': Variable.astype(np.float32), 'polar': Variable_straight.astype(np.float32), 'fft': fourier_variable.astype(np.float32)}
            
        Data_history.append(Raw_data)

    
        


Data_ranges = {}
for variable_name in variable_names:
    Data_ranges[variable_name] = {}
    for representation in ['var', 'polar', 'fft']:

        Max_data = np.max([d[variable_name][representation] for d in Data_history])
        Min_data = np.min([d[variable_name][representation] for d in Data_history])
        
        if representation == 'fft':
            for d in Data_history:
                d[variable_name]['fft'] /= Max_data
    
        Data_ranges[variable_name][representation] = [Max_data, Min_data]


### RAW DATA STORAGE TO AVOID RE-PROCESSING THE DATA

with open(os.path.join(folder, 'Data_history.pkl'), 'wb') as f:
    pickle.dump((Data_history, Data_ranges), f)
    
with open(os.path.join(folder, 'Data_history.pkl'), 'rb') as f:
    Data_history, Data_ranges = pickle.load(f)

### RAW DATA REPRESENTATION
    # In cartesian and polar coordinates
    # The time-evolution of the fourier spectrum

def update_var(frame):
    # for each frame, update the data stored on each artist.
    iter_text_var.set_text(f"iter = {frame*iter_per_step + frame_initial*iter_per_step}")
    plot_variable.set_data(Variable[frame].T)
    return plot_variable, iter_text_var

def update_polar(frame):
    # for each frame, update the data stored on each artist.
    iter_text_polar.set_text(f"iter = {frame*iter_per_step + frame_initial*iter_per_step}")
    plot_polar.set_data(Variable_straight[frame].T)
    return plot_polar, iter_text_polar

def update_fft(frame):
    # for each frame, update the data stored on each artist.
    iter_text_fft.set_text(f"iter = {frame*iter_per_step + frame_initial*iter_per_step}")
    plot_fft.set_data(gaussian_filter(fourier_variable[frame][:,:mode_plot_limit].T, 5, axes=1))
    return plot_fft, iter_text_fft

for variable_name in variable_names:
    print(f'Animating {variable_name} ...')
    Variable = [d[variable_name]['var'] for d in Data_history]
    All_Variable_max, All_Variable_min = Data_ranges[variable_name]['var']
    fig_variable, plot_variable, iter_text_var = ifc.Get_plot_variable(Variable[0].T, variable_name, Lx, Ly, All_Variable_min, All_Variable_max)

    Variable_straight = [d[variable_name]['polar'] for d in Data_history]
    All_Variable_straight_max, All_Variable_straight_min = Data_ranges[variable_name]['polar']
    fig_polar, plot_polar, iter_text_polar = ifc.Get_plot_polar(Variable_straight[0].T, variable_name, R_max, All_Variable_straight_min, All_Variable_straight_max)
        
    fourier_variable = [d[variable_name]['fft'] for d in Data_history]
    fig_fft, plot_fft, iter_text_fft = ifc.Get_plot_fft(fourier_variable[0][:,:mode_plot_limit].T, mode_plot_limit, variable_name, R_max)

    
    anim_var = animation.FuncAnimation(fig=fig_variable, func=update_var, frames=nframes, interval=30)
    anim_var.save(os.path.join(directory,variable_name,variable_name+'_var.mp4'), fps=15, writer='ffmpeg')
    
    anim_polar = animation.FuncAnimation(fig=fig_polar, func=update_polar, frames=nframes, interval=30)
    anim_polar.save(os.path.join(directory,variable_name,variable_name+'_polar.mp4'), fps=15, writer='ffmpeg')
    
    anim_fft = animation.FuncAnimation(fig=fig_fft, func=update_fft, frames=nframes, interval=30)
    anim_fft.save(os.path.join(directory,variable_name,variable_name+'_fft.mp4'), fps=15, writer='ffmpeg')
    
    plt.close(fig_variable)
    plt.close(fig_polar)
    plt.close(fig_fft)
    

#print(f"Total size: {ifc.deep_getsizeof(Data_history)/1e6} Mb")






# =============================================================================
# MODE AMPLITUDE EVOLUTION STUDY
# =============================================================================



r_0 = 0.0075             # Given radius in m
r_width_average = 100    # In points, not in m

m_list = list(range(0,501))     # Given mode numbers

variable_name = 'phi'

log_amplitude_list, iterations = ifc.get_amplitude_evolution(Data_history, nframes, iter_per_step, m_list, r_0, r_width_average, variable_name, R_max, N_r, Data_ranges)


# ifc.write_amplitude_evo(directory, variable_name, iterations, log_amplitude_list)
log_amplitude_list, iterations = ifc.read_amplitude_evo(directory, variable_name)


modes_list = 3
iter_range = 0,800
normalization_flag = False

plt.rcParams.update({'font.size': 15})
plot_amplitude = ifc.get_amplitude_evolution_plot(gaussian_filter(log_amplitude_list,sigma=5,axes=1), np.array(iterations), modes_list, iter_range, normalization_flag, variable_name, r_0, r_width_average, R_max, N_r, 1)
plt.show()


plt.figure()
plt.scatter(m_list[1:],log_amplitude_list[m_list[1:],-1])
#plt.xticks(m_list)
plt.show()


log_amplitude_dic = {}
for variable_name in variable_names:
    log_amplitude_list, iterations = ifc.get_amplitude_evolution(Data_history, nframes, iter_per_step, m_list, r_0, r_width_average, variable_name, R_max, N_r, Data_ranges)
    ifc.write_amplitude_evo(directory, variable_name, iterations, log_amplitude_list)
    log_amplitude_dic[variable_name] = log_amplitude_list
    

modes_list = [13,18]
iter_range = 0,-1
normalization_flag = False

fig, ax = plt.subplots(1,4)

ind = 0
for variable_name in variable_names:
    log_amplitude_list = log_amplitude_dic[variable_name]
    iter_range_left, iter_range_right = iter_range
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
        ax[ind].plot(iterations[iter_range_left:iter_range_right], log_amplitude_corrected_list[m2plot_list[i],:] , linestyle='-', label = f'm = {m2plot_list[i]}', color=colors[i])
    ax[ind].set_xlabel("Iterations")
    ax[ind].set_ylabel(f"log |{variable_name}|")
    ax[ind].legend()
    #matplotx.line_labels()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[ind].set_title(f"Log(|fft_amp|) vs Iterations (Radius={round(r_0*100,2)}+-{r_width_average/N_r*R_max*100} cm)")
    ind += 1

plt.show()



modes_list = list(range(1,51))
fig, ax = plt.subplots(1,4)

ind = 0
for variable_name in variable_names:
    log_amplitude_last = log_amplitude_dic[variable_name][:,-1]
    for i in range(len(m2plot_list)):
        ax[ind].plot(modes_list, gaussian_filter(log_amplitude_last[modes_list],2), c='k')
        ax[ind].vlines([13,18], ymin = np.min(gaussian_filter(log_amplitude_last[modes_list],1)), ymax = np.max(gaussian_filter(log_amplitude_last[modes_list],1)), colors='b', ls='--')
    ax[ind].set_xlabel("mode number (m)")
    ax[ind].set_ylabel(f"final log |{variable_name}|")
    ax[ind].set_title(f"final Log(|fft_amp|) for each mode (Radius={round(r_0*100,2)}+-{r_width_average/N_r*R_max*100} cm)")
    ind += 1

plt.show()








# =============================================================================
# B STUDY PLOT
# =============================================================================


folder = r"C:\Users\ivana\Desktop\FusionEP Master año 1\Semester 2\Internship\0 Computational analysis\Fourier analysis of perturbations\B_study"
cases = [r"B_005", r"B_01", r"B_015", r"B_02"]
B_values = [0.05, 0.1, 0.15, 0.2]

variable_names = ['n_e','n_i','phi','Ue']

Nx, Ny, Lx, Ly = 257, 257, 0.02, 0.02
R_max = min(Lx, Ly) / 2
N_r, N_theta = 1000, 1000

log_amplitude_all_dic = {}
for i in range(len(cases)):
    case_B = cases[i]
    directory = os.path.join(folder,case_B)
    directory_dump = os.path.join(directory,'dump_data')

    B = B_values[i]
    time_per_iter =  3.57356e-12/B
    
    log_amplitude_dic = {}
    for variable_name in variable_names:
        log_amplitude_list, iterations = ifc.read_amplitude_evo(directory, variable_name)
        log_amplitude_dic[variable_name] = log_amplitude_list
        
    log_amplitude_all_dic[case_B] = log_amplitude_dic
    log_amplitude_all_dic[case_B]['iter'] = iterations
    log_amplitude_all_dic[case_B]['dt'] = time_per_iter
    
    
    

iter_range = 0,-1
modes_list = 1
r_0 = 0.0075
r_width_average = 100

fig, ax = plt.subplots(1,2)

variable_names = ['n_e', 'n_i']

ind = 0
for variable_name in variable_names:
    for case_B in cases:
        log_amplitude_list = log_amplitude_all_dic[case_B][variable_name]
        iterations = log_amplitude_all_dic[case_B]['iter']
        time_per_iter = log_amplitude_all_dic[case_B]['dt']
        iter_range_left, iter_range_right = iter_range

        log_amplitude_corrected_list = log_amplitude_list[:,iter_range_left:iter_range_right]
        
        m2plot_list = log_amplitude_corrected_list[:,-1].argsort()[-modes_list:]
    
        for i in range(len(m2plot_list)):
            ax[ind].plot(time_per_iter * np.array(iterations[iter_range_left:iter_range_right]), gaussian_filter(log_amplitude_corrected_list[m2plot_list[i],:] , 2) , linestyle='-', label = f'm = {m2plot_list[i]} {case_B}')
        ax[ind].set_xlabel("Time (s)")
        ax[ind].set_ylabel(f"log |{variable_name}|")
        ax[ind].legend()
        #matplotx.line_labels()
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[ind].set_title(f"Log(|fft_amp|) vs Iterations (Radius={round(r_0*100,2)}+-{r_width_average/N_r*R_max*100} cm)")
    ind += 1

plt.show()


plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()
for i in range(len(cases)):
    case_B = cases[i]
    ax.scatter(B_values[i], np.array(log_amplitude_all_dic[case_B]['iter'])[-1]*log_amplitude_all_dic[case_B]['dt'])

ax.set_title='Time required to reach the instability saturation'
ax.set_xlabel('B (T)')
ax.set_ylabel('time (s)')
ax.set_xticks(B_values)
plt.show()
    








# =============================================================================
# GET SHEAR LAYER PROFILE FROM DRIFT VELOCITY
# =============================================================================


variable_name = 'Ue'
Ue_history = [d['Ue']['polar'] for d in Data_history]
Variable_straight = Ue_history

anim_shear = ifc.get_shear_layer_anim(Variable_straight,R_max,nframes,iter_per_step,frame_initial)
anim_shear.save(os.path.join(directory,variable_name,variable_name+'_shear.mp4'), fps=15, writer='ffmpeg')











# =============================================================================
# GET INSTABILITY CRITERIA FOR ARBITRARY UE PROFILE
# =============================================================================



Ue_history = [d['Ue']['polar'] for d in Data_history]
Variable_straight = Ue_history
variable_name = 'Ue'
r_0 = 0.0075

m_values =  np.linspace(1, 50, 50)
r4plot = np.linspace(0.001*R_max,R_max,len(Variable_straight[0]))
dy = r4plot[1] - r4plot[0]


resample_size=100
avg_Variable_straight = [resample(np.sqrt(np.mean(Variable_straight[i],axis=1)*2/8.8541878128e-12)/B , resample_size)   for i in range(len(Variable_straight))]

instability_data = np.zeros((len(avg_Variable_straight),len(m_values)))

for i in range(len(avg_Variable_straight)):
    print(f'iteración {i}/{len(avg_Variable_straight)-1}')
    Ue_profile_raw = avg_Variable_straight[i]
    max_growth_rates = ifc.get_instability_data(Ue_profile_raw, r_0, R_max, m_values,dy)
    instability_data[i,:] = max_growth_rates




# ifc.write_instability_data(directory, variable_name, list(range(frame_initial,nframes+1)), instability_data)
instability_data, iterations = ifc.read_instability_data(directory, variable_name)



instability_data_smooth = gaussian_filter(instability_data, 3, axes=0)

anim_inst = ifc.get_instability_anim(instability_data_smooth, m_values, nframes, iter_per_step, frame_initial)
anim_inst.save(os.path.join(directory,variable_name,variable_name+'_inst.mp4'), fps=15, writer='ffmpeg')
