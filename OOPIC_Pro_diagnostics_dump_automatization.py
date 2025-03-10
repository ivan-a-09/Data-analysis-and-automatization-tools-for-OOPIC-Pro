from pywinauto import Application
from pywinauto import send_keys
import time

app = Application(backend='uia').connect(title_re=".*OOPIC Pro.*")
main_window = app.window(title_res="-*OOPIC Pro.*")


iter4dump = 10      ### NUMBER OF ITERATIOS BETWEEN SAVE FILES
maxiter = 2000      ### TOTAL NUMBER OF ITERATIONS FOR THE SIMULATION

for iteration in range(1*iter4dump,maxiter+iter4dump,iter4dump):

    main_window.set_focus()
    ### OPEN THE RUNTIME OPTIONS MENU
    send_keys('%F')
    send_keys('{RIGHT}')
    send_keys('{RIGHT}')
    send_keys('{RIGHT}')
    send_keys('{DOWN}')
    send_keys('{ENTER}')
    ### ESTABLISH THE NUMBER OF ITERATIONS TO RUN BEFORE SAVING THE DATA
    main_window.click_input(coords=(210,150))
    for i in range(10):
        send_keys('{BACKSPACE}')
    send_keys(str(iteration))
    send_keys('{ENTER}')
    ### RUN FOR THE "iter4dump" ITERATIONS
    main_window.set_focus()
    send_keys('{%R}')
    ### WAIT UNTIL ALL THE ITERATIONS HAVE BEEN COMPLETED
    #   This time has to be estimated beforehand in order to optimize the process
    time.sleep(30)
    ### CLOSE THE WARNING POPUP
    send_keys('{ENTER}')
    ### CLOSE THE RUNTIME OPTIONS WINDOW
    send_keys('{ENTER}')

    ### LOOP FOR EVERY DIAGNOSTICS AND SAVE IT
    diag_list = ['n_e']
    for diag_var in diag_list:
        diag_window_name = ".*" + diag_var + ".*"
        diag_window = app.window(title_re=diag_window_name)
        diag_window.set_focus()
        ### ENTER THE SAVING DATA MENU
        send_keys('%F')
        send_keys('{DOWN}')
        send_keys('{DOWN}')
        send_keys('{ENTER}')
        ### INPUT THE MAGNITUDE NAME AND ITERATION NUMBER
        send_keys(diag_var+'_i'+str(iteration))
        send_keys('{ENTER}')
