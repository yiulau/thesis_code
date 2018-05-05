from adapt_list.update_list_util import *
def return_update_lists(tune_l, tune_fast, tune_medium, tune_slow, ini_buffer=75, end_buffer=50, window_size=25,
                        min_medium_updates=10,min_slow_updates=2):
    # returns three lists
    # fast list , medium list , slow list


    if tune_fast == True:
        if tune_medium== True:
            if tune_slow==True:
                out = fmsmf(window_size,ini_buffer,end_buffer,min_medium_updates,min_slow_updates,tune_l)
            else:
                out = fmf(window_size,ini_buffer,end_buffer,min_medium_updates,tune_l)
        else:
            if tune_slow==True:
                out = fsf(window_size,ini_buffer,end_buffer,min_slow_updates,tune_l)
            else:
                out = f(tune_l)
    else:
        if tune_medium:
            if tune_slow:
                out = msm(window_size,min_medium_updates,min_slow_updates,tune_l)
            else:
                out = m(window_size,min_medium_updates,tune_l)
        else:
            if tune_slow:
                out = s(window_size,min_slow_updates,tune_l)
            else:
                out = [[],[],[]]
    return(out)