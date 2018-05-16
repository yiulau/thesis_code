from adapt_util.update_list_util import *
def return_update_lists(adapter_meta,adapter_setting):
    # returns three lists
    # fast list , medium list , slow list

    window_size = adapter_setting["window_size"]
    tune_l = adapter_meta.tune_l
    tune_fast = adapter_meta.tune_fast
    tune_medium = adapter_meta.tune_medium
    tune_slow = adapter_meta.tune_slow

    if tune_fast == True:
        ini_buffer = adapter_setting["ini_buffer"]
        end_buffer = adapter_setting["end_buffer"]
    if tune_medium == True:
        min_medium_updates = adapter_setting["min_medium_updates"]
    if tune_slow == True:
        min_slow_updates = adapter_meta.adapt["min_slow_updates"]


    #print(window_size,ini_buffer,end_buffer,min_medium_updates,tune_l)
    #exit()
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
                #print("yes")
                out = f(tune_l)
                #print(out)
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
                out = ([],[],[])

    iters_dict = {"fast":out[0],"medium":out[1],"slow":out[2]}
    return(iters_dict)