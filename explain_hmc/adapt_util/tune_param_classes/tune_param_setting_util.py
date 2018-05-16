def default_par_type(name):
    default_dict = {"epsilon": "fast", "evolve_L": "medium", "evolve_t": "medium", "alpha": "medium",
                          "xhmc_delta": "medium",
                          "diag_cov": "slow", "cov": "slow"}

    return(default_dict[name])
def dual_default_arguments(name):
    output ={"name":name,"target":0.65,"gamma":0.05,"t_0":10,
             "kappa":0.75,"obj_fun":"accept_rate","par_type":default_par_type(name)}

    return(output)

available_obj_funs = ("accept_rate","ESJD","ESJD_g_normalized")
def opt_default_arguments(name_list,par_type,bounds_list=None):
    # if leave bounds undefined it will be initiated by default
    # either leave all bounds out or provide all of them
    if bounds_list is None:
        output = {"obj_fun":"ESJD","par_type":par_type,"name":"opt","params_tuple":tuple(name_list)}
    else:
        assert len(name_list)==len(bounds_list)
        output = {"obj_fun":"ESJD","par_type":par_type,"name":"opt","params_tuple":tuple(name_list),
                  "bounds_tuple":tuple(bounds_list)}
    return(output)
def adapt_cov_default_arguments(par_type):
    return({"par_type":par_type,"name":"cov"})
def other_default_arguments():
    output = {"maximum_second_per_sample":0.5}
    return(output)
def tuning_settings(dual_arguments,opt_arguments,adapt_cov_arguments,other_arguments):


    fast_tune_setting_dict = {"dual":[],"opt":[]}
    medium_tune_setting_dict = {"dual":[],"opt":[],"adapt_cov":[]}
    slow_tune_setting_dict = {"dual":[],"opt":[],"adapt_cov":[]}
    dict_par_name = {}
    for obj in dual_arguments:
        if obj["par_type"]=="fast":
            fast_tune_setting_dict["dual"].append(obj)
        elif obj["par_type"]=="medium":
            medium_tune_setting_dict["dual"].append(obj)
        elif obj["par_type"]=="slow":
            slow_tune_setting_dict["dual"].append(obj)
        else:
            raise ValueError("should not happen")
        dict_par_name.update({obj["name"]:obj})

    for obj in opt_arguments:
        if obj["par_type"]=="fast":
            fast_tune_setting_dict["opt"].append(obj)
        elif obj["par_type"]=="medium":
            medium_tune_setting_dict["opt"].append(obj)
        elif obj["par_type"]=="slow":
            slow_tune_setting_dict["opt"].append(obj)
        else:
            raise ValueError("should not happen")
        for i in range(len(obj["params_tuple"])):
            if "bounds_tuple" in obj:
                dict_par_name.update({obj["params_tuple"][i]:{"bounds":obj["bounds_list"][i],"par_type":obj["par_type"]}})
            else:
                dict_par_name.update({obj["params_tuple"][i]:{"par_type":obj["par_type"]}})
    for obj in adapt_cov_arguments:
        if obj["par_type"]=="medium":
            medium_tune_setting_dict["adapt_cov"].append(obj)
        elif obj["par_type"]=="slow":
            slow_tune_setting_dict["adapt_cov"].append(obj)
        else:
            raise ValueError("should not happen")
        dict_par_name.update({"cov":{"par_type":obj["par_type"]}})

    others_dict = other_arguments
    dict_par_type = {"fast":fast_tune_setting_dict,"medium":medium_tune_setting_dict,"slow":slow_tune_setting_dict}
    out = {"par_type":dict_par_type,"par_name":dict_par_name,"others":others_dict}
    # dict_par_name is dual only
    # at this point should look at all tuning parameters and fill in by default values any yet to be specified variables




    # for param,val in tune_dict:
    #     if param.par_type=="fast":
    #         if val == "dual":
    #
    #             tlist = fast_tune_setting_dict["dual"]
    #             exists = False
    #             for arg in tlist:
    #                 if param.name in arg:
    #                     exists = True
    #             tlist.append(dual_default_arguments(name=param.name))
    #
    #         elif val=="opt":
    #             param_names = slow_dict and opt_dict
    #             chosen_list = {}
    #             for param in param_names:
    #                 chosen_list.update({param,False})
    #             tlist = fast_tune_setting_dict["opt"]
    #             for arg in tlist:
    #                 if param.name in arg["params"]:
    #                     chosen_list.update({param,True})
    #
    #
    #     if param.par_type=="medium":
    #         if val == "dual":
    #             tlist = medium_tune_setting_dict["dual"]
    #         elif val=="opt":
    #             tlist = medium_tune_setting_dict["opt"]
    #         elif val=="adapt":
    #             tlist = medium_tune_setting_dict["adapt"]
    #     if param.par_type=="slow":
    #         if val == "dual":
    #             tlist = slow_tune_setting_dict["dual"]
    #         elif val=="opt":
    #             tlist = slow_tune_setting_dict["opt"]
    #         elif val=="adapt":
    #             tlist = slow_tune_setting_dict["adapt"]
    #

    return(out)

def default_adapter_setting():
    out = {"ini_buffer":75,"end_buffer":50,"window_size":25,"min_medium_updates":10,"min_slow_updates":2}
    return(out)