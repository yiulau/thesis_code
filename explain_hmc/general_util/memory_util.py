import pickle,os
def to_pickle_memory(object):
    # returns object's size in memory, in mb
    with open("temp_object_volume.pkl", 'wb') as f:
        pickle.dump(object, f)
    size = os.path.getsize("./temp_object_volume.pkl") / (1024. * 1024)
    os.remove("./temp_object_volume.pkl")
    return(size)