# case1
# fmsmf
def fmsmf(window_size=25,ini_buffer=75,end_buffer=50,min_medium_updates=5,min_slow_updates=2,tune_l=250):
    case = 1
    fast_list = []
    medium_list = []
    slow_list = []
# if case==1:
    slow_window_size = window_size
    min_length = ini_buffer + end_buffer + window_size * (round(min_medium_updates * (1.75)))
    n = (2 ** (min_slow_updates + 1) - 1) * window_size
    min_length = min_length + n
    if tune_l < min_length:
        raise ValueError("warm up not long enough")
    else:
        counter = 0
        overshoots = False
        while overshoots == False:
            if counter < ini_buffer:
                fast_list.append(counter)
                counter += 1
            elif counter < ini_buffer + min_medium_updates * window_size:
                #print("yes")
                #print("counter")
                #print(ini_buffer + min_medium_updates * window_size)
                temp = counter +  window_size
                #print(temp)
                #print(ini_buffer + min_medium_updates * window_size)
                if  temp < ini_buffer + min_medium_updates * window_size:
                    medium_list.append(counter)
                    fast_list.append(counter)
                    counter += window_size
                else:
                    counter = ini_buffer + min_medium_updates * window_size


            elif counter < tune_l - end_buffer - round(min_medium_updates * window_size * 3 / 4):
                temp = counter +  slow_window_size
                if temp < tune_l - end_buffer - round(min_medium_updates * window_size * 3 / 4):
                    slow_list.append(counter)
                    medium_list.append(counter)
                    fast_list.append(counter)
                    counter += slow_window_size
                    slow_window_size *= 2
                else:
                    counter = tune_l - end_buffer - round(min_medium_updates * window_size * 3 / 4)
                    slow_list.append(counter)

            elif counter < tune_l - end_buffer:
                temp = counter + window_size
                if temp < tune_l - end_buffer:
                    medium_list.append(counter)
                    fast_list.append(counter)
                    counter += window_size
                else:
                    counter = tune_l - end_buffer
                    medium_list.append(counter)
            elif counter < tune_l:
                fast_list.append(counter)
                counter += 1
            else:
                overshoots = True
    return(fast_list,medium_list,slow_list)

#out = case1(tune_l=2500,window_size=25)
#print(out[0])
#print(out[1])
#print(out[2])
# case2
def fmf(window_size=25,ini_buffer=75,end_buffer=50,min_medium_updates=5,tune_l=250):
    case = 2
    fast_list = []
    medium_list = []
    slow_list = []
# if case==1:
    slow_window_size = window_size
    min_length = ini_buffer + end_buffer + window_size * round(min_medium_updates*(1.75))


    if tune_l < min_length:
        raise ValueError("warm up not long enough")
    else:
        counter = 0
        overshoots = False
        while overshoots == False:
            if counter < ini_buffer:
                fast_list.append(counter)
                counter += 1

            elif counter < tune_l - end_buffer:
                temp = counter +  window_size
                if temp < tune_l - end_buffer:
                    medium_list.append(counter)
                    fast_list.append(counter)
                    counter += window_size
                else:
                    counter = tune_l - end_buffer
                    medium_list.append(counter)

            elif counter < tune_l:
                fast_list.append(counter)
                counter += 1
            else:
                overshoots = True
    return(fast_list,medium_list,slow_list)


#out = case2(tune_l=500,window_size=25)
#print(out[0])
#print(out[1])
#print(out[2])

# case3
def fsf(window_size=25,ini_buffer=75,end_buffer=50,min_slow_updates=2,tune_l=250):
    case = 1
    fast_list = []
    medium_list = []
    slow_list = []
# if case==1:
    slow_window_size = window_size
    min_length = ini_buffer + end_buffer
    n = (2 ** (min_slow_updates + 1) - 1) * window_size
    min_length = min_length + n

    if tune_l < min_length:
        raise ValueError("warm up not long enough")
    else:
        counter = 0
        overshoots = False
        while overshoots == False:
            if counter < ini_buffer:
                fast_list.append(counter)
                counter += 1

            elif counter < tune_l - end_buffer:
                print("counter {}".format(counter))
                print(slow_window_size)
                temp = counter + slow_window_size
                if temp < tune_l - end_buffer :
                    slow_list.append(counter)
                    fast_list.append(counter)
                    counter += slow_window_size
                    slow_window_size *= 2
                else:
                    counter = tune_l - end_buffer
                    slow_list.append(counter)


            elif counter < tune_l:
                fast_list.append(counter)
                counter += 1
            else:
                overshoots = True
    return(fast_list,medium_list,slow_list)

#out = case3(tune_l=2500,window_size=25)
#print(out[0])
#print(out[1])
#print(out[2])

# case 4
# msm
def msm(window_size=25,min_medium_updates=5,min_slow_updates=2,tune_l=250):
    case = 4
    fast_list = []
    medium_list = []
    slow_list = []
# if case==1:
    slow_window_size = window_size
    min_length = window_size * (round(min_medium_updates * (1.75)))
    n = (2 ** (min_slow_updates + 1) - 1) * window_size
    min_length = min_length + n

    if tune_l < min_length:
        raise ValueError("warm up not long enough")
    else:
        counter = 0
        overshoots = False
        while overshoots == False:
            if counter <  min_medium_updates * window_size:
                temp = counter +  window_size
                if  temp <  min_medium_updates * window_size:
                    medium_list.append(counter)
                    counter += window_size
                else:
                    counter =  min_medium_updates * window_size
            elif counter < tune_l  - round(min_medium_updates * window_size * 3 / 4):
                temp = counter +  slow_window_size
                if temp < tune_l- round(min_medium_updates * window_size * 3 / 4):
                    slow_list.append(counter)
                    medium_list.append(counter)

                    counter += slow_window_size
                    slow_window_size *= 2
                else:
                    counter = tune_l  - round(min_medium_updates * window_size * 3 / 4)
                    slow_list.append(counter)
            elif counter < tune_l :
                temp = counter + window_size
                if temp < tune_l :
                    medium_list.append(counter)

                    counter += window_size
                else:
                    counter = tune_l
                    medium_list.append(counter)
            else:
                overshoots = True
    return(fast_list,medium_list,slow_list)

#out = case4(tune_l=2500,window_size=25)
#print(out[0])
#print(out[1])
#print(out[2])

# f
# case5
def f(tune_l=250):
    #print("yes")
    case = 5
    fast_list = []
    medium_list = []
    slow_list = []
    counter = 0
    overshoots = False
    while overshoots == False:
        if counter < tune_l:
            fast_list.append(counter)
            counter += 1
        else:
            overshoots = True


    return(fast_list,medium_list,slow_list)

#out = f(tune_l=2500)
#print(out[0])
#print(out[1])
#print(out[2])


#case 6
# m
def m(window_size=25,min_medium_updates=5,tune_l=250):
    case = 6
    fast_list = []
    medium_list = []
    slow_list = []
# if case==1:
    slow_window_size = window_size
    min_length = window_size * (round(min_medium_updates *(1.75)))
    #print("tune_l {}".format(tune_l))

    if tune_l < min_length:
        raise ValueError("warm up not long enough")
    else:
        counter = 0
        overshoots = False
        while overshoots == False:
            if counter <  tune_l:
                temp = counter +  window_size
                if  temp <  tune_l:
                    medium_list.append(counter)
                    counter += window_size
                else:
                    counter =  tune_l
                    medium_list.append(counter)
            else:
                overshoots = True
    return(fast_list,medium_list,slow_list)

#out = case6(tune_l=570,window_size=25)
#print(out[0])
#print(out[1])
#print(out[2])

def s(window_size=25,min_slow_updates=3,tune_l=250):
    case = 7
    fast_list = []
    medium_list = []
    slow_list = []
# if case==1:
    slow_window_size = window_size

    min_length = (2 ** (min_slow_updates + 1) - 1) * window_size

    if tune_l < min_length:
        raise ValueError("warm up not long enough")
    else:
        counter = 0
        overshoots = False
        while overshoots == False:
            if counter <  tune_l:
                temp = counter + slow_window_size
                if temp < tune_l:
                    slow_list.append(counter)
                    counter += slow_window_size
                    slow_window_size *= 2
                else:
                    counter = tune_l
                    slow_list.append(counter)
            else:
                overshoots = True
    return(fast_list,medium_list,slow_list)

#out = case7(tune_l=1570,window_size=25)
#print(out[0])
#print(out[1])
#print(out[2])