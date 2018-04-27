# convert to stable implementation
import math
def finite_1stderiv(f,h):

    out = f(+h)-f(-h)
    out = out/(2*h)
    #signout = 2*float(out>0)-1.
    #out = math.exp(math.log(abs(out)) - math.log(2*h))*signout
    return(out)


def finite_2ndderiv(f_2var,h):

    out = (f_2var(+h,+h) + f_2var(-h,-h)) - (f_2var(+h,-h) + f_2var(-h,+h))
    out = out/(4*h*h)

    return(out)

def finite_3rdderiv(f_3var,h):
    out = (f_3var(+h,+h,+h) + f_3var(+h,-h,-h) + f_3var(-h,+h,-h) +f_3var(-h,-h,+h))\
          -( f_3var(+h,-h,+h) + f_3var(+h,+h,-h) + f_3var(-h,+h,+h) + f_3var(-h,-h,-h))

    out = out/(8*h*h*h)
    return(out)


