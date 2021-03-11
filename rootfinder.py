

def bisect(f, a, b, eps = 1.e-6):
    fa, fb, gap  = f(a), f(b), abs(b-a)

    if (fa*fb > 0.0):
        print('Bisection error: no root bracketed')
        return None
    elif fa == 0.0: return a
    elif fb == 0.0: return b

    while(True):
        xmid = 0.5*(a+b)
        fmid = f(xmid)

        if (fa*fmid > 0.0):
            a, fa = xmid, fmid
        else :b = xmid

        if (fmid == 0.0 or abs (b-a) < eps*gap): break

    return xmid

def newton( f, df, eps = 1.e-6):
    nmax, fx = 20, f(x)

    if (fx == 0.0): return x

    for i in range(nmax):
        delta = fx/df(x)

        if (i == 0): gap = abs(delta)
        x = x - delta
        fx = f(x)
        if (fx == 0.0 or abs(delta) < eps*gap): break

    return x
