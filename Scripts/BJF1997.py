import numpy as np
#======================================
#         Loading data
#======================================
loq1 = 'coefficients.txt' # coefficients for BJF97 GMPE
coeff = np.loadtxt( loq1)
period=coeff[:,0]
h     =coeff[:,9]
B1ss  =coeff[:,1]
B1rv  =coeff[:,2]
B1all =coeff[:,3]
B2    =coeff[:,4]
B3    =coeff[:,5]
B5    =coeff[:,6]
Bv    =coeff[:,7]
Va    =coeff[:,8]
sigma1=coeff[:,10]
sigmae=coeff[:,11]
#======================================
#        functions
#======================================
T=0.001
# main function to get Sa from GMPE BJF97
def main(M, rjb, T, Fault_Type, Vs):
    cc=len(np.where(period == T))
    cc=cc-1
    if cc == 0:
        return BJF_1997_horiz(M, rjb, T, Fault_Type, Vs)
    else:
        i=period[np.where(period==T)]
        i=i[0]
        r = np.sqrt(rjb**2 + h[i]**2)
        if(Fault_Type == 1):
            b1 = B1ss[i]
        elif (Fault_Type == 2):
            b1 = B1rv[i]
        else:
            b1 = B1all[i]
    lny= b1 + B2[i]*(M-6) + B3[i]*(M-6)**2 + B5[i]*np.log(r) + Bv[i]*np.log(Vs / Va[i])   
    sa = np.exp(lny)
    sigma = np.sqrt(sigma1[i]**2 + sigmae[i]**2)
    return sa,sigma
#sa=print( ' Spectral acceleration=%.3f'%(sa))
def BJF_1997_horiz(M, rjb, T, Fault_Type, Vs):
    cc=len(np.where(period == T))
    cc=cc-1
    if cc == 0:
        index_low = np.where(period < T)
        result=[]
        for t in index_low:
            for x in t:
               result.append(x)       
        ind=result[-1] 
        T_low = period[ind]
        T_hi = period[ind+1]
    sa_low, sigma_low = BJF_1997_T_low(M, rjb, T_low, Fault_Type, Vs)
    sa_hi, sigma_hi = BJF_1997_T_hi(M, rjb, T_hi, Fault_Type, Vs)
#    interpolate between periods if neccesary
    m_sa = abs((sa_hi-sa_low)/(T_hi-T_low))
    sa=m_sa*(T-T_low)+sa_low
    m_sig=abs((sigma_hi-sigma_low)/(T_hi-T_low))
    sigma=m_sig*(T-T_low)+sigma_low
    return sa,sigma  
# function to find the T_low and T_high for interpolation 
def T_low_hi(T):
    cc=len(np.where(period == T))
    cc=cc-1
    if cc == 0:
        index_low = np.where(period < T)
        result=[]
        for t in index_low:
            for x in t:
               result.append(x)       
        ind=result[-1] 
        T_low = period[ind]
        T_hi = period[ind+1]
    return T_low, T_hi
def BJF_1997_T_low(M, rjb, T_low, Fault_Type, Vs):
    i=np.where(period==T_low)
#    i=int(ii)
    r = np.sqrt(rjb**2 + h[i]**2)
    if(Fault_Type == 1):
        b1 = B1ss[i]
    elif (Fault_Type == 2):
         b1 = B1rv[i]
    else:
        b1 = B1all[i]
    lny_low= b1 + B2[i]*(M-6) + B3[i]*(M-6)**2 + B5[i]*np.log(r) + Bv[i]*np.log(Vs / Va[i])   
    sa_low = np.exp(lny_low)
    sigma_low = np.sqrt(sigma1[i]**2 + sigmae[i]**2)
    return sa_low,sigma_low
def BJF_1997_T_hi(M, rjb, T_hi, Fault_Type, Vs):
    i=np.where(period==T_hi)
#    i=i[0]
    r = np.sqrt(rjb**2 + h[i]**2)
    if(Fault_Type == 1):
        b1 = B1ss[i]
    elif (Fault_Type == 2):
         b1 = B1rv[i]
    else:
        b1 = B1all[i]
    lny_hi= b1 + B2[i]*(M-6) + B3[i]*(M-6)**2 + B5[i]*np.log(r) + Bv[i]*np.log(Vs / Va[i])   
    sa_hi = np.exp(lny_hi)
    sigma_hi = np.sqrt(sigma1[i]**2 + sigmae[i]**2)
    return sa_hi,sigma_hi