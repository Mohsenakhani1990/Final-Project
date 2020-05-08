#!python3.7
"""
             Calculation of Risk of a Structure based on BJF97 GMPE model
@author: mkhnsnjn
"""
#======================================
#            modules
#======================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from BJF1997 import main
#======================================
#        functions
#======================================
# function that calculate Spectral Acceleration of any Earthquake
def eval_Sa (ACC,m,Delta_t,zeta,Tn):
    P_eff0=-m*ACC # force
    u=np.zeros([len(ACC),200])
    udot=np.zeros([len(ACC),200])
    u2dot=np.zeros([len(ACC),200])
    D1=np.zeros(200)
    #for i in range(5):
    Wn=2*np.pi/Tn #omega in frequency
    k=(Wn**2)*m
    WD=Wn*np.sqrt(1-(zeta**2))# 
    l=zeta/(np.sqrt(1-(zeta**2)))
    q=np.exp(-zeta*Wn*Delta_t)
    z=2*zeta/(Wn*Delta_t)
    r=(1-2*(zeta**2))/(WD*Delta_t)
    s=Wn/(np.sqrt(1-(zeta**2)))
    x=1/Delta_t
    A=q*(l*np.sin(WD*Delta_t)+np.cos(WD*Delta_t))
    B=q*(np.sin(WD*Delta_t)/WD)
    C=(z+q*( (r-l)*np.sin(WD*Delta_t)- ((1+z)*np.cos(WD*Delta_t))))/k
    D=(1-z+ q*(-r*np.sin(WD*Delta_t)+z*np.cos(WD*Delta_t)))/k
    A_prin=-q*(s*np.sin(WD*Delta_t))
    B_prin=q*( np.cos(WD*Delta_t)-(l*np.sin(WD*Delta_t)))
    C_prin=(-x+q*((np.sin(WD*Delta_t)*(s+(l/Delta_t)))+(x*np.cos(WD*Delta_t))))/k
    D_prin=(1-q*(np.cos(WD*Delta_t)+(l*np.sin(WD*Delta_t))))/(k*Delta_t)
    for j in range(len(Tn)):
        for i in range(len(ACC)-1):
            u[i+1,j]=A[j]*u[i,j]+B[j]*udot[i,j]+C[j]*P_eff0[i]+D[j]*P_eff0[i+1]
            udot[i+1,j]=A_prin[j]*u[i,j]+B_prin[j]*udot[i,j]+C_prin[j]*P_eff0[i]+D_prin[j]*P_eff0[i+1]
            u2dot[i+1,j]=(udot[i+1,j]-udot[i,j])/Delta_t
    u0=u
    V=np.zeros(200)
    Sa=np.zeros(200)
    for i in range(200):
        D1[i]=np.max(np.abs(u0[:,i]))
        V[i]=Wn[i]*D1[i]
        Sa[i]=(Wn[i]**2)*D1[i]
    return Sa
#======================================
#         Loading data
#======================================
loq1 = 'coefficients.txt'
coeff = np.loadtxt( loq1)# Coefficientf OF BJF97 to get the Sa from the function BJF97
loq2='inputs_GMPE.txt'
inp=np.loadtxt( loq2) #inputs of GMPE BJF97
loq3 = 'acceleration.txt'
ACC = np.loadtxt( loq3)# ACCELERATION OF EARTHQUAKES
loq7 = 'Response.txt'
Res = np.loadtxt( loq7)
period=coeff[:,0]
M=inp[:,0]
rjb=inp[:,1]
Fault_Type=inp[:,3]
Vs30=inp[:,4]
lo = 'H.txt' # AS97 Sa
H = np.loadtxt( lo) #AS97 MAF
lo2 = 'H1.txt'
H1 = np.loadtxt( lo2) #AS97 MAF Adjustment
lo3 = 'H2.txt'
H2 = np.loadtxt( lo3)
#======================================
#        Parameters
#======================================
zeta=0.05 # damping ratio
m=0.253 #mass of structure
Delta_t=[0.01,0.01,0.02,0.01,0.02] #time step for earthquakes
Tn=np.linspace(0.01,2,200) #Period 
min_eps0, max_eps0=-4.5,4.5
eps0 = np.arange (min_eps0, max_eps0, 0.25) # epsilon of hazard maps
stN=2 # number of stories
RDR=0.085 #roof drift ratio
T0=0.63 #First Period of structure
#======================================
#        Function executions
#======================================
#Epectral acceleration for 5 earthquakes.
Sa_0=eval_Sa ( ACC[:,0],m,Delta_t[0],zeta,Tn)
Sa_1=eval_Sa ( ACC[:,1],m,Delta_t[1],zeta,Tn)
Sa_2=eval_Sa ( ACC[:,2],m,Delta_t[2],zeta,Tn)
Sa_3=eval_Sa ( ACC[:,3],m,Delta_t[3],zeta,Tn)
Sa_4=eval_Sa ( ACC[:,4],m,Delta_t[4],zeta,Tn)

lnY=np.zeros(len(period))
sigma=np.zeros(len(period))
for i in range(len(period)-1):
    lnY[i],sigma[i]=main(M[i],rjb[i],period[i+1],Fault_Type[i],Vs30[i])
lnS0=Sa_0[np.where(Tn==T0)]
lnS0=lnS0[0]
lnS1=Sa_1[np.where(Tn==T0)]
lnS1=lnS1[0]
lnS2=Sa_2[np.where(Tn==T0)]
lnS2=lnS2[0]
lnS3=Sa_3[np.where(Tn==T0)]
lnS3=lnS3[0]
lnS4=Sa_4[np.where(Tn==T0)]
lnS4=lnS4[0]
lnS=[lnS0,lnS1,lnS2,lnS3,lnS4]
Eps=np.zeros(5)
lny=np.zeros(5)
sig=np.zeros(5)
# calculation of Epsilon by estimating differnce between Sa earthquake and Sa GMPE
for i in range(5):
   lny[i],sig[i] =main(M[i],rjb[i],T0,Fault_Type[i],Vs30[i])
   Eps[i]=(lnS[i]-lny[i])/sig[i]
#--------------------------------
   #calculation of mean of Sa collapse from the earthquakes
mbar=np.zeros(len(eps0))
epsbar=np.mean(Eps)
mt=np.mean(np.log(Res))
st=np.std(np.log(Res))
b2=0.4*(2+stN)**0.35*RDR**0.38
for i in range(len(eps0)):
     mbar[i]=mt+b2*(eps0[i]-epsbar)   
## characteresitic event
    
RP=200
M=7.2
Vs30=360
Fault_Type=1
Rjb=10
T0=0.63
lnY0, siglnY0=main(M,Rjb,T0,Fault_Type,Vs30)
ln_Y=np.zeros(len(eps0))
#hazard curve
for i in range(len(eps0)):
        ln_Y[i]=np.log(lnY0)+eps0[i]*siglnY0
        
L=(1-norm.cdf(eps0))/200
# MAF calculation Hazard curve*fragility curve
maftot=norm.cdf(ln_Y,mt,st)*L
mafadj=norm.cdf(ln_Y,mbar,st)*L
cdfmaf_tot=norm.cdf(ln_Y,mt,st)
cdfmaf_adj=norm.cdf(ln_Y,mbar*0.85,st)
# calculation of area under the MAF curves
x=np.exp(ln_Y)
x1=np.append(0,x)
x2=np.append(x,0)
dx=x2-x1
dx=np.delete(dx,-1,0)
M_tot=maftot*dx
M_adj=mafadj*dx
MAF_tot=np.sum(M_tot)
MAF_adj=np.sum(M_adj)
###======================================
###            Plot
####=====================================
plt.figure(1)
plt.plot( Tn, Sa_0, 'r', label = 'Earthquake1')
plt.plot( Tn, Sa_1, 'g', label = 'Earthquake2')
plt.plot( Tn, Sa_2, 'b', label = 'Earthquake3')
plt.plot( Tn, Sa_3, 'm', label = 'Earthquake4')
plt.plot( Tn, Sa_4, 'c', label = 'Earthquake5')
plt.plot( period, lnY, 'y', linewidth=4,label = 'GMPE')
plt.legend()
plt.xlabel( 'Period')
plt.ylabel( 'Spectral Acceleration')
plt.show() 

plt.figure(2)
plt.semilogx(np.exp(ln_Y),L,'r', label = 'hazard curve')
plt.legend()
plt.xlabel( 'Spectral Acceleration(IM) [g]')
plt.ylabel( 'Frequency of Occurence')
plt.show() 
plt.figure(3)
plt.semilogx(np.exp(ln_Y),cdfmaf_tot,'r', label = 'fragility curve (without filteration on epsilon)')
plt.semilogx(np.exp(ln_Y),cdfmaf_adj,'g',label = 'fragility curve Adjustment method')

plt.legend()
plt.xlabel( 'Intensity measure(IM) [g]')
plt.ylabel( 'Probabiliy of Collapse')
plt.show()  

plt.figure(4)
plt.semilogx(np.exp(ln_Y),maftot,'r', label = 'MAF Total BJF97 without Filteration ')
plt.semilogx(np.exp(ln_Y),mafadj,'g',label = 'MAF Adjustment Method')
plt.semilogx(np.exp(H),H1,'m', label = 'MAF ADJ_AS97')
plt.semilogx(np.exp(H),H2,'c', label = 'MAF total AS97 without Filteration')
plt.legend()
plt.xlabel( 'Intensity measure(IM) [g]')
plt.ylabel( 'Hazard curve*fragility curve')
plt.show()

print( "MAF Total BJF97model is:",( MAF_tot))
print( 'MAF Adjustment BJF97 Model is',( MAF_adj))
