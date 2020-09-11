#Standard python imports
# cython: profile=True
# cython: infer_types=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from __future__ import division
import  numpy as np
cimport numpy as np
cimport cython
from scipy.special.cython_special cimport i0e, i1e, k0e, k1e
from libc.math cimport M_PI, sin, cos, exp, log, sqrt, acos, atan2,log10,isfinite

'''this is in Km^2 Kpc / (M_sun s^2).'''
cdef double G = 4.299e-6


#---------------------------------------------------------------------------
#N.B. ---> RICORDARSI QUANDO SI FANNO LE DIVISIONI 5/2 = 2 MENTRE 5.0/2.0 = 2.5
#---------------------------------------------------------------------------
#rebuild all in order to do just one cycle in the fun model or in likelihood
#---------------------------------------------------------------------------
                    #HERQUINST
#---------------------------------------------------------------------------
#Faccio una struc per herquinst così calcolo rho e sigma insieme senza fare
#valutare due volte X_function per esempio.
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double median (double[::1] x,     #variabile su cui faccio la mediana
                     double flag):      #flag = 0. in x non ci sono termini neg/ flag != 0 considero anche termini negativi
    #mediana fatta considerando che termini nulli non vengono considerati
    cdef Py_ssize_t i,j
    cdef Py_ssize_t N = x.shape[0]
    cdef double tmp1 = 0.
    cdef double tmp2 = 0.
    cdef int non_zero_counter = 0
    cdef int zero_counter = 0
    cdef int count = 0
    cdef np.ndarray[double,ndim=1,mode="c"] zero_first = np.zeros((N))
    cdef double[::1] zero_first_view = zero_first
    
    if flag == 0.:  #per sigma e rho non ho termini negativi
        for i in range(N):
            tmp1 = x[i]
            if tmp1 !=0.0:
                non_zero_counter += 1
            else:
                zero_counter +=1

        if non_zero_counter == 0:
            return 0.0

        for i in range(N-1):
            for j in range(i+1,N):
                if x[i] > x[j]:
                    tmp2 = x[j]
                    x[j] = x[i]
                    x[i] = tmp2 
                else: 
                    pass

        if non_zero_counter%2 == 0.:
            j = zero_counter + int((non_zero_counter-1)/2.0)
            return 0.5*(x[j] +x[j+1]) 
        else:
            j = zero_counter + int(non_zero_counter/2.0)
            return x[j]
    
    else:
        for i in range(N):
            tmp1 = x[i]
            if tmp1 !=0.0:
                non_zero_counter += 1
            else:
                zero_counter +=1

        if non_zero_counter == 0:
            return 0.0

        count = 0
        for i in range(N):
            tmp1 = x[i]
            if tmp1 != 0:
                zero_first_view[zero_counter+count] = tmp1
                count +=1
            else:
                pass 

        for i in range(zero_counter,N-1):
            for j in range(i+1,N):
                if zero_first_view[i] > zero_first_view[j]:
                    tmp2 = zero_first_view[j]
                    zero_first_view[j] = zero_first_view[i]
                    zero_first_view[i] = tmp2 
                else: 
                    pass

        if non_zero_counter%2 == 0.:
            j = zero_counter + int((non_zero_counter-1)/2.0)
            return 0.5*(zero_first_view[j] +zero_first_view[j+1]) 
        else:
            j = zero_counter + int(non_zero_counter/2.0)
            return zero_first_view[j]
    

cdef struct HERQUINST:
    double rho 
    double sigma 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double radius(double x, double y) nogil :
    return sqrt(x*x+y*y)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double Xfunction(double s) nogil :
    # see Hernquist 1990
    cdef double one_minus_s2 = 1.0-s*s
    if s < 1.0:
        return log((1.0 + sqrt(one_minus_s2)) / s) / sqrt(one_minus_s2)
    elif s ==  1.0:
        return 1.0
    else:
        return acos(1.0/s)/sqrt(-one_minus_s2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef HERQUINST herquinst_function(double M, double Rb, double r) nogil:
    #return HERQUINST struct
    cdef HERQUINST herquinst
    herquinst.rho = 0.
    herquinst.sigma = 0.
    cdef double Rb2 = Rb*Rb 
    cdef double s = r/Rb
    cdef double s2 = 0.
    cdef double s3 = 0.
    cdef double s4 = 0.
    cdef double s6 = 0.
    cdef double one_minus_s2 = 0.
    cdef double one_minus_s2_2 = 0.
    cdef double A,B,C,D
    cdef double sigma = 0.
    cdef double X = 0.0
    cdef double rho = 0.
    cdef double M_PI2 = M_PI*M_PI
    cdef double M_PI3 = M_PI2*M_PI

    if (M == 0.) or (Rb == 0.): 
        return herquinst 
    
    if s == 0:
        s = 1e-8

    if s >= 0.98 and s <=1.02:
        rho = 4.0*M / (30.0*M_PI*Rb2)
        sigma = G*M*(332.0 - 105.0*M_PI)/28.0*Rb
    else:
        X = Xfunction(s)
        s2 = s*s
        one_minus_s2 = (1.0 - s2)
        one_minus_s2_2 = one_minus_s2*one_minus_s2
        rho = (M / (2.0 * M_PI * Rb2 * one_minus_s2_2)) * ((2.0 + s2) * X - 3.0) 
        if s >= 10.0:
            s3 = s*s2
            A = (G * M * 8.0) / (15.0 * s * M_PI * Rb)
            B = (8.0/M_PI - 75.0*M_PI/64.0) / s
            C = (64.0/(M_PI2) - 297.0/56.0) / s2
            D = (512.0/(M_PI3) - 1199.0/(21.0*M_PI) - 75.0*M_PI/512.0) / s3
            sigma = A * (1 + B + C + D)   
        else:
            s4 = s2*s2 
            s6 = s4*s2
            A = (G * M*M ) / (12.0 * M_PI * Rb2 * Rb * rho)
            B = 1.0 /  (2.0*one_minus_s2_2*one_minus_s2)
            C = (-3.0 * s2) * X * (8.0*s6 - 28.0*s4 + 35.0*s2 - 20.0) - 24.0*s6 + 68.0*s4 - 65.0*s2 + 6.0
            sigma = A * (B*C - 6.0*M_PI*s)
    
    herquinst.rho = rho
    herquinst.sigma = sigma
    
    return herquinst


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double v_H(double Mb,double Rb,double r) nogil:
    #circular v^2 for herquinst (actually bulge and halo)
    cdef double v_c2 = 0.
    if Mb == 0. or Rb == 0.:
        return v_c2
    cdef double r_tmp = r/Rb
    cdef double one_plus_r = 1.0 + r_tmp
    v_c2 = G * Mb * r_tmp / (Rb*one_plus_r*one_plus_r )
    
    return v_c2

#---------------------------------------------------------------------------
                    #EXP DISC
#---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double v_D(double Md,double Rd,double r) nogil:
    #circular v^2 for exp disc
    cdef double v_c2 = 0.
    if Md == 0 :
        return v_c2
    cdef double r_tmp = 0.
    r_tmp = r/(2.0*Rd)
    v_c2 =  ((G * Md * r) / (Rd*Rd)) * r_tmp * (i0e(r_tmp)*k0e(r_tmp) - i1e(r_tmp)*k1e(r_tmp))
    
    return v_c2 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double rho_D(double Md,double Rd,double r,double c_incl) nogil:
    #surface density for a thin disc with no
    #thickness (no z coord). So I don't have any 
    #integration on the LOS,but only rotation,traslation 
    #and deprojection of the coordinates.
    #c_incl is already cos(incl)
    cdef double rho = 0.
    if (Md == 0.) or (Rd == 0.):
        return rho
    
    rho = (Md / (2.0 * M_PI * Rd*Rd) ) * exp(-r/Rd)
    return(rho/c_incl)


#---------------------------------------------------------------------------
                    #TOTAL QUANTITIES
#---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double v_tot( double Mb,
                   double Rb,
                   double Md,
                   double Rd,
                   double Mh,
                   double Rh,
                   double s_incl,    #sin(incl)
                   double r,         #proj radius
                   double phi) nogil:#phi computed in model function
                                              
    cdef double r_tmp = 0.
    cdef double v_tot = 0.
    if (Md==0) or (Rd==0):
        return v_tot       
    v_tot = s_incl * cos(phi) * sqrt(v_D(Md,Rd,r) + v_H(Mb,Rb,r) + v_H(Mh,Rh,r)) 
    return v_tot 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double rho_tot(double rho_B,double rho_D) nogil:
    
    cdef double rho_tot = rho_B + rho_D
    return rho_tot


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double sigma_tot(double rhoD,                  #disc density
                      double v_i,                   #v of the 'particle'
                      double v_bin,                 #mean v of the bin
                      double rho_dot_sigma2) nogil: #rhoH * sigmaH**2
    
    cdef double sigma_tot = 0.
    cdef double v_diff = v_i - v_bin
    sigma_tot = rhoD*v_diff*v_diff + rho_dot_sigma2
    return sigma_tot

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
                    #MODEL
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

##                    ##
##     NORMAL MODEL   ##
##                    ##
#modello normale senza psf
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=3,mode='c'] model (double[:,:,::1] x,#x position refined grid è 99x99x100
                                                double[:,:,::1] y,#y position refined grid
                                                double Mb,        #bulge mass log
                                                double Rb,        #bulge radius
                                                double Md,        #disc mass log
                                                double Rd,        #disc radius
                                                double Mh,        #halo mass log
                                                double Rh,        #halo radius
                                                double xcm,       #center x position
                                                double ycm,       #center y position
                                                double theta,     #P.A. angle (rotation in sky-plane)
                                                double incl):     #inclination angle
    
    cdef double x0 = 0.               #x-xcm  traslation
    cdef double y0 = 0.               #y-ycm  traslation
    cdef double xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef double yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef double yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef double r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef double r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef double c_theta = cos(theta)  #cos(theta) calculated once
    cdef double s_theta = sin(theta)  #sin(theta) calculated once
    cdef double c_incl = cos(incl)    #cos(incl) calculated once
    cdef double s_incl = sin(incl)    #cos(incl) calculated once
    cdef double phi = 0.
    cdef Py_ssize_t i,j,k
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = x.shape[1]
    cdef Py_ssize_t K = 100
    
    #define HERQUINST STRUC 
    cdef HERQUINST herquinst  

    #   RHO DEFINITIONS
    cdef double rhoB = 0.
    cdef double rhoD = 0.
    cdef double sum_rhoD = 0.         #mi serve per mediare la vel
    #per il momento la creo 100X100X100 perchè così sono i dati
    #si può migliorare
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    cdef np.ndarray[double,ndim=3,mode="c"] all_rhoD = np.zeros((N,J,K))
    cdef double[:,:,::1] all_rhoD_view = all_rhoD
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    cdef np.ndarray[double,ndim=3,mode = "c"] rho_dot_sigma2 = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_dot_sigma2_view = rho_dot_sigma2
    
    #V DEFINITIONS
    cdef double v_tmp = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
    cdef np.ndarray[double,ndim=3,mode = "c"] all_v = np.zeros((N,J,K))
    cdef double[:,:,::1] all_v_view = all_v

    #questa è una cosa inutile quando tutto funziona va modificato
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N+1,J+1,3))
    cdef double[:,:,::1] tot_view = tot 
    
    #masse passate in logaritmo
    Mb = 10**Mb #is it ok or better import pow from math?
    Md = 10**Md
    Mh = 10**Mh
    Rb = 10**Rb 
    Rd = 10**Rd
    Rh = 10**Rh 

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    
    for i in range(N):
        for j in range(J):
            sum_rhoD = 0.
            for k in range(K):
                #queste operazioni è meglio farle in una funzione?
                #per il fatto che sarebbe cdef e nogil...
                x0 = x[i,j,k]-xcm
                y0 = y[i,j,k]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)

                #rhoB = hernquist_rho(Mb,Rb,r_proj)
                herquinst = herquinst_function(Mb,Rb,r_proj)
                rhoB = herquinst.rho 
                rhoD = rho_D(Md,Rd,r_true,c_incl)
                all_rhoD_view[i,j,k] = rhoD
                sum_rhoD = sum_rhoD + rhoD
                tot_view[i,j,0] += rho_tot(rhoB,rhoD)

                v_tmp = v_tot(Mb,Rb,Md,Rd,Mh,Rh,s_incl,r_true,phi)
                all_v_view[i,j,k] = v_tmp
                tot_view[i,j,1] += rhoD*v_tmp

                #questo è rho_H*sigma_H2 andrà sommato nel prossimo ciclo 
                #con la dispersione del disco che devo ancora calcolare
                #rho_dot_sigma2_view[i,j,k] = rhoB*hernquist_sigma(Mb,Rb,rhoB,r_proj)
                rho_dot_sigma2_view[i,j,k] = rhoB*herquinst.sigma
            
            #le sommo finito il ciclo divido per 100, per fare la media
            tot_view[i,j,0] = tot_view[i,j,0]/100.0
            if sum_rhoD == 0. :
                tot_view[i,j,1] = 0.
            else:
                tot_view[i,j,1] = tot_view[i,j,1]/sum_rhoD 
    
    v_tmp = 0.
    for i in range(N):
        for j in range(J):
            #mean v in bin
            v_tmp = tot_view[i,j,1]
            for k in range(K):
                tot_view[i,j,2] +=  sigma_tot(all_rhoD_view[i,j,k],all_v_view[i,j,k],v_tmp,rho_dot_sigma2_view[i,j,k])

            #come denominatore avrei bisogno della somma su k delle densità
            #ma dato che so la densità media, e so che ho 100 punti, so anche la somma
            tot_view[i,j,2] = sqrt(tot_view[i,j,2]/(tot_view[i,j,0]*100.0))
    
    
    #si può ragionare su come ritornare direttamente un vettore
    #in questo modo dall'altra parte evito di fare .ravel
    return tot 


##                    ##
##     MODEL PSF NAN  ##
##                    ##
#questo include la psf gli passo anche i dati perchè calcola il modello solo
#dove i dati non sono nan, lo utilizzo per calcolare la likelihoo più velocemente
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=3,mode='c'] model_psf (double[:,:,::1] x,          #x position refined grid è 99x99x100
                                                    double[:,:,::1] y,          #y position refined grid
                                                    double Mb,                  #bulge mass log
                                                    double Rb,                  #bulge radius
                                                    double Md,                  #disc mass log
                                                    double Rd,                  #disc radius
                                                    double Mh,                  #halo mass log
                                                    double Rh,                  #halo radius
                                                    double xcm,                 #center x position
                                                    double ycm,                 #center y position
                                                    double theta,               #P.A. angle (rotation in sky-plane)
                                                    double incl,                #inclination angle
                                                    double[:,:,::1]ydata,
                                                    double[:,:,::1]psf_weight):   #psf weight
    
    cdef double x0 = 0.               #x-xcm  traslation
    cdef double y0 = 0.               #y-ycm  traslation
    cdef double xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef double yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef double yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef double r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef double r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef double c_theta = cos(theta)  #cos(theta) calculated once
    cdef double s_theta = sin(theta)  #sin(theta) calculated once
    cdef double c_incl = cos(incl)    #cos(incl) calculated once
    cdef double s_incl = sin(incl)    #cos(incl) calculated once
    cdef double phi = 0.
    cdef Py_ssize_t i,j,k
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = x.shape[1]
    cdef Py_ssize_t K = 100
    
    #define HERQUINST STRUC 
    cdef HERQUINST herquinst  

    #   RHO DEFINITIONS
    cdef double rhoB = 0.
    cdef double rhoD = 0.
    cdef double sum_rhoD = 0.         #mi serve per mediare la vel
    #per il momento la creo 100X100X100 perchè così sono i dati
    #si può migliorare
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    cdef np.ndarray[double,ndim=3,mode="c"] all_rhoD = np.zeros((N,J,K))
    cdef double[:,:,::1] all_rhoD_view = all_rhoD
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    cdef np.ndarray[double,ndim=3,mode = "c"] rho_dot_sigma2 = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_dot_sigma2_view = rho_dot_sigma2
    
    #V DEFINITIONS
    cdef double v_tmp = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
    cdef np.ndarray[double,ndim=3,mode = "c"] all_v = np.zeros((N,J,K))
    cdef double[:,:,::1] all_v_view = all_v

    #questa è una cosa inutile quando tutto funziona va modificato
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N+1,J+1,3))
    cdef double[:,:,::1] tot_view = tot 
    
    #PSF definitions
    cdef double psf_tmp=0.
    cdef double sum_psf = 0.
    cdef np.ndarray[double,ndim=2,mode='c'] sum_psf_saved = np.zeros((N,J))
    cdef double[:,::1] sum_psf_saved_view = sum_psf_saved
    #masse passate in logaritmo
    Mb = 10**Mb #is it ok or better import pow from math?
    Md = 10**Md
    Mh = 10**Mh
    Rb = 10**Rb 
    Rd = 10**Rd
    Rh = 10**Rh 

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    
    for i in range(N):
        for j in range(J):
            sum_rhoD = 0.
            sum_psf = 0.
            if 10**ydata[i,j,0] > 0:
                for k in range(K):
                    psf_tmp = psf_weight[i,j,k]
                    #queste operazioni è meglio farle in una funzione?
                    #per il fatto che sarebbe cdef e nogil...
                    x0 = x[i,j,k]-xcm
                    y0 = y[i,j,k]-ycm
                    xr = x0*c_theta + y0*s_theta 
                    yr = y0*c_theta - x0*s_theta
                    yd = yr/c_incl
                    r_proj = radius(x0,y0) 
                    r_true = radius(xr,yd)
                    phi = atan2(yd,xr)

                    #rhoB = hernquist_rho(Mb,Rb,r_proj)
                    herquinst = herquinst_function(Mb,Rb,r_proj)
                    rhoB = herquinst.rho 
                    rhoD = rho_D(Md,Rd,r_true,c_incl)
                    all_rhoD_view[i,j,k] = rhoD
                    #sum_rhoD = sum_rhoD + rhoD
                    sum_rhoD = sum_rhoD + rhoD*psf_tmp
                    tot_view[i,j,0] += psf_tmp*rho_tot(rhoB,rhoD)

                    v_tmp = v_tot(Mb,Rb,Md,Rd,Mh,Rh,s_incl,r_true,phi)
                    all_v_view[i,j,k] = v_tmp
                    tot_view[i,j,1] += psf_tmp*rhoD*v_tmp
                    
                    sum_psf += psf_tmp
                    #questo è rho_H*sigma_H2 andrà sommato nel prossimo ciclo 
                    #con la dispersione del disco che devo ancora calcolare
                    #rho_dot_sigma2_view[i,j,k] = rhoB*hernquist_sigma(Mb,Rb,rhoB,r_proj)
                    rho_dot_sigma2_view[i,j,k] = rhoB*herquinst.sigma

                #le sommo finito il ciclo divido per 100, per fare la media
                #tot_view[i,j,0] = tot_view[i,j,0]/100.0
                tot_view[i,j,0] = tot_view[i,j,0]/sum_psf
                sum_psf_saved_view[i,j] = sum_psf
                if sum_rhoD == 0. :
                    tot_view[i,j,1] = 0.
                else:
                    #ma sum_rhoD contiene anche la PSF
                    tot_view[i,j,1] = tot_view[i,j,1]/sum_rhoD
            else:
                pass 
    
    v_tmp = 0.
    psf_tmp = 0.
    sum_psf = 0.
    for i in range(N):
        for j in range(J):
            #mean v in bin
            if 10**ydata[i,j,0]>0:
                v_tmp = tot_view[i,j,1]
                sum_psf = sum_psf_saved_view[i,j]
                for k in range(K):
                    tot_view[i,j,2] +=  psf_weight[i,j,k]*sigma_tot(all_rhoD_view[i,j,k],all_v_view[i,j,k],v_tmp,rho_dot_sigma2_view[i,j,k])

                #come denominatore avrei bisogno della somma su k delle densità
                #ma dato che so la densità media, e so che ho 100 punti, so anche la somma
                tot_view[i,j,2] = sqrt(tot_view[i,j,2]/(tot_view[i,j,0]*sum_psf))
            else:
                pass
    
    #si può ragionare su come ritornare direttamente un vettore
    #in questo modo dall'altra parte evito di fare .ravel
    return tot 


##                                                              ##
##     MODEL PSF WITH V AS RHO_DISC*V_LOS/(RHO_DISC+RHO_BULGE)  ##
##                                                              ##
#questo include la psf gli passo anche i dati perchè calcola il modello solo
#dove i dati non sono nan, lo utilizzo per calcolare la likelihoo più velocemente
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=3,mode='c'] model_psf_modified_vel (double[:,:,::1] x,          #x position refined grid è 99x99x100
                                                    double[:,:,::1] y,          #y position refined grid
                                                    double Mb,                  #bulge mass log
                                                    double Rb,                  #bulge radius
                                                    double Md,                  #disc mass log
                                                    double Rd,                  #disc radius
                                                    double Mh,                  #halo mass log
                                                    double Rh,                  #halo radius
                                                    double xcm,                 #center x position
                                                    double ycm,                 #center y position
                                                    double theta,               #P.A. angle (rotation in sky-plane)
                                                    double incl,                #inclination angle
                                                    double[:,:,::1]ydata,
                                                    double[:,:,::1]psf_weight):   #psf weight
    
    cdef double x0 = 0.               #x-xcm  traslation
    cdef double y0 = 0.               #y-ycm  traslation
    cdef double xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef double yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef double yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef double r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef double r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef double c_theta = cos(theta)  #cos(theta) calculated once
    cdef double s_theta = sin(theta)  #sin(theta) calculated once
    cdef double c_incl = cos(incl)    #cos(incl) calculated once
    cdef double s_incl = sin(incl)    #cos(incl) calculated once
    cdef double phi = 0.
    cdef Py_ssize_t i,j,k
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = x.shape[1]
    cdef Py_ssize_t K = 100
    
    #define HERQUINST STRUC 
    cdef HERQUINST herquinst  

    #   RHO DEFINITIONS
    cdef double rhoB = 0.
    cdef double rhoD = 0.
    cdef double sum_rhoD = 0.         #mi serve per mediare la vel
    cdef double sum_rhoB = 0.
    #per il momento la creo 100X100X100 perchè così sono i dati
    #si può migliorare
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    cdef np.ndarray[double,ndim=3,mode="c"] all_rhoD = np.zeros((N,J,K))
    cdef double[:,:,::1] all_rhoD_view = all_rhoD
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    cdef np.ndarray[double,ndim=3,mode = "c"] rho_dot_sigma2 = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_dot_sigma2_view = rho_dot_sigma2
    
    #V DEFINITIONS
    cdef double v_tmp = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
    cdef np.ndarray[double,ndim=3,mode = "c"] all_v = np.zeros((N,J,K))
    cdef double[:,:,::1] all_v_view = all_v
    cdef np.ndarray[double,ndim=2,mode = "c"] v_disc = np.zeros((N,J))
    cdef double[:,::1] v_disc_view = v_disc

    #questa è una cosa inutile quando tutto funziona va modificato
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N+1,J+1,3))
    cdef double[:,:,::1] tot_view = tot 
    
    #PSF definitions
    cdef double psf_tmp=0.
    cdef double sum_psf = 0.
    cdef np.ndarray[double,ndim=2,mode='c'] sum_psf_saved = np.zeros((N,J))
    cdef double[:,::1] sum_psf_saved_view = sum_psf_saved
    #masse passate in logaritmo
    Mb = 10**Mb #is it ok or better import pow from math?
    Md = 10**Md
    Mh = 10**Mh
    Rb = 10**Rb 
    Rd = 10**Rd
    Rh = 10**Rh 

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    
    for i in range(N):
        for j in range(J):
            sum_rhoD = 0.
            sum_rhoB = 0.
            sum_psf = 0.
            if 10**ydata[i,j,0] > 0:
                for k in range(K):
                    psf_tmp = psf_weight[i,j,k]
                    #queste operazioni è meglio farle in una funzione?
                    #per il fatto che sarebbe cdef e nogil...
                    x0 = x[i,j,k]-xcm
                    y0 = y[i,j,k]-ycm
                    xr = x0*c_theta + y0*s_theta 
                    yr = y0*c_theta - x0*s_theta
                    yd = yr/c_incl
                    r_proj = radius(x0,y0) 
                    r_true = radius(xr,yd)
                    phi = atan2(yd,xr)

                    #rhoB = hernquist_rho(Mb,Rb,r_proj)
                    herquinst = herquinst_function(Mb,Rb,r_proj)
                    rhoB = herquinst.rho 
                    rhoD = rho_D(Md,Rd,r_true,c_incl)
                    all_rhoD_view[i,j,k] = rhoD
                    #sum_rhoD = sum_rhoD + rhoD
                    sum_rhoD = sum_rhoD + rhoD*psf_tmp
                    sum_rhoB = sum_rhoB + rhoB*psf_tmp
                    tot_view[i,j,0] += psf_tmp*rho_tot(rhoB,rhoD)

                    v_tmp = v_tot(Mb,Rb,Md,Rd,Mh,Rh,s_incl,r_true,phi)
                    all_v_view[i,j,k] = v_tmp
                    tot_view[i,j,1] += psf_tmp*rhoD*v_tmp
                    v_disc_view[i,j] += psf_tmp*rhoD*v_tmp

                    sum_psf += psf_tmp
                    #questo è rho_H*sigma_H2 andrà sommato nel prossimo ciclo 
                    #con la dispersione del disco che devo ancora calcolare
                    #rho_dot_sigma2_view[i,j,k] = rhoB*hernquist_sigma(Mb,Rb,rhoB,r_proj)
                    rho_dot_sigma2_view[i,j,k] = rhoB*herquinst.sigma

                #le sommo finito il ciclo divido per 100, per fare la media
                #tot_view[i,j,0] = tot_view[i,j,0]/100.0
                tot_view[i,j,0] = tot_view[i,j,0]/sum_psf
                sum_psf_saved_view[i,j] = sum_psf
                if sum_rhoD == 0. :
                    tot_view[i,j,1] = 0.
                else:
                    #ma sum_rhoD contiene anche la PSF
                    tot_view[i,j,1] = tot_view[i,j,1]/(sum_rhoD+sum_rhoB)
                    v_disc_view[i,j] = v_disc_view[i,j]/sum_rhoD
            else:
                pass 
    
    v_tmp = 0.
    psf_tmp = 0.
    sum_psf = 0.
    for i in range(N):
        for j in range(J):
            #mean v in bin
            if 10**ydata[i,j,0]>0:
                v_tmp = v_disc_view[i,j]
                sum_psf = sum_psf_saved_view[i,j]
                for k in range(K):
                    #NB CI SONO DIVERSI MODI PER CALCOLARE LA SIGMA
                    tot_view[i,j,2] +=  psf_weight[i,j,k]*sigma_tot(all_rhoD_view[i,j,k],all_v_view[i,j,k],v_tmp,rho_dot_sigma2_view[i,j,k])

                #come denominatore avrei bisogno della somma su k delle densità
                #ma dato che so la densità media, e so che ho 100 punti, so anche la somma
                tot_view[i,j,2] = sqrt(tot_view[i,j,2]/(tot_view[i,j,0]*sum_psf))
            else:
                pass
    
    #si può ragionare su come ritornare direttamente un vettore
    #in questo modo dall'altra parte evito di fare .ravel
    return tot 


##                      ##
## MODEL PSF EVERYWERE  ##
##                      ##
#questo include la psf e calcola il modello ovunque
#lo utilizzo per il plot
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=3,mode='c'] model_psf_total (double[:,:,::1] x,          #x position refined grid è 99x99x100
                                                    double[:,:,::1] y,          #y position refined grid
                                                    double Mb,                  #bulge mass log
                                                    double Rb,                  #bulge radius
                                                    double Md,                  #disc mass log
                                                    double Rd,                  #disc radius
                                                    double Mh,                  #halo mass log
                                                    double Rh,                  #halo radius
                                                    double xcm,                 #center x position
                                                    double ycm,                 #center y position
                                                    double theta,               #P.A. angle (rotation in sky-plane)
                                                    double incl,                #inclination angle
                                                    double[:,:,::1]psf_weight):   #psf weight
    
    cdef double x0 = 0.               #x-xcm  traslation
    cdef double y0 = 0.               #y-ycm  traslation
    cdef double xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef double yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef double yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef double r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef double r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef double c_theta = cos(theta)  #cos(theta) calculated once
    cdef double s_theta = sin(theta)  #sin(theta) calculated once
    cdef double c_incl = cos(incl)    #cos(incl) calculated once
    cdef double s_incl = sin(incl)    #cos(incl) calculated once
    cdef double phi = 0.
    cdef Py_ssize_t i,j,k
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = x.shape[1]
    cdef Py_ssize_t K = 100
    
    #define HERQUINST STRUC 
    cdef HERQUINST herquinst  

    #   RHO DEFINITIONS
    cdef double rhoB = 0.
    cdef double rhoD = 0.
    cdef double sum_rhoD = 0.         #mi serve per mediare la vel
    #per il momento la creo 100X100X100 perchè così sono i dati
    #si può migliorare
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    cdef np.ndarray[double,ndim=3,mode="c"] all_rhoD = np.zeros((N,J,K))
    cdef double[:,:,::1] all_rhoD_view = all_rhoD
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    cdef np.ndarray[double,ndim=3,mode = "c"] rho_dot_sigma2 = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_dot_sigma2_view = rho_dot_sigma2
    
    #V DEFINITIONS
    cdef double v_tmp = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
    cdef np.ndarray[double,ndim=3,mode = "c"] all_v = np.zeros((N,J,K))
    cdef double[:,:,::1] all_v_view = all_v

    #questa è una cosa inutile quando tutto funziona va modificato
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N+1,J+1,3))
    cdef double[:,:,::1] tot_view = tot 
    
    #PSF definitions
    cdef double psf_tmp=0.
    cdef double sum_psf = 0.
    cdef np.ndarray[double,ndim=2,mode='c'] sum_psf_saved = np.zeros((N,J))
    cdef double[:,::1] sum_psf_saved_view = sum_psf_saved

    #masse passate in logaritmo
    Mb = 10**Mb #is it ok or better import pow from math?
    Md = 10**Md
    Mh = 10**Mh
    Rb = 10**Rb 
    Rd = 10**Rd
    Rh = 10**Rh 

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    
    for i in range(N):
        for j in range(J):
            sum_rhoD = 0.
            sum_psf = 0.
            for k in range(K):
                psf_tmp = psf_weight[i,j,k]
                #queste operazioni è meglio farle in una funzione?
                #per il fatto che sarebbe cdef e nogil...
                x0 = x[i,j,k]-xcm
                y0 = y[i,j,k]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)

                #rhoB = hernquist_rho(Mb,Rb,r_proj)
                herquinst = herquinst_function(Mb,Rb,r_proj)
                rhoB = herquinst.rho 
                rhoD = rho_D(Md,Rd,r_true,c_incl)
                all_rhoD_view[i,j,k] = rhoD
                #sum_rhoD = sum_rhoD + rhoD
                sum_rhoD = sum_rhoD + rhoD*psf_tmp
                tot_view[i,j,0] += psf_tmp*rho_tot(rhoB,rhoD)

                v_tmp = v_tot(Mb,Rb,Md,Rd,Mh,Rh,s_incl,r_true,phi)
                all_v_view[i,j,k] = v_tmp
                tot_view[i,j,1] += psf_tmp*rhoD*v_tmp

                #questo è rho_H*sigma_H2 andrà sommato nel prossimo ciclo 
                #con la dispersione del disco che devo ancora calcolare
                #rho_dot_sigma2_view[i,j,k] = rhoB*hernquist_sigma(Mb,Rb,rhoB,r_proj)
                rho_dot_sigma2_view[i,j,k] = rhoB*herquinst.sigma
                sum_psf += psf_tmp

            #le sommo finito il ciclo divido per 100, per fare la media
            tot_view[i,j,0] = tot_view[i,j,0]/sum_psf
            sum_psf_saved_view[i,j] = sum_psf
            if sum_rhoD == 0. :
                tot_view[i,j,1] = 0.
            else:
                tot_view[i,j,1] = tot_view[i,j,1]/sum_rhoD
    
    v_tmp = 0.
    psf_tmp = 0.
    sum_psf = 0.
    for i in range(N):
        for j in range(J):
            #mean v in bin
            v_tmp = tot_view[i,j,1]
            sum_psf = sum_psf_saved_view[i,j]
            for k in range(K):
                tot_view[i,j,2] +=  psf_weight[i,j,k]*sigma_tot(all_rhoD_view[i,j,k],all_v_view[i,j,k],v_tmp,rho_dot_sigma2_view[i,j,k])
        
            #come denominatore avrei bisogno della somma su k delle densità
            #ma dato che so la densità media, e so che ho 100 punti, so anche la somma
            tot_view[i,j,2] = sqrt(tot_view[i,j,2]/(tot_view[i,j,0]*sum_psf))
    
    #si può ragionare su come ritornare direttamente un vettore
    #in questo modo dall'altra parte evito di fare .ravel
    return tot 

##                                                 ##
## MODEL PSF EVERYWERE with v = v*rhoD/(rhoD+rhoB) ##
##                                                 ##
#questo include la psf e calcola il modello ovunque
#lo utilizzo per il plot
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=3,mode='c'] model_psf_modified_vel_total (double[:,:,::1] x,          #x position refined grid è 99x99x100
                                                    double[:,:,::1] y,          #y position refined grid
                                                    double Mb,                  #bulge mass log
                                                    double Rb,                  #bulge radius
                                                    double Md,                  #disc mass log
                                                    double Rd,                  #disc radius
                                                    double Mh,                  #halo mass log
                                                    double Rh,                  #halo radius
                                                    double xcm,                 #center x position
                                                    double ycm,                 #center y position
                                                    double theta,               #P.A. angle (rotation in sky-plane)
                                                    double incl,                #inclination angle
                                                    double[:,:,::1]psf_weight):   #psf weight
    
    cdef double x0 = 0.               #x-xcm  traslation
    cdef double y0 = 0.               #y-ycm  traslation
    cdef double xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef double yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef double yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef double r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef double r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef double c_theta = cos(theta)  #cos(theta) calculated once
    cdef double s_theta = sin(theta)  #sin(theta) calculated once
    cdef double c_incl = cos(incl)    #cos(incl) calculated once
    cdef double s_incl = sin(incl)    #cos(incl) calculated once
    cdef double phi = 0.
    cdef Py_ssize_t i,j,k
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = x.shape[1]
    cdef Py_ssize_t K = 100
    
    #define HERQUINST STRUC 
    cdef HERQUINST herquinst  

    #   RHO DEFINITIONS
    cdef double rhoB = 0.
    cdef double rhoD = 0.
    cdef double sum_rhoD = 0.         #mi serve per mediare la vel
    cdef double sum_rhoB = 0.
    #per il momento la creo 100X100X100 perchè così sono i dati
    #si può migliorare
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    cdef np.ndarray[double,ndim=3,mode="c"] all_rhoD = np.zeros((N,J,K))
    cdef double[:,:,::1] all_rhoD_view = all_rhoD
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    cdef np.ndarray[double,ndim=3,mode = "c"] rho_dot_sigma2 = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_dot_sigma2_view = rho_dot_sigma2
    
    #V DEFINITIONS
    cdef double v_tmp = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
    cdef np.ndarray[double,ndim=3,mode = "c"] all_v = np.zeros((N,J,K))
    cdef double[:,:,::1] all_v_view = all_v
    cdef np.ndarray[double,ndim=2,mode = "c"] v_disc = np.zeros((N,J))
    cdef double[:,::1] v_disc_view = v_disc

    #questa è una cosa inutile quando tutto funziona va modificato
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N+1,J+1,3))
    cdef double[:,:,::1] tot_view = tot 
    
    #PSF definitions
    cdef double psf_tmp=0.
    cdef double sum_psf = 0.
    cdef np.ndarray[double,ndim=2,mode='c'] sum_psf_saved = np.zeros((N,J))
    cdef double[:,::1] sum_psf_saved_view = sum_psf_saved

    #masse passate in logaritmo
    Mb = 10**Mb #is it ok or better import pow from math?
    Md = 10**Md
    Mh = 10**Mh
    Rb = 10**Rb 
    Rd = 10**Rd
    Rh = 10**Rh 

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    
    for i in range(N):
        for j in range(J):
            sum_rhoD = 0.
            sum_rhoB = 0.
            sum_psf = 0.
            for k in range(K):
                psf_tmp = psf_weight[i,j,k]
                #queste operazioni è meglio farle in una funzione?
                #per il fatto che sarebbe cdef e nogil...
                x0 = x[i,j,k]-xcm
                y0 = y[i,j,k]-ycm
                xr = x0*c_theta + y0*s_theta 
                yr = y0*c_theta - x0*s_theta
                yd = yr/c_incl
                r_proj = radius(x0,y0) 
                r_true = radius(xr,yd)
                phi = atan2(yd,xr)

                #rhoB = hernquist_rho(Mb,Rb,r_proj)
                herquinst = herquinst_function(Mb,Rb,r_proj)
                rhoB = herquinst.rho 
                rhoD = rho_D(Md,Rd,r_true,c_incl)
                all_rhoD_view[i,j,k] = rhoD
                #sum_rhoD = sum_rhoD + rhoD
                sum_rhoD = sum_rhoD + rhoD*psf_tmp
                sum_rhoB = sum_rhoB + rhoB*psf_tmp
                tot_view[i,j,0] += psf_tmp*rho_tot(rhoB,rhoD)

                v_tmp = v_tot(Mb,Rb,Md,Rd,Mh,Rh,s_incl,r_true,phi)
                all_v_view[i,j,k] = v_tmp
                tot_view[i,j,1] += psf_tmp*rhoD*v_tmp
                v_disc_view[i,j] += psf_tmp*rhoD*v_tmp
                #questo è rho_H*sigma_H2 andrà sommato nel prossimo ciclo 
                #con la dispersione del disco che devo ancora calcolare
                #rho_dot_sigma2_view[i,j,k] = rhoB*hernquist_sigma(Mb,Rb,rhoB,r_proj)
                rho_dot_sigma2_view[i,j,k] = rhoB*herquinst.sigma
                sum_psf += psf_tmp

            #le sommo finito il ciclo divido per 100, per fare la media
            tot_view[i,j,0] = tot_view[i,j,0]/sum_psf
            sum_psf_saved_view[i,j] = sum_psf
            if sum_rhoD == 0. :
                tot_view[i,j,1] = 0.
            else:
                tot_view[i,j,1] = tot_view[i,j,1]/(sum_rhoD+sum_rhoB)
                v_disc_view[i,j] = v_disc_view[i,j]/(sum_rhoD)
    v_tmp = 0.
    psf_tmp = 0.
    sum_psf = 0.
    for i in range(N):
        for j in range(J):
            #mean v in bin
            #v_tmp = tot_view[i,j,1]
            v_tmp = v_disc_view[i,j]
            sum_psf = sum_psf_saved_view[i,j]
            for k in range(K):
                tot_view[i,j,2] +=  psf_weight[i,j,k]*sigma_tot(all_rhoD_view[i,j,k],all_v_view[i,j,k],v_tmp,rho_dot_sigma2_view[i,j,k])
        
            #come denominatore avrei bisogno della somma su k delle densità
            #ma dato che so la densità media, e so che ho 100 punti, so anche la somma
            tot_view[i,j,2] = sqrt(tot_view[i,j,2]/(tot_view[i,j,0]*sum_psf))
    
    #si può ragionare su come ritornare direttamente un vettore
    #in questo modo dall'altra parte evito di fare .ravel
    return tot 

##                                  ##
## MODEL PSF EVERYWERE + SMOOTHING  ##
##                                  ##
#questo prende i dati li passa a model_psf_total e poi fa lo smoothing
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=3,mode='c'] model_smoothing (double[:,:,::1] x,          #x position refined grid è 99x99x100
                                   double[:,:,::1] y,          #y position refined grid
                                   double Mb,                  #bulge mass log
                                   double Rb,                  #bulge radius
                                   double Md,                  #disc mass log
                                   double Rd,                  #disc radius
                                   double Mh,                  #halo mass log
                                   double Rh,                  #halo radius
                                   double xcm,                 #center x position
                                   double ycm,                 #center y position
                                   double theta,               #P.A. angle (rotation in sky-plane)
                                   double incl,                #inclination angle
                                   double [:,:,::1] psf_weight,  #weight of the PSF
                                   double [:,::1] smoothing):  #smoothing matrix
    
    #uso direttamente model perchè tanto devo salvare tutto
    cdef Py_ssize_t i,j,k,l,m,n
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = x.shape[1]
    cdef Py_ssize_t K = 100
    cdef double s = 0.     #smoothing
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N+1,J+1,3))
    cdef double rho_tmp_data = 0.
    cdef double sigma_tmp_data = 0.
    cdef double rho_partial_tmp = 0.

    #smoothed quantities def
    cdef np.ndarray[double,ndim=3,mode='c'] tot_smooth = np.zeros((N+1,J+1,3))
    cdef double[:,:,::1] tot_smooth_view = tot_smooth
    
    cdef double rho_smooth = 0.
    cdef double v_smooth = 0.
    cdef double sigma_smooth = 0.

    #partial quantities case s = 2
    cdef np.ndarray[double,ndim=1,mode='c'] s2_rho = np.zeros((9)) 
    cdef double[::1] s2_rho_view = s2_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s2_v = np.zeros((9)) 
    cdef double[::1] s2_v_view = s2_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s2_sigma = np.zeros((9)) 
    cdef double[::1] s2_sigma_view = s2_sigma 

    #partial quantities case s = 3
    cdef np.ndarray[double,ndim=1,mode='c'] s3_rho = np.zeros(25) 
    cdef double[::1] s3_rho_view = s3_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s3_v = np.zeros(25) 
    cdef double[::1] s3_v_view = s3_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s3_sigma = np.zeros(25) 
    cdef double[::1] s3_sigma_view = s3_sigma

    #partial quantities case s = 4
    cdef np.ndarray[double,ndim=1,mode='c'] s4_rho = np.zeros((49)) 
    cdef double[::1] s4_rho_view = s4_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s4_v = np.zeros((49)) 
    cdef double[::1] s4_v_view = s4_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s4_sigma = np.zeros((49)) 
    cdef double[::1] s4_sigma_view = s4_sigma

    #partial quantities case s = 5
    cdef np.ndarray[double,ndim=1,mode='c'] s5_rho = np.zeros((81)) 
    cdef double[::1] s5_rho_view = s5_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s5_v = np.zeros((81)) 
    cdef double[::1] s5_v_view = s5_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s5_sigma = np.zeros((81)) 
    cdef double[::1] s5_sigma_view = s5_sigma

    
    tot = model_psf_total(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,psf_weight)
    #ora devo fare lo smoothing dei dati
    
    for i in range(N):
        for j in range(J):
            s = smoothing[i,j]
            if s == 1.0 :
                tot_smooth_view[i,j,0] = tot[i,j,0]
                tot_smooth_view[i,j,1] = tot[i,j,1]
                tot_smooth_view[i,j,2] = tot[i,j,2]
                
            elif s == 2.0:
                n = 0
                for k in range(i-1,i+2):

                    if k < 0 or k>N:#check che non sia al bordo della griglia
                        s2_rho_view[n] = 0.
                        s2_v_view[n] = 0.
                        s2_sigma_view[n] = 0.
                        n+=1
                    else:
                        for l in range(j-1,j+2):
                            if l < 0 or l>J:#check bordo
                                s2_rho_view[n] = 0.
                                s2_v_view[n] = 0.
                                s2_sigma_view[n] = 0.
                                n+=1
                            else:
                                s2_rho_view[n] = tot[k,l,0]
                                s2_v_view[n] = tot[k,l,1]
                                s2_sigma_view[n] = tot[k,l,2]
                                n+=1

                tot_smooth_view[i,j,0] = median(s2_rho_view,0.)
                tot_smooth_view[i,j,1] = median(s2_v_view,1.)
                tot_smooth_view[i,j,2] = median(s2_sigma_view,0.)

            elif s == 3.0:
                n = 0
                for k in range(i-2,i+3):
                    if k < 0 or k>N:#check che non sia al bordo della griglia
                        s3_rho_view[n] = 0.
                        s3_v_view[n] = 0.
                        s3_sigma_view[n] = 0.
                        n +=1                              
                    else:
                        for l in range(j-2,j+3):
                            if l < 0 or l>J:#check bordo
                                s3_rho_view[n] = 0.
                                s3_v_view[n] = 0.
                                s3_sigma_view[n] = 0.
                                n +=1
                            else:
                                s3_rho_view[n] = tot[k,l,0]
                                s3_v_view[n] = tot[k,l,1]
                                s3_sigma_view[n] = tot[k,l,2]
                                n +=1

                tot_smooth_view[i,j,0] = median(s3_rho_view,0.)
                tot_smooth_view[i,j,1] = median(s3_v_view,1.)
                tot_smooth_view[i,j,2] = median(s3_sigma_view,0.)
 
            elif s == 4.0:
                n = 0
                for k in range(i-3,i+4):
                    if k < 0 or k>N:#check che non sia al bordo della griglia
                        s4_rho_view[n] = 0.
                        s4_v_view[n] = 0.
                        s4_sigma_view[n] = 0.
                        n +=1
                    else:
                        for l in range(j-3,j+4):
                            if l < 0 or l>J:#check bordo
                                s4_rho_view[n] = 0.
                                s4_v_view[n] = 0.
                                s4_sigma_view[n] = 0.
                                n +=1
                            else:
                                s4_rho_view[n] = tot[k,l,0]
                                s4_v_view[n] = tot[k,l,1]
                                s4_sigma_view[n] = tot[k,l,2]
                                n +=1

                tot_smooth_view[i,j,0] = median(s4_rho_view,0.)
                tot_smooth_view[i,j,1] = median(s4_v_view,1.)
                tot_smooth_view[i,j,2] = median(s4_sigma_view,0.)

            elif s == 5.0:
                n = 0
                for k in range(i-4,i+5):
                    if k < 0 or k>N:#check che non sia al bordo della griglia
                        s5_rho_view[n] = 0.
                        s5_v_view[n] = 0.
                        s5_sigma_view[n] = 0.
                        n +=1
                    else:
                        for l in range(j-4,j+5):
                            if l < 0 or l>J:#check bordo
                                s5_rho_view[n] = 0.
                                s5_v_view[n] = 0.
                                s5_sigma_view[n] = 0.
                                n +=1
                            else:
                                s5_rho_view[n] = tot[k,l,0] 
                                s5_v_view[n] = tot[k,l,1]
                                s5_sigma_view[n] = tot[k,l,2]
                                n +=1
                tot_smooth_view[i,j,0] = median(s5_rho_view,0.)
                tot_smooth_view[i,j,1] = median(s5_v_view,1.)
                tot_smooth_view[i,j,2] = median(s5_sigma_view,0.)

 
    return tot_smooth


##                                                              ##
## MODEL PSF EVERYWERE + SMOOTHING with v = rhoD*v/(rhoD+rhoB)  ##
##                                                              ##
#questo prende i dati li passa a model_psf_total e poi fa lo smoothing
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=3,mode='c'] model_smoothing_modified_vel (double[:,:,::1] x,          #x position refined grid è 99x99x100
                                   double[:,:,::1] y,          #y position refined grid
                                   double Mb,                  #bulge mass log
                                   double Rb,                  #bulge radius
                                   double Md,                  #disc mass log
                                   double Rd,                  #disc radius
                                   double Mh,                  #halo mass log
                                   double Rh,                  #halo radius
                                   double xcm,                 #center x position
                                   double ycm,                 #center y position
                                   double theta,               #P.A. angle (rotation in sky-plane)
                                   double incl,                #inclination angle
                                   double [:,:,::1] psf_weight,  #weight of the PSF
                                   double [:,::1] smoothing):  #smoothing matrix
    
    #uso direttamente model perchè tanto devo salvare tutto
    cdef Py_ssize_t i,j,k,l,m,n
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = x.shape[1]
    cdef Py_ssize_t K = 100
    cdef double s = 0.     #smoothing
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N+1,J+1,3))
    cdef double rho_tmp_data = 0.
    cdef double sigma_tmp_data = 0.
    cdef double rho_partial_tmp = 0.

    #smoothed quantities def
    cdef np.ndarray[double,ndim=3,mode='c'] tot_smooth = np.zeros((N+1,J+1,3))
    cdef double[:,:,::1] tot_smooth_view = tot_smooth
    
    cdef double rho_smooth = 0.
    cdef double v_smooth = 0.
    cdef double sigma_smooth = 0.

    #partial quantities case s = 2
    cdef np.ndarray[double,ndim=1,mode='c'] s2_rho = np.zeros((9)) 
    cdef double[::1] s2_rho_view = s2_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s2_v = np.zeros((9)) 
    cdef double[::1] s2_v_view = s2_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s2_sigma = np.zeros((9)) 
    cdef double[::1] s2_sigma_view = s2_sigma 

    #partial quantities case s = 3
    cdef np.ndarray[double,ndim=1,mode='c'] s3_rho = np.zeros(25) 
    cdef double[::1] s3_rho_view = s3_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s3_v = np.zeros(25) 
    cdef double[::1] s3_v_view = s3_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s3_sigma = np.zeros(25) 
    cdef double[::1] s3_sigma_view = s3_sigma

    #partial quantities case s = 4
    cdef np.ndarray[double,ndim=1,mode='c'] s4_rho = np.zeros((49)) 
    cdef double[::1] s4_rho_view = s4_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s4_v = np.zeros((49)) 
    cdef double[::1] s4_v_view = s4_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s4_sigma = np.zeros((49)) 
    cdef double[::1] s4_sigma_view = s4_sigma

    #partial quantities case s = 5
    cdef np.ndarray[double,ndim=1,mode='c'] s5_rho = np.zeros((81)) 
    cdef double[::1] s5_rho_view = s5_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s5_v = np.zeros((81)) 
    cdef double[::1] s5_v_view = s5_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s5_sigma = np.zeros((81)) 
    cdef double[::1] s5_sigma_view = s5_sigma

    
    tot = model_psf_modified_vel_total(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,psf_weight)
    #ora devo fare lo smoothing dei dati
    
    for i in range(N):
        for j in range(J):
            s = smoothing[i,j]
            if s == 1.0 :
                tot_smooth_view[i,j,0] = tot[i,j,0]
                tot_smooth_view[i,j,1] = tot[i,j,1]
                tot_smooth_view[i,j,2] = tot[i,j,2]
                
            elif s == 2.0:
                n = 0
                for k in range(i-1,i+2):

                    if k < 0 or k>N:#check che non sia al bordo della griglia
                        s2_rho_view[n] = 0.
                        s2_v_view[n] = 0.
                        s2_sigma_view[n] = 0.
                        n+=1
                    else:
                        for l in range(j-1,j+2):
                            if l < 0 or l>J:#check bordo
                                s2_rho_view[n] = 0.
                                s2_v_view[n] = 0.
                                s2_sigma_view[n] = 0.
                                n+=1
                            else:
                                s2_rho_view[n] = tot[k,l,0]
                                s2_v_view[n] = tot[k,l,1]
                                s2_sigma_view[n] = tot[k,l,2]
                                n+=1

                tot_smooth_view[i,j,0] = median(s2_rho_view,0.)
                tot_smooth_view[i,j,1] = median(s2_v_view,1.)
                tot_smooth_view[i,j,2] = median(s2_sigma_view,0.)

            elif s == 3.0:
                n = 0
                for k in range(i-2,i+3):
                    if k < 0 or k>N:#check che non sia al bordo della griglia
                        s3_rho_view[n] = 0.
                        s3_v_view[n] = 0.
                        s3_sigma_view[n] = 0.
                        n +=1                              
                    else:
                        for l in range(j-2,j+3):
                            if l < 0 or l>J:#check bordo
                                s3_rho_view[n] = 0.
                                s3_v_view[n] = 0.
                                s3_sigma_view[n] = 0.
                                n +=1
                            else:
                                s3_rho_view[n] = tot[k,l,0]
                                s3_v_view[n] = tot[k,l,1]
                                s3_sigma_view[n] = tot[k,l,2]
                                n +=1

                tot_smooth_view[i,j,0] = median(s3_rho_view,0.)
                tot_smooth_view[i,j,1] = median(s3_v_view,1.)
                tot_smooth_view[i,j,2] = median(s3_sigma_view,0.)
 
            elif s == 4.0:
                n = 0
                for k in range(i-3,i+4):
                    if k < 0 or k>N:#check che non sia al bordo della griglia
                        s4_rho_view[n] = 0.
                        s4_v_view[n] = 0.
                        s4_sigma_view[n] = 0.
                        n +=1
                    else:
                        for l in range(j-3,j+4):
                            if l < 0 or l>J:#check bordo
                                s4_rho_view[n] = 0.
                                s4_v_view[n] = 0.
                                s4_sigma_view[n] = 0.
                                n +=1
                            else:
                                s4_rho_view[n] = tot[k,l,0]
                                s4_v_view[n] = tot[k,l,1]
                                s4_sigma_view[n] = tot[k,l,2]
                                n +=1

                tot_smooth_view[i,j,0] = median(s4_rho_view,0.)
                tot_smooth_view[i,j,1] = median(s4_v_view,1.)
                tot_smooth_view[i,j,2] = median(s4_sigma_view,0.)

            elif s == 5.0:
                n = 0
                for k in range(i-4,i+5):
                    if k < 0 or k>N:#check che non sia al bordo della griglia
                        s5_rho_view[n] = 0.
                        s5_v_view[n] = 0.
                        s5_sigma_view[n] = 0.
                        n +=1
                    else:
                        for l in range(j-4,j+5):
                            if l < 0 or l>J:#check bordo
                                s5_rho_view[n] = 0.
                                s5_v_view[n] = 0.
                                s5_sigma_view[n] = 0.
                                n +=1
                            else:
                                s5_rho_view[n] = tot[k,l,0] 
                                s5_v_view[n] = tot[k,l,1]
                                s5_sigma_view[n] = tot[k,l,2]
                                n +=1
                tot_smooth_view[i,j,0] = median(s5_rho_view,0.)
                tot_smooth_view[i,j,1] = median(s5_v_view,1.)
                tot_smooth_view[i,j,2] = median(s5_sigma_view,0.)

 
    return tot_smooth


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
                    #LIKELIHOOD
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#espressione della likelihood
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double lk(double mod,double data,double data_err) nogil:
    cdef double mod_minus_data = data - mod 
    cdef double data_err2 = data_err*data_err
    cdef double lk = 0.
    lk = -0.5 * (mod_minus_data*mod_minus_data/data_err2 + log(2*M_PI*data_err2))

    return lk



##                    ##
##     NORMAL LIKELI  ##
##                    ##
#calcolo likelihood senza psf e smoothing
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double likelihood (double[:,:,::1] x,#x position refined grid è 99x99x100
                    double[:,:,::1] y,#y position refined grid
                    double Mb,        #bulge mass log
                    double Rb,        #bulge radius
                    double Md,        #disc mass log
                    double Rd,        #disc radius
                    double Mh,        #halo mass log
                    double Rh,        #halo radius
                    double xcm,       #center x position
                    double ycm,       #center y position
                    double theta,     #P.A. angle (rotation in sky-plane)
                    double incl,      #inclination angle
                    double [:,:,::1]ydata,     #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                    double [:,:,::1]yerr):     #err
    
    cdef double x0 = 0.               #x-xcm  traslation
    cdef double y0 = 0.               #y-ycm  traslation
    cdef double xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef double yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef double yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef double r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef double r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef double c_theta = cos(theta)  #cos(theta) calculated once
    cdef double s_theta = sin(theta)  #sin(theta) calculated once
    cdef double c_incl = cos(incl)    #cos(incl) calculated once
    cdef double s_incl = sin(incl)    #cos(incl) calculated once
    cdef double phi = 0.
    cdef Py_ssize_t i,j,k
    #così ciclo sui dati non sulla griglia
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = y.shape[1]
    cdef Py_ssize_t K = 100

    #define HERQUINST STRUCTURE
    cdef HERQUINST herquinst
    #   RHO DEFINITIONS
    cdef double rhoB = 0.
    cdef double rhoD = 0.
    cdef double sum_rhoD = 0.         #mi serve per mediare la vel
    cdef double rho_tmp = 0.
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    cdef np.ndarray[double,ndim=3,mode="c"] all_rhoD = np.zeros((N,J,K))
    cdef double[:,:,::1] all_rhoD_view = all_rhoD
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    cdef np.ndarray[double,ndim=3,mode = "c"] rho_dot_sigma2 = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_dot_sigma2_view = rho_dot_sigma2
    
    #V DEFINITIONS
    cdef double v_tmp = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
    cdef np.ndarray[double,ndim=3,mode = "c"] all_v = np.zeros((N,J,K))
    cdef double[:,:,::1] all_v_view = all_v

    #SIGMA DEFINITIONS
    cdef double sigma_tmp = 0.

    #si può forse pensare ad un altro moto di salvare le quantità che calcolo
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N,J,2))
    cdef double[:,:,::1] tot_view = tot 
    
    #LIKELIHOOD DEFINITIONS
    cdef double likelihood = 0.
    #masse passate in logaritmo
    Mb = 10**Mb #is it ok or better import pow from math?
    Md = 10**Md
    Mh = 10**Mh
    Rb = 10**Rb
    Rd = 10**Rd 
    Rh = 10**Rh

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    for i in range(N):
        for j in range(J):
            sum_rhoD = 0.
            #solo dove ho dati faccio i conti
            if 10**ydata[i,j,0] > 0.:

                for k in range(K):
                    #queste operazioni è meglio farle in una funzione?
                    #per il fatto che sarebbe cdef e nogil...
                    x0 = x[i,j,k]-xcm
                    y0 = y[i,j,k]-ycm
                    xr = x0*c_theta + y0*s_theta 
                    yr = y0*c_theta - x0*s_theta
                    yd = yr/c_incl
                    r_proj = radius(x0,y0) 
                    r_true = radius(xr,yd)
                    phi = atan2(yd,xr)
                    
                    herquinst = herquinst_function(Mb,Rb,r_proj)
                    rhoB = herquinst.rho 
                    rhoD = rho_D(Md,Rd,r_true,c_incl)
                    all_rhoD_view[i,j,k] = rhoD
                    sum_rhoD = sum_rhoD + rhoD
                    tot_view[i,j,0] += rho_tot(rhoB,rhoD)

                    v_tmp = v_tot(Mb,Rb,Md,Rd,Mh,Rh,s_incl,r_true,phi)
                    all_v_view[i,j,k] = v_tmp
                    tot_view[i,j,1] += rhoD*v_tmp

                    #questo è rho_H*sigma_H2 andrà sommato nel prossimo ciclo 
                    #con la dispersione del disco che devo ancora calcolare
                    rho_dot_sigma2_view[i,j,k] = rhoB*herquinst.sigma
            
            else:
                pass

            #le sommo finito il ciclo divido per 100, per fare la media
            tot_view[i,j,0] = tot_view[i,j,0]/100.0
            if sum_rhoD == 0. :
                tot_view[i,j,1] = 0.
            else:
                tot_view[i,j,1] = tot_view[i,j,1]/sum_rhoD 

    v_tmp = 0.
    for i in range(N):
        for j in range(J):
            #vedere se è il caso di mettere altri check
            #questo serve nel caso ho una griglia in cui dei punti sono con densità nulla
            #i calcoli li faccio solo dove serve
            rho_tmp = ydata[i,j,0]
            #questo rallenta? c'è un supporto in C di inf?
            if 10**rho_tmp > 0.:
                #mean v in bin
                v_tmp = tot_view[i,j,1]
                #usando sigma_tmp evito di allocare memoria per la sigma
                #purtroppo non posso evitarlo per rho e v dato che mi servono per calcolare sigma
                sigma_tmp = 0.
                if ydata[i,j,2] > 120.:
                    for k in range(K):
                        #tot_view[i,j,2] +=  sigma_tot(all_rhoD_view[i,j,k],all_v_view[i,j,k],v_tmp,rho_dot_sigma2_view[i,j,k])

                        sigma_tmp +=  sigma_tot(all_rhoD_view[i,j,k],all_v_view[i,j,k],v_tmp,rho_dot_sigma2_view[i,j,k])
                    #come denominatore avrei bisogno della somma su k delle densità
                    #ma dato che so la densità media, e so che ho 100 punti, so anche la somma
                    #tot_view[i,j,2] = sqrt(tot_view[i,j,2]/(tot_view[i,j,0]*100.0))

                    sigma_tmp = sqrt(sigma_tmp/(tot_view[i,j,0]*100.0))

                    likelihood += lk(log10(tot_view[i,j,0]),rho_tmp,yerr[i,j,0]) + lk(tot_view[i,j,1],ydata[i,j,1],yerr[i,j,1]) + lk(sigma_tmp,ydata[i,j,2],yerr[i,j,2])
                else:
                    likelihood += lk(log10(tot_view[i,j,0]),rho_tmp,yerr[i,j,0]) + lk(tot_view[i,j,1],ydata[i,j,1],yerr[i,j,1]) 
            else:
                pass
    #si può ragionare su come ritornare direttamente un vettore
    #in questo modo dall'altra parte evito di fare .ravel
    return likelihood


##                    ##
##     SMOOTHED LK    ##
##                    ##
#calcolo likelihood con smothing e psf utilizzo model_psf e poi ci faccio lo smoothing
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double likelihood_smoothing (double[:,:,::1] x,          #x position refined grid è 99x99x100
                                   double[:,:,::1] y,          #y position refined grid
                                   double Mb,                  #bulge mass log
                                   double Rb,                  #bulge radius
                                   double Md,                  #disc mass log
                                   double Rd,                  #disc radius
                                   double Mh,                  #halo mass log
                                   double Rh,                  #halo radius
                                   double xcm,                 #center x position
                                   double ycm,                 #center y position
                                   double theta,               #P.A. angle (rotation in sky-plane)
                                   double incl,                #inclination angle
                                   double [:,:,::1]ydata,      #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                                   double [:,:,::1]yerr,       #err
                                   double [:,:,::1] psf_weight,  #weight of the PSF
                                   double [:,::1] smoothing):  #smoothing matrix
    
    #uso direttamente model perchè tanto devo salvare tutto
    cdef Py_ssize_t i,j,k,l,m,n
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = x.shape[1]
    cdef Py_ssize_t K = 100
    cdef double s = 0.     #smoothing
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N+1,J+1,3))
    cdef double rho_tmp_data = 0.
    cdef double sigma_tmp_data = 0.
    cdef double rho_partial_tmp = 0.

    cdef double rho_smooth = 0.
    cdef double v_smooth = 0.
    cdef double sigma_smooth = 0.

    #partial quantities case s = 2
    cdef np.ndarray[double,ndim=1,mode='c'] s2_rho = np.zeros((9)) 
    cdef double[::1] s2_rho_view = s2_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s2_v = np.zeros((9)) 
    cdef double[::1] s2_v_view = s2_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s2_sigma = np.zeros((9)) 
    cdef double[::1] s2_sigma_view = s2_sigma 

    #partial quantities case s = 3
    cdef np.ndarray[double,ndim=1,mode='c'] s3_rho = np.zeros(25) 
    cdef double[::1] s3_rho_view = s3_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s3_v = np.zeros(25) 
    cdef double[::1] s3_v_view = s3_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s3_sigma = np.zeros(25) 
    cdef double[::1] s3_sigma_view = s3_sigma

    #partial quantities case s = 4
    cdef np.ndarray[double,ndim=1,mode='c'] s4_rho = np.zeros((49)) 
    cdef double[::1] s4_rho_view = s4_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s4_v = np.zeros((49)) 
    cdef double[::1] s4_v_view = s4_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s4_sigma = np.zeros((49)) 
    cdef double[::1] s4_sigma_view = s4_sigma

    #partial quantities case s = 5
    cdef np.ndarray[double,ndim=1,mode='c'] s5_rho = np.zeros((81)) 
    cdef double[::1] s5_rho_view = s5_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s5_v = np.zeros((81)) 
    cdef double[::1] s5_v_view = s5_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s5_sigma = np.zeros((81)) 
    cdef double[::1] s5_sigma_view = s5_sigma


    #likelihood definitions
    cdef double likelihood = 0.

    tot = model_psf(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,ydata,psf_weight)
    #ora devo fare lo smoothing dei dati
    for i in range(N):
        for j in range(J):
            rho_tmp_data = ydata[i,j,0]
            if 10**rho_tmp_data > 0.:
                s = smoothing[i,j]
                if s == 1.0 :
                    rho_smooth = tot[i,j,0]
                    v_smooth = tot[i,j,1]
                    sigma_smooth = tot[i,j,2]
                
                elif s == 2.0:
                    n = 0
                    for k in range(i-1,i+2):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s2_rho_view[n] = 0.
                            s2_v_view[n] = 0.
                            s2_sigma_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-1,j+2):
                                if l < 0 or l>J:#check bordo
                                    s2_rho_view[n] = 0.
                                    s2_v_view[n] = 0.
                                    s2_sigma_view[n] = 0.
                                    n+=1
                                else:
                                    s2_rho_view[n] = tot[k,l,0]
                                    s2_v_view[n] = tot[k,l,1]
                                    s2_sigma_view[n] = tot[k,l,2]
                                    n+=1

                    rho_smooth = median(s2_rho_view,0.)
                    v_smooth = median(s2_v_view,1.)
                    sigma_smooth = median(s2_sigma_view,0.)
                
                elif s == 3.0:
                    n = 0
                    for k in range(i-2,i+3):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s3_rho_view[n] = 0.
                            s3_v_view[n] = 0.
                            s3_sigma_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-2,j+3):
                                if l < 0 or l>J:#check bordo
                                    s3_rho_view[n] = 0.
                                    s3_v_view[n] = 0.
                                    s3_sigma_view[n] = 0.
                                    n+=1
                                else:
                                    s3_rho_view[n] = tot[k,l,0]
                                    s3_v_view[n] = tot[k,l,1]
                                    s3_sigma_view[n] = tot[k,l,2]
                                    n+=1

                    rho_smooth = median(s3_rho_view,0.)
                    v_smooth = median(s3_v_view,1.)
                    sigma_smooth = median(s3_sigma_view,0.)

                    
                elif s == 4.0:
                    n = 0
                    for k in range(i-3,i+4):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s4_rho_view[n] = 0.
                            s4_v_view[n] = 0.
                            s4_sigma_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-3,j+4):
                                if l < 0 or l>J:#check bordo
                                    s4_rho_view[n] = 0.
                                    s4_v_view[n] = 0.
                                    s4_sigma_view[n] = 0.
                                    n+=1
                                else:
                                    s4_rho_view[n] = tot[k,l,0]
                                    s4_v_view[n] = tot[k,l,1]
                                    s4_sigma_view[n] = tot[k,l,2]
                                    n+=1

                    rho_smooth = median(s4_rho_view,0.)
                    v_smooth = median(s4_v_view,1.)
                    sigma_smooth = median(s4_sigma_view,0.)
                
                elif s == 5.0:
                    n = 0
                    for k in range(i-4,i+5):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s5_rho_view[n] = 0.
                            s5_v_view[n] = 0.
                            s5_sigma_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-4,j+5):
                                if l < 0 or l>J:#check bordo
                                    s5_rho_view[n] = 0.
                                    s5_v_view[n] = 0.
                                    s5_sigma_view[n] = 0.
                                    n+=1
                                else:
                                    s5_rho_view[n] = tot[k,l,0]
                                    s5_v_view[n] = tot[k,l,1]
                                    s5_sigma_view[n] = tot[k,l,2]
                                    n+=1

                    rho_smooth = median(s5_rho_view,0.)
                    v_smooth = median(s5_v_view,1.)
                    sigma_smooth = median(s5_sigma_view,0.)
                
                sigma_tmp_data = ydata[i,j,2]
                if sigma_tmp_data > 120.:
                    likelihood += lk(log10(rho_smooth),rho_tmp_data,yerr[i,j,0]) + lk(v_smooth,ydata[i,j,1],yerr[i,j,1]) + lk(sigma_smooth,sigma_tmp_data,yerr[i,j,2])
                else:
                    likelihood += lk(log10(rho_smooth),rho_tmp_data,yerr[i,j,0]) + lk(v_smooth,ydata[i,j,1],yerr[i,j,1]) 
            else:
                pass
    return likelihood


##                                                ##
##     SMOOTHED LK with v = rhoD*v/(rhoD+rhoB)    ##
##                                                ##
#calcolo likelihood con smothing e psf utilizzo model_psf e poi ci faccio lo smoothing
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double likelihood_smoothing_modified_vel (double[:,:,::1] x,          #x position refined grid è 99x99x100
                                   double[:,:,::1] y,          #y position refined grid
                                   double Mb,                  #bulge mass log
                                   double Rb,                  #bulge radius
                                   double Md,                  #disc mass log
                                   double Rd,                  #disc radius
                                   double Mh,                  #halo mass log
                                   double Rh,                  #halo radius
                                   double xcm,                 #center x position
                                   double ycm,                 #center y position
                                   double theta,               #P.A. angle (rotation in sky-plane)
                                   double incl,                #inclination angle
                                   double [:,:,::1]ydata,      #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                                   double [:,:,::1]yerr,       #err
                                   double [:,:,::1] psf_weight,  #weight of the PSF
                                   double [:,::1] smoothing):  #smoothing matrix
    
    #uso direttamente model perchè tanto devo salvare tutto
    cdef Py_ssize_t i,j,k,l,m,n
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = x.shape[1]
    cdef Py_ssize_t K = 100
    cdef double s = 0.     #smoothing
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N+1,J+1,3))
    cdef double rho_tmp_data = 0.
    cdef double sigma_tmp_data = 0.
    cdef double rho_partial_tmp = 0.

    cdef double rho_smooth = 0.
    cdef double v_smooth = 0.
    cdef double sigma_smooth = 0.

    #partial quantities case s = 2
    cdef np.ndarray[double,ndim=1,mode='c'] s2_rho = np.zeros((9)) 
    cdef double[::1] s2_rho_view = s2_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s2_v = np.zeros((9)) 
    cdef double[::1] s2_v_view = s2_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s2_sigma = np.zeros((9)) 
    cdef double[::1] s2_sigma_view = s2_sigma 

    #partial quantities case s = 3
    cdef np.ndarray[double,ndim=1,mode='c'] s3_rho = np.zeros(25) 
    cdef double[::1] s3_rho_view = s3_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s3_v = np.zeros(25) 
    cdef double[::1] s3_v_view = s3_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s3_sigma = np.zeros(25) 
    cdef double[::1] s3_sigma_view = s3_sigma

    #partial quantities case s = 4
    cdef np.ndarray[double,ndim=1,mode='c'] s4_rho = np.zeros((49)) 
    cdef double[::1] s4_rho_view = s4_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s4_v = np.zeros((49)) 
    cdef double[::1] s4_v_view = s4_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s4_sigma = np.zeros((49)) 
    cdef double[::1] s4_sigma_view = s4_sigma

    #partial quantities case s = 5
    cdef np.ndarray[double,ndim=1,mode='c'] s5_rho = np.zeros((81)) 
    cdef double[::1] s5_rho_view = s5_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s5_v = np.zeros((81)) 
    cdef double[::1] s5_v_view = s5_v 
    cdef np.ndarray[double,ndim=1,mode='c'] s5_sigma = np.zeros((81)) 
    cdef double[::1] s5_sigma_view = s5_sigma


    #likelihood definitions
    cdef double likelihood = 0.

    tot = model_psf_modified_vel(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,ydata,psf_weight)
    #ora devo fare lo smoothing dei dati
    for i in range(N):
        for j in range(J):
            rho_tmp_data = ydata[i,j,0]
            if 10**rho_tmp_data > 0.:
                s = smoothing[i,j]
                if s == 1.0 :
                    rho_smooth = tot[i,j,0]
                    v_smooth = tot[i,j,1]
                    sigma_smooth = tot[i,j,2]
                
                elif s == 2.0:
                    n = 0
                    for k in range(i-1,i+2):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s2_rho_view[n] = 0.
                            s2_v_view[n] = 0.
                            s2_sigma_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-1,j+2):
                                if l < 0 or l>J:#check bordo
                                    s2_rho_view[n] = 0.
                                    s2_v_view[n] = 0.
                                    s2_sigma_view[n] = 0.
                                    n+=1
                                else:
                                    s2_rho_view[n] = tot[k,l,0]
                                    s2_v_view[n] = tot[k,l,1]
                                    s2_sigma_view[n] = tot[k,l,2]
                                    n+=1

                    rho_smooth = median(s2_rho_view,0.)
                    v_smooth = median(s2_v_view,1.)
                    sigma_smooth = median(s2_sigma_view,0.)
                
                elif s == 3.0:
                    n = 0
                    for k in range(i-2,i+3):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s3_rho_view[n] = 0.
                            s3_v_view[n] = 0.
                            s3_sigma_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-2,j+3):
                                if l < 0 or l>J:#check bordo
                                    s3_rho_view[n] = 0.
                                    s3_v_view[n] = 0.
                                    s3_sigma_view[n] = 0.
                                    n+=1
                                else:
                                    s3_rho_view[n] = tot[k,l,0]
                                    s3_v_view[n] = tot[k,l,1]
                                    s3_sigma_view[n] = tot[k,l,2]
                                    n+=1

                    rho_smooth = median(s3_rho_view,0.)
                    v_smooth = median(s3_v_view,1.)
                    sigma_smooth = median(s3_sigma_view,0.)

                    
                elif s == 4.0:
                    n = 0
                    for k in range(i-3,i+4):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s4_rho_view[n] = 0.
                            s4_v_view[n] = 0.
                            s4_sigma_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-3,j+4):
                                if l < 0 or l>J:#check bordo
                                    s4_rho_view[n] = 0.
                                    s4_v_view[n] = 0.
                                    s4_sigma_view[n] = 0.
                                    n+=1
                                else:
                                    s4_rho_view[n] = tot[k,l,0]
                                    s4_v_view[n] = tot[k,l,1]
                                    s4_sigma_view[n] = tot[k,l,2]
                                    n+=1

                    rho_smooth = median(s4_rho_view,0.)
                    v_smooth = median(s4_v_view,1.)
                    sigma_smooth = median(s4_sigma_view,0.)
                
                elif s == 5.0:
                    n = 0
                    for k in range(i-4,i+5):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s5_rho_view[n] = 0.
                            s5_v_view[n] = 0.
                            s5_sigma_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-4,j+5):
                                if l < 0 or l>J:#check bordo
                                    s5_rho_view[n] = 0.
                                    s5_v_view[n] = 0.
                                    s5_sigma_view[n] = 0.
                                    n+=1
                                else:
                                    s5_rho_view[n] = tot[k,l,0]
                                    s5_v_view[n] = tot[k,l,1]
                                    s5_sigma_view[n] = tot[k,l,2]
                                    n+=1

                    rho_smooth = median(s5_rho_view,0.)
                    v_smooth = median(s5_v_view,1.)
                    sigma_smooth = median(s5_sigma_view,0.)
                
                sigma_tmp_data = ydata[i,j,2]
                if sigma_tmp_data > 120.:
                    likelihood += lk(log10(rho_smooth),rho_tmp_data,yerr[i,j,0]) + lk(v_smooth,ydata[i,j,1],yerr[i,j,1]) + lk(sigma_smooth,sigma_tmp_data,yerr[i,j,2])
                else:
                    likelihood += lk(log10(rho_smooth),rho_tmp_data,yerr[i,j,0]) + lk(v_smooth,ydata[i,j,1],yerr[i,j,1]) 
            else:
                pass
    return likelihood


#---------------------------------------------------------------------------
                    #ONLY VELOCITY
#---------------------------------------------------------------------------
##                    ##
##   ONLY VELOCITY    ##
##                    ##
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double likelihood_only_v (double[:,:,::1] x,#x position refined grid è 99x99x100
                    double[:,:,::1] y,#y position refined grid
                    double Mb,        #bulge mass log
                    double Rb,        #bulge radius
                    double Md,        #disc mass log
                    double Rd,        #disc radius
                    double Mh,        #halo mass log
                    double Rh,        #halo radius
                    double xcm,       #center x position
                    double ycm,       #center y position
                    double theta,     #P.A. angle (rotation in sky-plane)
                    double incl,      #inclination angle
                    double [:,:,::1]ydata,     #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                    double [:,:,::1]yerr):     #err
    
    cdef double x0 = 0.               #x-xcm  traslation
    cdef double y0 = 0.               #y-ycm  traslation
    cdef double xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef double yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef double yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef double r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef double r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef double c_theta = cos(theta)  #cos(theta) calculated once
    cdef double s_theta = sin(theta)  #sin(theta) calculated once
    cdef double c_incl = cos(incl)    #cos(incl) calculated once
    cdef double s_incl = sin(incl)    #cos(incl) calculated once
    cdef double phi = 0.
    cdef Py_ssize_t i,j,k
    #così ciclo sui dati non sulla griglia
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = y.shape[1]
    cdef Py_ssize_t K = 100

    #define HERQUINST STRUCTURE
    cdef HERQUINST herquinst
    #   RHO DEFINITIONS
    cdef double rhoB = 0.
    cdef double rhoD = 0.
    cdef double sum_rhoD = 0.         #mi serve per mediare la vel
    cdef double rho_tmp = 0.
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    cdef np.ndarray[double,ndim=3,mode="c"] all_rhoD = np.zeros((N,J,K))
    cdef double[:,:,::1] all_rhoD_view = all_rhoD
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    cdef np.ndarray[double,ndim=3,mode = "c"] rho_dot_sigma2 = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_dot_sigma2_view = rho_dot_sigma2
    
    #V DEFINITIONS
    cdef double v_tmp = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
    cdef np.ndarray[double,ndim=3,mode = "c"] all_v = np.zeros((N,J,K))
    cdef double[:,:,::1] all_v_view = all_v

    #SIGMA DEFINITIONS
    cdef double sigma_tmp = 0.

    #si può forse pensare ad un altro moto di salvare le quantità che calcolo
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N,J,2))
    cdef double[:,:,::1] tot_view = tot 
    
    #LIKELIHOOD DEFINITIONS
    cdef double likelihood = 0.
    #masse passate in logaritmo
    Mb = 10**Mb #is it ok or better import pow from math?
    Md = 10**Md
    Mh = 10**Mh
    Rb = 10**Rb
    Rd = 10**Rd 
    Rh = 10**Rh

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    for i in range(N):
        for j in range(J):
            sum_rhoD = 0.
            #solo dove ho dati faccio i conti
            if 10**ydata[i,j,0] > 0.:

                for k in range(K):
                    #queste operazioni è meglio farle in una funzione?
                    #per il fatto che sarebbe cdef e nogil...
                    x0 = x[i,j,k]-xcm
                    y0 = y[i,j,k]-ycm
                    xr = x0*c_theta + y0*s_theta 
                    yr = y0*c_theta - x0*s_theta
                    yd = yr/c_incl
                    r_proj = radius(x0,y0) 
                    r_true = radius(xr,yd)
                    phi = atan2(yd,xr)
                    
                    herquinst = herquinst_function(Mb,Rb,r_proj)
                    rhoB = herquinst.rho 
                    rhoD = rho_D(Md,Rd,r_true,c_incl)
                    all_rhoD_view[i,j,k] = rhoD
                    sum_rhoD = sum_rhoD + rhoD
                    tot_view[i,j,0] += rho_tot(rhoB,rhoD)

                    v_tmp = v_tot(Mb,Rb,Md,Rd,Mh,Rh,s_incl,r_true,phi)
                    all_v_view[i,j,k] = v_tmp
                    tot_view[i,j,1] += rhoD*v_tmp

                    #questo è rho_H*sigma_H2 andrà sommato nel prossimo ciclo 
                    #con la dispersione del disco che devo ancora calcolare
                    rho_dot_sigma2_view[i,j,k] = rhoB*herquinst.sigma
            
            
                #le sommo finito il ciclo divido per 100, per fare la media
                tot_view[i,j,0] = tot_view[i,j,0]/100.0
                if sum_rhoD == 0. :
                    tot_view[i,j,1] = 0.
                else:
                    tot_view[i,j,1] = tot_view[i,j,1]/sum_rhoD 

                
                likelihood += lk(tot_view[i,j,1],ydata[i,j,1],yerr[i,j,1]) 
            
            else:
                pass

    #si può ragionare su come ritornare direttamente un vettore
    #in questo modo dall'altra parte evito di fare .ravel
    #print(likelihood)

    return likelihood

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double likelihood_only_v_smooth (double[:,:,::1] x,#x position refined grid è 99x99x100
                    double[:,:,::1] y,#y position refined grid
                    double Mb,        #bulge mass log
                    double Rb,        #bulge radius
                    double Md,        #disc mass log
                    double Rd,        #disc radius
                    double Mh,        #halo mass log
                    double Rh,        #halo radius
                    double xcm,       #center x position
                    double ycm,       #center y position
                    double theta,     #P.A. angle (rotation in sky-plane)
                    double incl,      #inclination angle
                    double [:,:,::1]ydata,     #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                    double [:,:,::1]yerr,     #err
                    double [:,:,::1] psf_weight,  #weight of the PSF
                    double [:,::1] smoothing):  #smoothing matrix
    
    cdef double x0 = 0.               #x-xcm  traslation
    cdef double y0 = 0.               #y-ycm  traslation
    cdef double xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef double yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef double yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef double r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef double r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef double c_theta = cos(theta)  #cos(theta) calculated once
    cdef double s_theta = sin(theta)  #sin(theta) calculated once
    cdef double c_incl = cos(incl)    #cos(incl) calculated once
    cdef double s_incl = sin(incl)    #cos(incl) calculated once
    cdef double phi = 0.
    cdef Py_ssize_t i,j,k
    #così ciclo sui dati non sulla griglia
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = y.shape[1]
    cdef Py_ssize_t K = 100

    #define HERQUINST STRUCTURE
    cdef HERQUINST herquinst
    #   RHO DEFINITIONS
    cdef double rhoB = 0.
    cdef double rhoD = 0.
    cdef double sum_rhoD = 0.         #mi serve per mediare la vel
    cdef double rho_tmp = 0.
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    cdef np.ndarray[double,ndim=3,mode="c"] all_rhoD = np.zeros((N,J,K))
    cdef double[:,:,::1] all_rhoD_view = all_rhoD
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    cdef np.ndarray[double,ndim=3,mode = "c"] rho_dot_sigma2 = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_dot_sigma2_view = rho_dot_sigma2
    
    #V DEFINITIONS
    cdef double v_tmp = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
    cdef np.ndarray[double,ndim=3,mode = "c"] all_v = np.zeros((N,J,K))
    cdef double[:,:,::1] all_v_view = all_v

    #SIGMA DEFINITIONS
    cdef double sigma_tmp = 0.

    #si può forse pensare ad un altro moto di salvare le quantità che calcolo
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N,J,2))
    cdef double[:,:,::1] tot_view = tot 
    
    cdef double s = 0.     #smoothing
    cdef double rho_tmp_data = 0.
    cdef double sigma_tmp_data = 0.
    cdef double rho_partial_tmp = 0.

    #smoothed quantities def
    cdef double rho_smooth = 0.
    cdef double v_smooth = 0.
    cdef double sigma_smooth = 0.

    #partial quantities case s = 2
    cdef np.ndarray[double,ndim=1,mode='c'] s2_v = np.zeros((9)) 
    cdef double[::1] s2_v_view = s2_v 

    #partial quantities case s = 3
    cdef np.ndarray[double,ndim=1,mode='c'] s3_v = np.zeros((25)) 
    cdef double[::1] s3_v_view = s3_v 

    #partial quantities case s = 4
    cdef np.ndarray[double,ndim=1,mode='c'] s4_v = np.zeros((49)) 
    cdef double[::1] s4_v_view = s4_v 

    #partial quantities case s = 5
    cdef np.ndarray[double,ndim=1,mode='c'] s5_v = np.zeros((81)) 
    cdef double[::1] s5_v_view = s5_v 
    
    #PSF definitions
    cdef double psf_tmp=0.
    cdef double sum_psf = 0.

    #likelihood definitions
    cdef double likelihood = 0.
    #masse passate in logaritmo
    Mb = 10**Mb #is it ok or better import pow from math?
    Md = 10**Md
    Mh = 10**Mh
    Rb = 10**Rb
    Rd = 10**Rd 
    Rh = 10**Rh

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    for i in range(N):
        for j in range(J):
            sum_rhoD = 0.
            sum_psf = 0.
            #solo dove ho dati faccio i conti
            if 10**ydata[i,j,0] > 0.:

                for k in range(K):
                    psf_tmp = psf_weight[i,j,k]
                    #queste operazioni è meglio farle in una funzione?
                    #per il fatto che sarebbe cdef e nogil...
                    x0 = x[i,j,k]-xcm
                    y0 = y[i,j,k]-ycm
                    xr = x0*c_theta + y0*s_theta 
                    yr = y0*c_theta - x0*s_theta
                    yd = yr/c_incl
                    r_proj = radius(x0,y0) 
                    r_true = radius(xr,yd)
                    phi = atan2(yd,xr)
                    
                    herquinst = herquinst_function(Mb,Rb,r_proj)
                    rhoB = herquinst.rho 
                    rhoD = rho_D(Md,Rd,r_true,c_incl)
                    all_rhoD_view[i,j,k] = rhoD
                    sum_rhoD = sum_rhoD + rhoD*psf_tmp
                    tot_view[i,j,0] += rho_tot(rhoB,rhoD)*psf_tmp

                    v_tmp = v_tot(Mb,Rb,Md,Rd,Mh,Rh,s_incl,r_true,phi)
                    all_v_view[i,j,k] = v_tmp
                    tot_view[i,j,1] += rhoD*v_tmp*psf_tmp

                    #questo è rho_H*sigma_H2 andrà sommato nel prossimo ciclo 
                    #con la dispersione del disco che devo ancora calcolare
                    rho_dot_sigma2_view[i,j,k] = rhoB*herquinst.sigma
                    sum_psf += psf_tmp
            
                #le sommo finito il ciclo divido per 100, per fare la media
                tot_view[i,j,0] = tot_view[i,j,0]/sum_psf
                if sum_rhoD == 0. :
                    tot_view[i,j,1] = 0.
                else:
                    tot_view[i,j,1] = tot_view[i,j,1]/sum_rhoD 

                
    #ora ho calcolato tot devo fare lo smoothing dei dati
    for i in range(N):
        for j in range(J):
            rho_tmp_data = ydata[i,j,0]
            if 10**rho_tmp_data > 0.:
                s = smoothing[i,j]
                if s == 1.0 :
                    v_smooth = tot[i,j,1]
                
                elif s == 2.0:
                    n = 0
                    for k in range(i-1,i+2):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s2_v_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-1,j+2):
                                if l < 0 or l>J:#check bordo
                                    s2_v_view[n] = 0.
                                    n+=1
                                else:
                                    s2_v_view[n] = tot[k,l,1]
                                    n+=1

                    v_smooth = median(s2_v_view,1.)
                
                elif s == 3.0:
                    n = 0
                    for k in range(i-2,i+3):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s3_v_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-2,j+3):
                                if l < 0 or l>J:#check bordo
                                    s3_v_view[n] = 0.
                                    n+=1
                                else:
                                    s3_v_view[n] = tot[k,l,1]
                                    n+=1

                    v_smooth = median(s3_v_view,1.)

                    
                elif s == 4.0:
                    n = 0
                    for k in range(i-3,i+4):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s4_v_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-3,j+4):
                                if l < 0 or l>J:#check bordo
                                    s4_v_view[n] = 0.
                                    n+=1
                                else:
                                    s4_v_view[n] = tot[k,l,1]
                                    n+=1

                    v_smooth = median(s4_v_view,1.)
                
                elif s == 5.0:
                    n = 0
                    for k in range(i-4,i+5):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s5_v_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-4,j+5):
                                if l < 0 or l>J:#check bordo
                                    s5_v_view[n] = 0.
                                    n+=1
                                else:
                                    s5_v_view[n] = tot[k,l,1]
                                    n+=1

                    v_smooth = median(s5_v_view,1.)


                likelihood +=  lk(v_smooth,ydata[i,j,1],yerr[i,j,1]) 
            else:
                pass
    #si può ragionare su come ritornare direttamente un vettore
    #in questo modo dall'altra parte evito di fare .ravel
    #print(likelihood)

    return likelihood

#---------------------------------------------------------------------------
                    #NO SIGMA
#---------------------------------------------------------------------------

##                    ##
## NORMAL LK NO SIGMA ##
##                    ##
#likelihood normale senza sigma
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double likelihood_no_sigma (double[:,:,::1] x,#x position refined grid è 99x99x100
                    double[:,:,::1] y,#y position refined grid
                    double Mb,        #bulge mass log
                    double Rb,        #bulge radius
                    double Md,        #disc mass log
                    double Rd,        #disc radius
                    double Mh,        #halo mass log
                    double Rh,        #halo radius
                    double xcm,       #center x position
                    double ycm,       #center y position
                    double theta,     #P.A. angle (rotation in sky-plane)
                    double incl,      #inclination angle
                    double [:,:,::1]ydata,     #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                    double [:,:,::1]yerr):     #err
    
    cdef double x0 = 0.               #x-xcm  traslation
    cdef double y0 = 0.               #y-ycm  traslation
    cdef double xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef double yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef double yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef double r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef double r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef double c_theta = cos(theta)  #cos(theta) calculated once
    cdef double s_theta = sin(theta)  #sin(theta) calculated once
    cdef double c_incl = cos(incl)    #cos(incl) calculated once
    cdef double s_incl = sin(incl)    #cos(incl) calculated once
    cdef double phi = 0.
    cdef Py_ssize_t i,j,k
    #così ciclo sui dati non sulla griglia
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = y.shape[1]
    cdef Py_ssize_t K = 100

    #define HERQUINST STRUCTURE
    cdef HERQUINST herquinst
    #   RHO DEFINITIONS
    cdef double rhoB = 0.
    cdef double rhoD = 0.
    cdef double sum_rhoD = 0.         #mi serve per mediare la vel
    cdef double rho_tmp = 0.
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    cdef np.ndarray[double,ndim=3,mode="c"] all_rhoD = np.zeros((N,J,K))
    cdef double[:,:,::1] all_rhoD_view = all_rhoD
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    cdef np.ndarray[double,ndim=3,mode = "c"] rho_dot_sigma2 = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_dot_sigma2_view = rho_dot_sigma2
    
    #V DEFINITIONS
    cdef double v_tmp = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
    cdef np.ndarray[double,ndim=3,mode = "c"] all_v = np.zeros((N,J,K))
    cdef double[:,:,::1] all_v_view = all_v

    #SIGMA DEFINITIONS
    cdef double sigma_tmp = 0.

    #si può forse pensare ad un altro moto di salvare le quantità che calcolo
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N,J,2))
    cdef double[:,:,::1] tot_view = tot 
    
    #LIKELIHOOD DEFINITIONS
    cdef double likelihood = 0.
    #masse passate in logaritmo
    Mb = 10**Mb #is it ok or better import pow from math?
    Md = 10**Md
    Mh = 10**Mh
    Rb = 10**Rb
    Rd = 10**Rd 
    Rh = 10**Rh

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    for i in range(N):
        for j in range(J):
            sum_rhoD = 0.
            #solo dove ho dati faccio i conti
            if 10**ydata[i,j,0] > 0.:

                for k in range(K):
                    #queste operazioni è meglio farle in una funzione?
                    #per il fatto che sarebbe cdef e nogil...
                    x0 = x[i,j,k]-xcm
                    y0 = y[i,j,k]-ycm
                    xr = x0*c_theta + y0*s_theta 
                    yr = y0*c_theta - x0*s_theta
                    yd = yr/c_incl
                    r_proj = radius(x0,y0) 
                    r_true = radius(xr,yd)
                    phi = atan2(yd,xr)
                    
                    herquinst = herquinst_function(Mb,Rb,r_proj)
                    rhoB = herquinst.rho 
                    rhoD = rho_D(Md,Rd,r_true,c_incl)
                    all_rhoD_view[i,j,k] = rhoD
                    sum_rhoD = sum_rhoD + rhoD
                    tot_view[i,j,0] += rho_tot(rhoB,rhoD)

                    v_tmp = v_tot(Mb,Rb,Md,Rd,Mh,Rh,s_incl,r_true,phi)
                    all_v_view[i,j,k] = v_tmp
                    tot_view[i,j,1] += rhoD*v_tmp

                    #questo è rho_H*sigma_H2 andrà sommato nel prossimo ciclo 
                    #con la dispersione del disco che devo ancora calcolare
                    rho_dot_sigma2_view[i,j,k] = rhoB*herquinst.sigma
            
            
                #le sommo finito il ciclo divido per 100, per fare la media
                tot_view[i,j,0] = tot_view[i,j,0]/100.0
                if sum_rhoD == 0. :
                    tot_view[i,j,1] = 0.
                else:
                    tot_view[i,j,1] = tot_view[i,j,1]/sum_rhoD 

                
                likelihood += lk(log10(tot_view[i,j,0]),ydata[i,j,0],yerr[i,j,0]) + lk(tot_view[i,j,1],ydata[i,j,1],yerr[i,j,1]) 
            
            else:
                pass

    #si può ragionare su come ritornare direttamente un vettore
    #in questo modo dall'altra parte evito di fare .ravel
    #print(likelihood)

    return likelihood

##                    ##
## SMOOTH LK NO SIGMA ##
##                    ##
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double likelihood_no_sigma_smoothing (double[:,:,::1] x,          #x position refined grid è 99x99x100
                                   double[:,:,::1] y,          #y position refined grid
                                   double Mb,                  #bulge mass log
                                   double Rb,                  #bulge radius
                                   double Md,                  #disc mass log
                                   double Rd,                  #disc radius
                                   double Mh,                  #halo mass log
                                   double Rh,                  #halo radius
                                   double xcm,                 #center x position
                                   double ycm,                 #center y position
                                   double theta,               #P.A. angle (rotation in sky-plane)
                                   double incl,                #inclination angle
                                   double [:,:,::1]ydata,      #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                                   double [:,:,::1]yerr,       #err
                                   double [:,:,::1] psf_weight,  #weight of the PSF
                                   double [:,::1] smoothing):  #smoothing matrix
    
    
    cdef double x0 = 0.               #x-xcm  traslation
    cdef double y0 = 0.               #y-ycm  traslation
    cdef double xr = 0.               #x0*np.cos(theta) + y0*np.sin(theta)  rotation
    cdef double yr = 0.               #y0*np.cos(theta) - x0*np.sin(theta)  rotation
    cdef double yd = 0.               #yd = yr/np.cos(incl)                 de-projection
    cdef double r_true = 0.           #sqrt(xr**2+yd**2) real radius
    cdef double r_proj = 0.           #sqrt(x0**2+y0**2) projected radius
    cdef double c_theta = cos(theta)  #cos(theta) calculated once
    cdef double s_theta = sin(theta)  #sin(theta) calculated once
    cdef double c_incl = cos(incl)    #cos(incl) calculated once
    cdef double s_incl = sin(incl)    #cos(incl) calculated once
    cdef double phi = 0.
    cdef Py_ssize_t i,j,k,l,m,n
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = x.shape[1]
    cdef Py_ssize_t K = 100
    

    #define HERQUINST STRUCTURE
    cdef HERQUINST herquinst
    #   RHO DEFINITIONS
    cdef double rhoB = 0.
    cdef double rhoD = 0.
    cdef double sum_rhoD = 0.         #mi serve per mediare la vel
    cdef double rho_tmp = 0.
    #devo salvarmi tutte le densità del disco perchè poi le utilizzo per la sigma
    cdef np.ndarray[double,ndim=3,mode="c"] all_rhoD = np.zeros((N,J,K))
    cdef double[:,:,::1] all_rhoD_view = all_rhoD
    #questo mi serve per salvarmi tutti i rho_H*sigma_H^2 nel primo ciclo
    #cosi nel secondo ciclo non devo più fare traslazioni,rotazioni o ricalcolare la rho
    cdef np.ndarray[double,ndim=3,mode = "c"] rho_dot_sigma2 = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_dot_sigma2_view = rho_dot_sigma2
    
    #V DEFINITIONS
    cdef double v_tmp = 0.
    #questi mi servono per salvarmi tutte le velocità che utilizzerò nel calcolo della sigma
    cdef np.ndarray[double,ndim=3,mode = "c"] all_v = np.zeros((N,J,K))
    cdef double[:,:,::1] all_v_view = all_v



    cdef double s = 0.     #smoothing
    cdef np.ndarray[double,ndim=3,mode='c'] tot = np.zeros((N,J,2))
    cdef double[:,:,::1] tot_view = tot 
    cdef double rho_tmp_data = 0.
    cdef double sigma_tmp_data = 0.
    cdef double rho_partial_tmp = 0.

    #smoothed quantities def
    cdef double rho_smooth = 0.
    cdef double v_smooth = 0.
    cdef double sigma_smooth = 0.

    #partial quantities case s = 2
    cdef np.ndarray[double,ndim=1,mode='c'] s2_rho = np.zeros((9)) 
    cdef double[::1] s2_rho_view = s2_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s2_v = np.zeros((9)) 
    cdef double[::1] s2_v_view = s2_v 

    #partial quantities case s = 3
    cdef np.ndarray[double,ndim=1,mode='c'] s3_rho = np.zeros((25)) 
    cdef double[::1] s3_rho_view = s3_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s3_v = np.zeros((25)) 
    cdef double[::1] s3_v_view = s3_v 

    #partial quantities case s = 4
    cdef np.ndarray[double,ndim=1,mode='c'] s4_rho = np.zeros((49)) 
    cdef double[::1] s4_rho_view = s4_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s4_v = np.zeros((49)) 
    cdef double[::1] s4_v_view = s4_v 

    #partial quantities case s = 5
    cdef np.ndarray[double,ndim=1,mode='c'] s5_rho = np.zeros((81)) 
    cdef double[::1] s5_rho_view = s5_rho 
    cdef np.ndarray[double,ndim=1,mode='c'] s5_v = np.zeros((81)) 
    cdef double[::1] s5_v_view = s5_v 

    #likelihood definitions
    cdef double likelihood = 0.

    #PSF definitions
    cdef double psf_tmp=0.
    cdef double sum_psf = 0.

    Mb = 10**Mb #is it ok or better import pow from math?
    Md = 10**Md
    Mh = 10**Mh
    Rb = 10**Rb
    Rd = 10**Rd 
    Rh = 10**Rh

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    for i in range(N):
        for j in range(J):
            sum_rhoD = 0.
            sum_psf = 0.
            #solo dove ho dati faccio i conti
            if 10**ydata[i,j,0] > 0.:

                for k in range(K):
                    psf_tmp = psf_weight[i,j,k]
                    #queste operazioni è meglio farle in una funzione?
                    #per il fatto che sarebbe cdef e nogil...
                    x0 = x[i,j,k]-xcm
                    y0 = y[i,j,k]-ycm
                    xr = x0*c_theta + y0*s_theta 
                    yr = y0*c_theta - x0*s_theta
                    yd = yr/c_incl
                    r_proj = radius(x0,y0) 
                    r_true = radius(xr,yd)
                    phi = atan2(yd,xr)
                    
                    herquinst = herquinst_function(Mb,Rb,r_proj)
                    rhoB = herquinst.rho 
                    rhoD = rho_D(Md,Rd,r_true,c_incl)
                    all_rhoD_view[i,j,k] = rhoD
                    sum_rhoD = sum_rhoD + psf_tmp*rhoD
                    tot_view[i,j,0] += psf_tmp*rho_tot(rhoB,rhoD)

                    v_tmp = v_tot(Mb,Rb,Md,Rd,Mh,Rh,s_incl,r_true,phi)
                    all_v_view[i,j,k] = v_tmp
                    tot_view[i,j,1] += rhoD*v_tmp*psf_tmp

                    #questo è rho_H*sigma_H2 andrà sommato nel prossimo ciclo 
                    #con la dispersione del disco che devo ancora calcolare
                    rho_dot_sigma2_view[i,j,k] = rhoB*herquinst.sigma

                    sum_psf += psf_tmp
                    
                #le sommo finito il ciclo divido per 100, per fare la media
                tot_view[i,j,0] = tot_view[i,j,0]/sum_psf
                if sum_rhoD == 0. :
                    tot_view[i,j,1] = 0.
                else:
                    tot_view[i,j,1] = tot_view[i,j,1]/sum_rhoD
            else:
                pass 
    
    #ora ho calcolato tot devo fare lo smoothing dei dati
    for i in range(N):
        for j in range(J):
            rho_tmp_data = ydata[i,j,0]
            if 10**rho_tmp_data > 0.:
                s = smoothing[i,j]
                if s == 1.0 :
                    rho_smooth = tot[i,j,0]
                    v_smooth = tot[i,j,1]
                
                elif s == 2.0:
                    n = 0
                    for k in range(i-1,i+2):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s2_rho_view[n] = 0.
                            s2_v_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-1,j+2):
                                if l < 0 or l>J:#check bordo
                                    s2_rho_view[n] = 0.
                                    s2_v_view[n] = 0.
                                    n+=1
                                else:
                                    s2_rho_view[n] = tot[k,l,0]
                                    s2_v_view[n] = tot[k,l,1]
                                    n+=1

                    rho_smooth = median(s2_rho_view,0.)
                    v_smooth = median(s2_v_view,1.)
                
                elif s == 3.0:
                    n = 0
                    for k in range(i-2,i+3):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s3_rho_view[n] = 0.
                            s3_v_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-2,j+3):
                                if l < 0 or l>J:#check bordo
                                    s3_rho_view[n] = 0.
                                    s3_v_view[n] = 0.
                                    n+=1
                                else:
                                    s3_rho_view[n] = tot[k,l,0]
                                    s3_v_view[n] = tot[k,l,1]
                                    n+=1

                    rho_smooth = median(s3_rho_view,0.)
                    v_smooth = median(s3_v_view,1.)

                    
                elif s == 4.0:
                    n = 0
                    for k in range(i-3,i+4):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s4_rho_view[n] = 0.
                            s4_v_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-3,j+4):
                                if l < 0 or l>J:#check bordo
                                    s4_rho_view[n] = 0.
                                    s4_v_view[n] = 0.
                                    n+=1
                                else:
                                    s4_rho_view[n] = tot[k,l,0]
                                    s4_v_view[n] = tot[k,l,1]
                                    n+=1

                    rho_smooth = median(s4_rho_view,0.)
                    v_smooth = median(s4_v_view,1.)
                
                elif s == 5.0:
                    n = 0
                    for k in range(i-4,i+5):
                        if k < 0 or k>N:#check che non sia al bordo della griglia
                            s5_rho_view[n] = 0.
                            s5_v_view[n] = 0.
                            n+=1
                        else:
                            for l in range(j-4,j+5):
                                if l < 0 or l>J:#check bordo
                                    s5_rho_view[n] = 0.
                                    s5_v_view[n] = 0.
                                    n+=1
                                else:
                                    s5_rho_view[n] = tot[k,l,0]
                                    s5_v_view[n] = tot[k,l,1]
                                    n+=1

                    rho_smooth = median(s5_rho_view,0.)
                    v_smooth = median(s5_v_view,1.)

                likelihood += lk(log10(rho_smooth),rho_tmp_data,yerr[i,j,0]) + lk(v_smooth,ydata[i,j,1],yerr[i,j,1]) 
            else:
                pass

    #print(likelihood)
    return likelihood


#---------------------------------------------------------------------------
                    #LIKELIHOOD PSO
#---------------------------------------------------------------------------
#likelihood per PSO perchè deve iterare sulle particelle
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=1,mode='c'] likelihood_pso(double[:,::1]par,  #all params in format [swarm_particles, number of params]
                                                                          #par[:,0] logMb;par[:,1] logRb;par[:,2] logMd;par[:,3] logRd;par[:,4] logMh;par[:,5] logRh;par[:,6] xcm;par[:,7] ycm;par[:,8] theta;par[:,9] incl; 
                                                        double[:,:,::1] x,#x position refined grid è 99x99x100
                                                        double[:,:,::1] y,#y position refined grid
                                                        double [:,:,::1]ydata,     #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                                                        double [:,:,::1]yerr):     #err
    cdef Py_ssize_t N_swarm = par.shape[0]
    cdef Py_ssize_t s 
    cdef np.ndarray[double,ndim=1,mode = "c"] lk_pso = np.zeros((N_swarm))
    cdef double[::1] lk_pso_view = lk_pso
    cdef double Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl = 0.0



    for s in range(N_swarm):
        Mb = par[s,0]
        Rb = par[s,1]
        Md = par[s,2]
        Rd = par[s,3]
        Mh = par[s,4]
        Rh = par[s,5]
        xcm = par[s,6]
        ycm = par[s,7]
        theta = par[s,8]
        incl = par[s,9]
        
        lk_pso_view[s] = -likelihood(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,ydata,yerr)


    return lk_pso 



##                    ##
##     PSO NO SIGMA   ##
##                    ##
#likelihhod per pso senza sigma
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=1,mode='c'] likelihood_pso_no_sigma(double[:,::1]par,  #all params in format [swarm_particles, number of params]
                                                                          #par[:,0] logMb;par[:,1] logRb;par[:,2] logMd;par[:,3] logRd;par[:,4] logMh;par[:,5] logRh;par[:,6] xcm;par[:,7] ycm;par[:,8] theta;par[:,9] incl; 
                                                                double[:,:,::1] x,#x position refined grid è 99x99x100
                                                                double[:,:,::1] y,#y position refined grid
                                                                double [:,:,::1]ydata,     #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                                                                double [:,:,::1]yerr):     #err
    cdef Py_ssize_t N_swarm = par.shape[0]
    cdef Py_ssize_t s 
    cdef np.ndarray[double,ndim=1,mode = "c"] lk_pso = np.zeros((N_swarm))
    cdef double[::1] lk_pso_view = lk_pso
    cdef double Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl = 0.0



    for s in range(N_swarm):
        Mb = par[s,0]
        Rb = par[s,1]
        Md = par[s,2]
        Rd = par[s,3]
        Mh = par[s,4]
        Rh = par[s,5]
        xcm = par[s,6]
        ycm = par[s,7]
        theta = par[s,8]
        incl = par[s,9]
        
        lk_pso_view[s] = -likelihood_no_sigma(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,ydata,yerr)


    return lk_pso 

##                           ##
## PSO NO SIGMA + SMOOTHING  ##
##                           ##
#likelihhod per pso senza sigma
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=1,mode='c'] likelihood_pso_no_sigma_smooth(double[:,::1]par,  #all params in format [swarm_particles, number of params]
                                                                                            #par[:,0] logMb;par[:,1] logRb;par[:,2] logMd;par[:,3] logRd;par[:,4] logMh;par[:,5] logRh;par[:,6] xcm;par[:,7] ycm;par[:,8] theta;par[:,9] incl; 
                                                                double[:,:,::1] x,#x position refined grid è 99x99x100
                                                                double[:,:,::1] y,#y position refined grid
                                                                double [:,:,::1]ydata,     #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                                                                double [:,:,::1]yerr,     #err
                                                                double [:,:,::1]psf_weight,
                                                                double [:,::1]smoothing):
    cdef Py_ssize_t N_swarm = par.shape[0]
    cdef Py_ssize_t s 
    cdef np.ndarray[double,ndim=1,mode = "c"] lk_pso = np.zeros((N_swarm))
    cdef double[::1] lk_pso_view = lk_pso
    cdef double Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl = 0.0



    for s in range(N_swarm):
        Mb = par[s,0]
        Rb = par[s,1]
        Md = par[s,2]
        Rd = par[s,3]
        Mh = par[s,4]
        Rh = par[s,5]
        xcm = par[s,6]
        ycm = par[s,7]
        theta = par[s,8]
        incl = par[s,9]
        
        lk_pso_view[s] = -likelihood_no_sigma_smoothing(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,ydata,yerr,psf_weight,smoothing)


    return lk_pso 


##                           ##
## PSO ONLY V   + SMOOTHING  ##
##                           ##
#likelihhod per pso senza sigma
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=1,mode='c'] likelihood_pso_only_v_smooth(double[:,::1]par,  #all params in format [swarm_particles, number of params]
                                                                                            #par[:,0] logMb;par[:,1] logRb;par[:,2] logMd;par[:,3] logRd;par[:,4] logMh;par[:,5] logRh;par[:,6] xcm;par[:,7] ycm;par[:,8] theta;par[:,9] incl; 
                                                                double[:,:,::1] x,#x position refined grid è 99x99x100
                                                                double[:,:,::1] y,#y position refined grid
                                                                double [:,:,::1]ydata,     #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                                                                double [:,:,::1]yerr,     #err
                                                                double [:,:,::1]psf_weight,
                                                                double [:,::1]smoothing):
    cdef Py_ssize_t N_swarm = par.shape[0]
    cdef Py_ssize_t s 
    cdef np.ndarray[double,ndim=1,mode = "c"] lk_pso = np.zeros((N_swarm))
    cdef double[::1] lk_pso_view = lk_pso
    cdef double Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl = 0.0



    for s in range(N_swarm):
        Mb = par[s,0]
        Rb = par[s,1]
        Md = par[s,2]
        Rd = par[s,3]
        Mh = par[s,4]
        Rh = par[s,5]
        xcm = par[s,6]
        ycm = par[s,7]
        theta = par[s,8]
        incl = par[s,9]
        
        lk_pso_view[s] = -likelihood_only_v_smooth(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,ydata,yerr,psf_weight,smoothing)


    return lk_pso 

##                    ##
##     PSO SMOOTHING  ##
##                    ##
#likelihood per pso con smoothing
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=1,mode='c'] likelihood_pso_smoothing(double[:,::1]par,  #all params in format [swarm_particles, number of params]
                                                                          #par[:,0] logMb;par[:,1] logRb;par[:,2] logMd;par[:,3] logRd;par[:,4] logMh;par[:,5] logRh;par[:,6] xcm;par[:,7] ycm;par[:,8] theta;par[:,9] incl; 
                                                                double[:,:,::1] x,#x position refined grid è 99x99x100
                                                                double[:,:,::1] y,#y position refined grid
                                                                double [:,:,::1]ydata,     #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                                                                double [:,:,::1]yerr,     #err
                                                                double [:,:,::1]psf_weight,
                                                                double [:,::1]smoothing):
    cdef Py_ssize_t N_swarm = par.shape[0]
    cdef Py_ssize_t s 
    cdef np.ndarray[double,ndim=1,mode = "c"] lk_pso = np.zeros((N_swarm))
    cdef double[::1] lk_pso_view = lk_pso
    cdef double Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl = 0.0



    for s in range(N_swarm):
        Mb = par[s,0]
        Rb = par[s,1]
        Md = par[s,2]
        Rd = par[s,3]
        Mh = par[s,4]
        Rh = par[s,5]
        xcm = par[s,6]
        ycm = par[s,7]
        theta = par[s,8]
        incl = par[s,9]
 
        lk_pso_view[s] = -likelihood_smoothing(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,ydata,yerr,psf_weight,smoothing)

    return lk_pso 


    ##                           ##
## PSO NO SIGMA + SMOOTHING  + mod_vel##
##                           ##
#likelihhod per pso senza sigma
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double,ndim=1,mode='c'] likelihood_pso_no_sigma_smooth_mod_vel(double[:,::1]par,  #all params in format [swarm_particles, number of params]
                                                                                            #par[:,0] logMb;par[:,1] logRb;par[:,2] logMd;par[:,3] logRd;par[:,4] logMh;par[:,5] logRh;par[:,6] xcm;par[:,7] ycm;par[:,8] theta;par[:,9] incl; 
                                                                double[:,:,::1] x,#x position refined grid è 99x99x100
                                                                double[:,:,::1] y,#y position refined grid
                                                                double [:,:,::1]ydata,     #dati per ora della forma (N-1)x(J-1)x3: 3 sono rho,v,sigma, N-1 e J-1 è questo perché sono quelli con cose diverse da zero
                                                                double [:,:,::1]yerr,     #err
                                                                double [:,:,::1]psf_weight,
                                                                double [:,::1]smoothing):
    cdef Py_ssize_t N_swarm = par.shape[0]
    cdef Py_ssize_t s 
    cdef np.ndarray[double,ndim=1,mode = "c"] lk_pso = np.zeros((N_swarm))
    cdef double[::1] lk_pso_view = lk_pso
    cdef double Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl = 0.0



    for s in range(N_swarm):
        Mb = par[s,0]
        Rb = par[s,1]
        Md = par[s,2]
        Rd = par[s,3]
        Mh = par[s,4]
        Rh = par[s,5]
        xcm = par[s,6]
        ycm = par[s,7]
        theta = par[s,8]
        incl = par[s,9]
        
        lk_pso_view[s] = -likelihood_no_sigma_smoothing(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,ydata,yerr,psf_weight,smoothing)


    return lk_pso 