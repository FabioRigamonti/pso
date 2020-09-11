import numpy as np 
import pyswarms as ps 
from utils import likelihood_pso,likelihood_pso_smoothing,model,model_smoothing,likelihood_pso_no_sigma_smooth,likelihood_pso_only_v_smooth
import matplotlib.pyplot as plt 
import data as d
import time



def PSF(r,            #raggio proiettato
        sigma=0.27046 #sigma PSF per califa 2.5''
        ):
        r2 = np.square(r)
        sigma2 = np.square(sigma)
        
        #psf = np.exp(-r2/(2*sigma2)) / (np.sqrt(2*np.pi) * sigma2)
        psf = np.exp(-r2/(2*sigma2)) 

        return psf


if __name__ == "__main__":
    
    x, y,rho,v_los,sigma_los, error_rho,error_v_los,error_sigma_los,smoothing = np.loadtxt('data_constant_error_smoothing_corretto.txt',usecols=(0,1,2,3,4,5,6,7,8),unpack = True)


    J = np.size(x[ x == x[0] ])
    N = np.size(y[ y == y[0] ])

    x_true = x[0 : J*N : J]
    y_true = y[0 : J]

    #ydata = np.zeros((N,J,3))
    #yerr = np.zeros((N,J,3))
    #ydata[:,:,0] = rho.reshape(N,J)
    #ydata[:,:,1] = v_los.reshape(N,J)
    #ydata[:,:,2] = sigma_los.reshape(N,J)
    #yerr[:,:,0] = error_rho.reshape(N,J)
    #yerr[:,:,1] = error_v_los.reshape(N,J)
    #yerr[:,:,2] = error_sigma_los.reshape(N,J)

    smoothing = smoothing.reshape(N,J)
    smoothing = smoothing.copy(order='C')
    #data without sigma
    ydata = np.zeros((N,J,2))
    yerr = np.zeros((N,J,2))
    ydata[:,:,0] = rho.reshape(N,J)
    ydata[:,:,1] = v_los.reshape(N,J)
    yerr[:,:,0] = error_rho.reshape(N,J)
    yerr[:,:,1] = error_v_los.reshape(N,J)



    x_rand = np.zeros((N,J,100))
    y_rand = np.zeros((N,J,100))

    #define data object 
    data = d.data(x,y,rho,v_los,sigma_los,0,0,0,0,0,0,0)

    #calculate 3 dimensional grid remember that the grid is [N-1,J-1,100]
    x_rand,y_rand = data.refined_grid()

    repeated_x = np.repeat(x_rand[:,:,0],100).reshape(N-1,J-1,100)
    repeated_y = np.repeat(y_rand[:,:,0],100).reshape(N-1,J-1,100)

    r = np.sqrt(np.square(repeated_x - x_rand) + np.square(repeated_y - y_rand))
    psf_weight = PSF(r)
    
    n_iter = 4000
    n_particles = 100 

    
    min_bounds = np.array([8.5,-1,8.5,-1,10,0.6,-2,-2,0.4,0.2])
    max_bounds = np.array([11.5,0,11.5,1,13,1.6,1,1,1.0,1.2])
    bounds = (min_bounds,max_bounds)
    #w peso in direzione iniziale
    #c1 peso in direzione ottimale del minimo singolo
    #c2 peso in direzione ottimale del minimo totale
    options = {'c1': 1.5, 'c2': 1.5, 'w':0.73}
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=10, options=options,bounds=bounds) 
    #cost, pos = optimizer.optimize(likelihood_pso_smoothing, iters=n_iter,x=x_rand,y=y_rand,ydata=ydata,yerr=yerr,psf_weight=psf_weight,smoothing=smoothing,n_processes=4)  
    cost, pos = optimizer.optimize(likelihood_pso_no_sigma_smooth, iters=n_iter,x=x_rand,y=y_rand,ydata=ydata,yerr=yerr,psf_weight=psf_weight,smoothing=smoothing,n_processes=1)  

    #options = {'c1': 0.8, 'c2': 2, 'w':0.4}
    #pos_i = np.array([pos for i in range(10)])
    #pos_i = np.random.normal(pos_i,scale=0.1)
    #optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=10, options=options,bounds=bounds,init_pos=pos_i) 
    #cost, pos = optimizer.optimize(likelihood_pso, iters=700,x=x_rand,y=y_rand,ydata=ydata,yerr=yerr,n_processes=4)  
    
    #lista di array
    #pos_history[0] è la prima iterazione
    #pos_history[0][0] è la prima iterazione della prima particella
    #pos_history[0][0,0] è il primo parametro della prima particella nella prima iterazione
    pos_history = optimizer.pos_history
    lk_history = optimizer.cost_history

    #minima likelihood
    min_lk = lk_history[-1]

    #valuto la likelihood all'ultimo step di tutte le particelle
    par_last_step = pos_history[-1]     #questo ha [n_part,n_params]
    #last_lk = likelihood_pso(par_last_step,x_rand,y_rand,ydata,yerr) 
    last_lk = likelihood_pso_smoothing(par_last_step,x_rand,y_rand,ydata,yerr,psf_weight,smoothing) 
    
    min_part = np.argmin(last_lk) #è la particella migliore
    max_part = np.argmax(last_lk) #è la particella peggiore

    #come diagnostico posso fare cosi:
    #uno valuto la likelihood finale di tutte le particelle e quelli li plotto tutti
    #poi plotto i parametri di quello più vicino all'ultima lk e quello più lontano 
    logMb_best = np.zeros(n_iter)
    logRb_best = np.zeros(n_iter)
    logMd_best = np.zeros(n_iter)
    logRd_best = np.zeros(n_iter)
    logMh_best = np.zeros(n_iter)
    logRh_best = np.zeros(n_iter)
    xcm_best = np.zeros(n_iter)
    ycm_best = np.zeros(n_iter)
    theta_best = np.zeros(n_iter)
    incl_best = np.zeros(n_iter)
    
    logMb_worst = np.zeros(n_iter)
    logRb_worst = np.zeros(n_iter)
    logMd_worst = np.zeros(n_iter)
    logRd_worst = np.zeros(n_iter)
    logMh_worst = np.zeros(n_iter)
    logRh_worst = np.zeros(n_iter)
    xcm_worst = np.zeros(n_iter)
    ycm_worst = np.zeros(n_iter)
    theta_worst = np.zeros(n_iter)
    incl_worst = np.zeros(n_iter)

    for i in range(n_iter):
        logMb_best[i] = pos_history[i][min_part][0]
        logRb_best[i] = pos_history[i][min_part][1]
        logMd_best[i] = pos_history[i][min_part][2]
        logRd_best[i] = pos_history[i][min_part][3]
        logMh_best[i] = pos_history[i][min_part][4]
        logRh_best[i] = pos_history[i][min_part][5]
        xcm_best[i] = pos_history[i][min_part][6]
        ycm_best[i] = pos_history[i][min_part][7]
        theta_best[i] = pos_history[i][min_part][8]
        incl_best[i] = pos_history[i][min_part][9]

        logMb_worst[i] = pos_history[i][max_part][0]
        logRb_worst[i] = pos_history[i][max_part][1]
        logMd_worst[i] = pos_history[i][max_part][2]
        logRd_worst[i] = pos_history[i][max_part][3]
        logMh_worst[i] = pos_history[i][max_part][4]
        logRh_worst[i] = pos_history[i][max_part][5]
        xcm_worst[i] = pos_history[i][max_part][6]
        ycm_worst[i] = pos_history[i][max_part][7]
        theta_worst[i] = pos_history[i][max_part][8]
        incl_worst[i] = pos_history[i][max_part][9]



    iteration = np.linspace(0,n_iter-1,n_iter)

    fig = plt.figure()
    ax1 = fig.add_subplot(341)
    ax2 = fig.add_subplot(342)
    ax3 = fig.add_subplot(343)
    ax4 = fig.add_subplot(344)
    ax5 = fig.add_subplot(345)
    ax6 = fig.add_subplot(346)
    ax7 = fig.add_subplot(347)
    ax8 = fig.add_subplot(348)
    ax9 = fig.add_subplot(349)
    ax10 = fig.add_subplot(3,4,10)
    ax11 = fig.add_subplot(3,4,11)

    ax1.plot(iteration,logMb_best,'b',lw = 0.8,label = 'best')
    ax1.plot(iteration,logMb_worst,'r',lw = 0.8,label = 'worst')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel(r'$log_{10}(M_b)$')
    ax1.legend(loc='best')

    ax2.plot(iteration,logRb_best,'b',lw = 0.8,label = 'best')
    ax2.plot(iteration,logRb_worst,'r',lw = 0.8,label = 'worst')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel(r'$log_{10}(R_b)$')

    ax3.plot(iteration,logMd_best,'b',lw = 0.8,label = 'best')
    ax3.plot(iteration,logMd_worst,'r',lw = 0.8,label = 'worst')
    ax3.set_xlabel('iteration')
    ax3.set_ylabel(r'$log_{10}(M_d)$')

    ax4.plot(iteration,logRd_best,'b',lw = 0.8,label = 'best')
    ax4.plot(iteration,logRd_worst,'r',lw = 0.8,label = 'worst')
    ax4.set_xlabel('iteration')
    ax4.set_ylabel(r'$log_{10}(R_d)$')

    ax5.plot(iteration,logMh_best,'b',lw = 0.8,label = 'best')
    ax5.plot(iteration,logMh_worst,'r',lw = 0.8,label = 'worst')
    ax5.set_xlabel('iteration')
    ax5.set_ylabel(r'$log_{10}(M_h)$')

    ax6.plot(iteration,logRh_best,'b',lw = 0.8,label = 'best')
    ax6.plot(iteration,logRh_worst,'r',lw = 0.8,label = 'worst')
    ax6.set_xlabel('iteration')
    ax6.set_ylabel(r'$log_{10}(R_h)$')

    ax7.plot(iteration,xcm_best,'b',lw = 0.8,label = 'best')
    ax7.plot(iteration,xcm_worst,'r',lw = 0.8,label = 'worst')
    ax7.set_xlabel('iteration')
    ax7.set_ylabel(r'$x_{cm}$')

    ax8.plot(iteration,ycm_best,'b',lw = 0.8,label = 'best')
    ax8.plot(iteration,ycm_worst,'r',lw = 0.8,label = 'worst')
    ax8.set_xlabel('iteration')
    ax8.set_ylabel(r'$y_{cm}$')

    ax9.plot(iteration,theta_best,'b',lw = 0.8,label = 'best')
    ax9.plot(iteration,theta_worst,'r',lw = 0.8,label = 'worst')
    ax9.set_xlabel('iteration')
    ax9.set_ylabel('P.A.')

    ax10.plot(iteration,incl_best,'b',lw = 0.8,label = 'best')
    ax10.plot(iteration,incl_worst,'r',lw = 0.8,label = 'worst')
    ax10.set_xlabel('iteration')
    ax10.set_ylabel('i')


    ax11.plot(iteration,lk_history,'b',lw = 0.8)
    ax11.plot(np.ones(n_particles)*iteration[-1],last_lk,'*k',ms = 1.2)
    ax11.set_xlabel('iteration')
    ax11.set_ylabel('log(L)')

    fig.show()

    plt.savefig('./iter.png')

    Mb_fit = pos[0]
    Rb_fit = pos[1]
    Md_fit = pos[2]
    Rd_fit = pos[3]
    Mh_fit = pos[4]
    Rh_fit = pos[5]
    xcm_fit = pos[6]
    ycm_fit = pos[7]
    theta_fit = pos[8]
    incl_fit = pos[9]  
    
    #Mb_fit= 	10.956
    #Rb_fit= 	-0.937  
    #Md_fit= 	8.986   
    #Rd_fit= 	-0.894  
    #Mh_fit= 	12.798  
    #Rh_fit= 	1.085   
    #xcm_fit= 	-1.057  
    #ycm_fit= 	-0.726  
    #theta_fit= 	0.786   
    #incl_fit= 	0.251   
    
    print('Valori al massimo della likelihood: \n')
    print('\t fit')
    print('Mb: \t{:.3f}\n'.format(Mb_fit))
    print('Rb: \t{:.3f}\n'.format(Rb_fit))
    print('Md: \t{:.3f}\n'.format(Md_fit))
    print('Rd: \t{:.3f}\n'.format(Rd_fit))
    print('Mh: \t{:.3f}\n'.format(Mh_fit))
    print('Rh: \t{:.3f}\n'.format(Rh_fit))
    print('xcm: \t{:.3f}\n'.format(xcm_fit))
    print('ycm: \t{:.3f}\n'.format(ycm_fit))
    print('P.A.: \t{:.3f}\n'.format(theta_fit))
    print('i: \t{:.3f}\n'.format(incl_fit))



    #plot eventuale
    ti = time.time()
    #tot = model(x_rand,y_rand,Mb_fit,Rb_fit,Md_fit,Rd_fit,Mh_fit,Rh_fit,xcm_fit,ycm_fit,theta_fit,incl_fit)
    tot = model_smoothing(x_rand,y_rand,Mb_fit,Rb_fit,Md_fit,Rd_fit,Mh_fit,Rh_fit,xcm_fit,ycm_fit,theta_fit,incl_fit,psf_weight,smoothing)
    print('con smoothing ci mette: {}'.format(time.time()-ti))
    rho_model=tot[:,:,0].ravel()
    v_model=tot[:,:,1].ravel()
    sigma_model=tot[:,:,2].ravel()
    
    mydata1 =d.data(x,y,10**rho,np.copy(v_los),np.copy(sigma_los),0.,0.,0.,0.,0.,0.,0.)

    fig6,ax6,pmesh6,cbar6 = mydata1.surface_density()
    ax6.set_title('data')
    limrho = pmesh6.get_clim()
    fig6.show()

    fig4,ax4,pmesh4,cbar4 = mydata1.velocity_map()
    ax4.set_title('data')
    limv = pmesh4.get_clim()
    fig4.show()

    fig5,ax5,pmesh5,cbar5 = mydata1.dispersion_map()
    ax5.set_title('data')
    limsigma = pmesh5.get_clim()
    fig5.show()

    mydata2 =d.data(x,y,np.copy(rho_model),np.copy(v_model),np.copy(sigma_model),0.,0.,0.,0.,0.,0.,0.)

    fig52,ax52,pmesh52,cbar52 = mydata2.surface_density()
    ax52.set_title('model')
    pmesh52.set_clim(limrho)
    cbar52.ax.set_ylabel(r' $log_{10}(\Sigma)$ [$M_{\odot}/Kpc^2$]')
    fig52.show()

    fig53,ax53,pmesh53,cbar53 = mydata2.velocity_map()
    ax53.set_title('model')
    pmesh53.set_clim(limv)
    cbar53.ax.set_ylabel(r'$v_{los}$ $[Km/s]$')
    fig53.show()

    fig54,ax54,pmesh54,cbar54 = mydata2.dispersion_map()
    ax54.set_title('model')
    pmesh54.set_clim(limsigma)
    cbar54.ax.set_ylabel(r'$\sigma_{los}$ $[Km/s]$')
    fig54.show()


    #where to evaluate the data becouse there are nan if there are no measuraments
    valuto_rho= rho > 0
    valuto_v_los = np.isnan(v_los) == False
    valuto_sigma_los = sigma_los > 120.
    valuto_err_v = np.isnan(error_v_los) == False
    valuto_err_sigma  = np.isnan(error_sigma_los) == False
    valuto_err_rho = np.isnan(error_rho) == False


    res_rho = 0*np.copy(x)
    res_v = 0*np.copy(x)
    res_sigma = 0*np.copy(x)
    res_rho[valuto_rho*valuto_err_rho] = np.abs((rho_model[valuto_rho*valuto_err_rho]-10**rho[valuto_rho*valuto_err_rho])/10**rho[valuto_rho*valuto_err_rho])
    res_v[valuto_v_los*valuto_err_v] = np.abs((v_model[valuto_v_los*valuto_err_v]-v_los[valuto_v_los*valuto_err_v])/v_los[valuto_v_los*valuto_err_v])
    res_sigma[valuto_sigma_los*valuto_err_sigma] = np.abs((sigma_model[valuto_sigma_los*valuto_err_sigma]-sigma_los[valuto_sigma_los*valuto_err_sigma])/sigma_los[valuto_sigma_los*valuto_err_sigma])

    res_v[np.argmax(res_v)] = np.min(res_v)

    mydata3 =d.data(x,y,np.copy(res_rho),np.copy(np.log10(res_v)),np.copy(np.log10(res_sigma)),0.,0.,0.,0.,0.,0.,0.)

    fig55,ax55,pmesh55,cbar55 = mydata3.surface_density()
    #ax55.set_title(r'$log \frac{model-data}{data}$')
    #cbar55.ax.set_ylabel(r'log $\left(\frac{\Sigma_{model} - \Sigma_{data}}{\Sigma_{data}}\right)$')
    cbar55.ax.set_ylabel(r' $log\left(\frac{\Sigma_{model} - \Sigma_{data}}{\Sigma_{data}}\right)$')
    fig55.show()

    fig56,ax56,pmesh56,cbar56 = mydata3.velocity_map()
    #ax56.set_title(r'$log \frac{model-data}{data}$')
    #cbar56.ax.set_ylabel(r'log $\left(\frac{v_{model} - v_{data}}{v_{data}}\right)$')
    cbar56.ax.set_ylabel(r' $log\left(\frac{v_{model} - v_{data}}{v_{data}}\right)$')
    fig56.show()

    fig57,ax57,pmesh57,cbar57 = mydata3.dispersion_map()
    #ax57.set_title(r'$log \frac{model-data}{data}$')
    #cbar57.ax.set_ylabel(r'log $\left(\frac{\sigma_{model} - \sigma_{data}}{\sigma_{data}}\right)$')
    cbar57.ax.set_ylabel(r'$log\left(\frac{\sigma_{model} - \sigma_{data}}{\sigma_{data}}\right)$')
    fig57.show()
