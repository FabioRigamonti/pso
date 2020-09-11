import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

'''In this case the grid is squared, so x and y are vectors with dimension N^2. In general
the grid can have different dimension in x and y, but I can calculate them. In this case 
the grid is equally spaced but in general it could not, but I can calculate the spacement 
point by point. THE ONLY THING THAT I MUST ASSUME IS THAT 

X IS OF THE FORM [  1,1,1,1,1,1 ...,J-1,
                    2,2,2,2,2,2 ...,J-1,
                    3,3,3,3,3,3 ...,J-1,
                            .
                            .
                            .       
                            N-1       ]

Y IS OF THE FORM [  1,2,3,4,5,6 ...,J-1,
                    1,2,3,4,5,6 ...,J-1,
                    1,2,3,4,5,6 ...,J-1,
                            .
                            .
                            .
                            N-1        ]   

where J is the lenght in the y direction and N in the x direction.'''

class data:

    def __init__(self,
             x,                   # grid coordinates, x
             y,                   # grid coordinates, y
             rho,                 # density value at coordinates
             v_los,               # line of sight velocity value at coordinates
             sigma_los,           # line of sight velocity dispersion value at coordinates
             h3,                  # 3th moment of the LOSVD at coordinates
             h4,                  # 4th moment of the LOSVD at coordinates
             error_rho,           # density error value at coordinates
             error_v_los,         # line of sight velocity error value at coordinates
             error_sigma_los,     # line of sight velocity dispersion error value at coordinates
             error_h3,            # 3th moment error
             error_h4,            # 4th moment error
             *args, **kwargs):

        self.x         = x
        self.y         = y
        self.rho       = rho
        self.v_los     = v_los
        self.sigma_los = sigma_los
        self.h3        = h3
        self.h4        = h4
        
        self.error_rho       = error_rho
        self.error_v_los     = error_v_los
        self.error_sigma_los = error_sigma_los
        self.error_lrho      = self.error_rho/self.rho
        self.error_h3        = error_h3
        self.error_h4        = error_h4
        
        #some useful things
        #min and max of the grid
        self.xmin, self.xmax = np.amin(x), np.amax(x)
        self.ymin, self.ymax = np.amin(y), np.amax(y)

        #dimension in the y direction (J) and in the x direction (N)
        self.J = np.size(x[ x == x[0] ])       #I will find J x[0]
        self.N = np.size(y[ y == y[0] ])       #I will find N y[0]

        #vectors with only the numbers without repetition
        #x_true has dimension N, while y_true has dimension J
        self.x_true = x[0 : self.J*self.N : self.J]
        self.y_true = y[0 : self.J]

        #dimension of the grid, dx size N-1 and dy size J-1. The grid is not supposed to be const
        #dx[0] is the distance between x1-x0 in the x direction
        #dy[0] is the distance between x1-x0 in the y direction
        self.dx = self.x_true[1:]-self.x_true[:-1]
        self.dy = self.y_true[1:]-self.y_true[:-1]


    def refined_grid(self,size=10):

        x_rand = np.zeros((self.N-1,self.J-1,size*size))
        y_rand = np.zeros((self.N-1,self.J-1,size*size))

        for k in range(0,self.N-1):
                
            a = np.linspace(self.x_true[k],self.x_true[k+1],size)
            xrand = np.repeat(a,size).reshape(size,size).T.ravel()

            for j in range(0,self.J-1):
                    
                b = np.linspace(self.y_true[j],self.y_true[j+1],size)
                yrand = np.repeat(b,size)

                x_rand[k,j,:] = xrand
                y_rand[k,j,:] = yrand
        
        return(x_rand,y_rand)
       

    def grid_plot(self):

        '''draw the grid return fig and ax object.
        This works for rectangular grid.
        Should be tested for circular grid. 
        There will be some problems becouse you can't use y_true and x_true.
        It could work becouse the idea is to use the fact the 
        x is [0,0,0,...,1,1,1....,2,2,2]
        y is [0,1,2....,0,1,2,...]
        So it is very easy to plor the vertical lines'''

        fig = plt.figure()
        ax = fig.add_subplot()

        '''In order to plot the orizontal lines I need to write x in the form
        x [0,1,2....0,1,2...] and y [0,0,0,....,1,1,1....].
        So x_plot has J repeated vectors of the form [0,1,...N]
        y_plot has J repeated vectors of the form [0,0,..] and so on
        in practice they are matrix J X N'''

        x_plot = np.array([ self.x_true for i in range (0,self.J)])
        y_plot = np.array([ self.y[i : self.J*self.N : self.J ] for i in range(0,self.J) ])


        for i in range (0,self.N):
            
            ax.plot(self.x[ i*self.J : (i+1)*self.J ], self.y[ i*self.J : (i+1)*self.J ],'r',lw = 0.3)
            

        for i in range(0,self.J):
          
            ax.plot(x_plot[i], y_plot[i],'r',lw = 0.3)

        ax.set_xlabel('x[Kpc]')
        ax.set_ylabel('y[Kpc]')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)

        return(fig,ax)




    def galaxy_plot(self,j = 0):

        '''plot the galaxy (rho > 0. points).
        if j = 1 plot also the grid. Default is 0.
        Return figure and axes.'''

        i = self.rho > 0.

        if (j == 1):

            fig,ax = self.grid_plot()
            
            ax.plot(self.x[i], self.y[i],'.b')
        
        else:

            fig = plt.figure()
            ax = fig.add_subplot()    

            ax.plot(self.x[i], self.y[i],'.b')

            ax.set_xlabel('x[Kpc]')
            ax.set_ylabel('y[Kpc]')
            ax.set_xlim(self.xmin,self.xmax)
            ax.set_ylim(self.ymin,self.ymax)


        return(fig,ax)


    
    def surface_density(self):

        '''pcolormesh need 3 things X and Y are 2 matrices that
        define the grid. rho matrix is the matrix of density.
        rho_matrix[i,j] is the density in X[i,j] and Y[i,j].
        In reality, if X and Y are of dimensions N X J, rho_matrix
        should be of dimensions (N-1) X (J-1) becouse pcolormesh 
        wants rho_matrix[i,j] to be the density in in the middle of the 
        quadrilateral { (X[i,j],Y[i,j]) ;  (X[i+1,j],Y[i+1,j]) ;
        (X[i,j+1],Y[i,j+1]) ; (X[i+1,j+1],Y[i+1,j+1])}. Dovrebbe essere giusto vedi quad
        Return: figure, axes, colorbar '''

        rho_matrix = self.rho.reshape((self.N,self.J))
        X = self.x.reshape((self.N,self.J))
        Y = self.y.reshape((self.N,self.J))

        #the plot is done in log scale, I've used the fact that when rho = 0 log(0) = inf 
        fig = plt.figure()
        ax = fig.add_subplot()    

        pmesh = ax.pcolormesh(X,Y,np.log10(rho_matrix),cmap = 'hot',shading='gouraud')
        ax.set_xlabel('x[Kpc]')
        ax.set_ylabel('y[Kpc]')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        cbar = plt.colorbar(pmesh)
        cbar.ax.set_ylabel(r'$log_{10}(\Sigma)$ $[M_{\odot}/Kpc^2]$')

        return (fig,ax,pmesh,cbar)


    
    def velocity_map(self):

        '''velocity map plot. I've set the velocity where
        rho is zero equal to nan. In this way I will plot
        only the velocity within the galaxy.
        Return: figure, axes, colorbar'''

        v_inf = self.v_los[:]
        v_inf[self.rho == 0.] = np.nan

        v_matrix = v_inf.reshape((self.N,self.J))
        X = self.x.reshape((self.N,self.J))
        Y = self.y.reshape((self.N,self.J))

        fig = plt.figure()
        ax = fig.add_subplot()
        pmesh = ax.pcolormesh(X,Y,v_matrix,cmap = 'hot',shading = 'gouraud' )
        ax.set_xlabel('x[Kpc]')
        ax.set_ylabel('y[Kpc]')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        cbar = fig.colorbar(pmesh)
        cbar.ax.set_ylabel(r'$v_{los}$ $[Km/s]$')
        
        return(fig,ax,pmesh,cbar)

    

    def dispersion_map(self):

        '''dispersion velocity map plot. I've set the dispersion where
        rho is zero equal to nan. In this way I will plot
        only the velocity within the galaxy.
        Return: figure, axes, colorbar'''

        sigma_inf = self.sigma_los[:]
        sigma_inf[self.rho == 0.] = np.nan

        sigma_matrix = sigma_inf.reshape((self.N,self.J))
        X = self.x.reshape((self.N,self.J))
        Y = self.y.reshape((self.N,self.J))

        fig = plt.figure()
        ax = fig.add_subplot()
        pmesh = ax.pcolormesh(X,Y,sigma_matrix,cmap = 'hot',shading = 'gouraud' )
        ax.set_xlabel('x[Kpc]')
        ax.set_ylabel('y[Kpc]')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        cbar = fig.colorbar(pmesh)
        cbar.ax.set_ylabel(r'$\sigma_{los}$ $[Km/s]$')

        return(fig,ax,pmesh,cbar)

    
    def h3_map(self):

        '''dispersion velocity map plot. I've set the dispersion where
        rho is zero equal to nan. In this way I will plot
        only the velocity within the galaxy.
        Return: figure, axes, colorbar'''

        h3_inf = self.h3[:]
        h3_inf[self.rho == 0.] = np.nan

        h3_matrix = h3_inf.reshape((self.N,self.J))
        X = self.x.reshape((self.N,self.J))
        Y = self.y.reshape((self.N,self.J))

        fig = plt.figure()
        ax = fig.add_subplot()
        pmesh = ax.pcolormesh(X,Y,h3_matrix,cmap = 'hot',shading = 'gouraud' )
        ax.set_xlabel('x[Kpc]')
        ax.set_ylabel('y[Kpc]')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        cbar = fig.colorbar(pmesh)
        cbar.ax.set_ylabel(r'$h3$')

        return(fig,ax,pmesh,cbar) 

    
    def h4_map(self):

        '''dispersion velocity map plot. I've set the dispersion where
        rho is zero equal to nan. In this way I will plot
        only the velocity within the galaxy.
        Return: figure, axes, colorbar'''

        h4_inf = self.h4[:]
        h4_inf[self.rho == 0.] = np.nan

        h4_matrix = h4_inf.reshape((self.N,self.J))
        X = self.x.reshape((self.N,self.J))
        Y = self.y.reshape((self.N,self.J))

        fig = plt.figure()
        ax = fig.add_subplot()
        pmesh = ax.pcolormesh(X,Y,h4_matrix,cmap = 'hot',shading = 'gouraud' )
        ax.set_xlabel('x[Kpc]')
        ax.set_ylabel('y[Kpc]')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        cbar = fig.colorbar(pmesh)
        cbar.ax.set_ylabel(r'$h4$')

        return(fig,ax,pmesh,cbar) 


    def density_3D(self):

        '''3D plot with positions and density.
        Return fig,ax'''

        fig = plt.figure()
        ax =fig.add_subplot(projection = '3d')
        ax.scatter(self.x,self.y,np.log10(self.rho),c = 'black',s = 1.5,marker = '.')
        ax.set_xlabel('x[Kpc]')
        ax.set_ylabel('y[Kpc]')
        ax.set_zlabel(r'$log_{10} \Sigma$  $M_{\odot}/Kpc^2$')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)

        return(fig,ax)


    
    def velocity_3D(self):

        '''3D plot with positions and velocities.
        Return fig,ax'''

        v_inf = self.v_los[:]
        v_inf[self.rho == 0.] = np.nan


        fig = plt.figure()
        ax =fig.add_subplot(projection = '3d')
        ax.scatter(self.x,self.y,v_inf,c = 'black',s = 1.5,marker = '.')
        ax.set_xlabel('x[Kpc]')
        ax.set_ylabel('y[Kpc]')
        ax.set_zlabel(r'$v_{los}$ $[Km/s]$')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)

        return(fig,ax)


    
    def dispersion_3D(self):

        '''3D plot with positions and dispersion velocities.
        Return fig,ax'''

        sigma_inf = self.sigma_los[:]
        sigma_inf[self.rho == 0.] = np.nan

        fig = plt.figure()
        ax =fig.add_subplot(projection = '3d')
        ax.scatter(self.x,self.y,sigma_inf,c = 'black',s = 1.5,marker = '.')
        ax.set_xlabel('x[Kpc]')
        ax.set_ylabel('y[Kpc]')
        ax.set_zlabel(r'$\sigma_{los}$ $[Km/s]$')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)

        return(fig,ax)
