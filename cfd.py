##### Stable Fluids #####

# https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf

## w0(x) -> add force -> w1(x) -> advect ->  w2(x) -> diffuse -> w3(x) -> project -> w4(x)


import taichi as ti

###########
## CONFIG
##########

##use CPU for debugging with print
ti.init(arch=ti.cpu)

N = 300 #field measurments
unit = 1/N


#w = vel
u = ti.field(dtype=ti.f32, shape=(N,N)) # x velocity field
v = ti.field(dtype=ti.f32, shape=(N,N)) # y velocity field
density = ti.field(dtype=ti.f32, shape=(N,N)) # density field
p = ti.field(dtype=ti.f32, shape=(N,N)) # density field
div = ti.field(dtype=ti.f32, shape=(N,N)) # density field
gravity = -9.8

u_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step x velocity field
v_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step y velocity field
density_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step density field
decay = 0.995
dt = 0.01 #timestep

diff = 0.0001 #diffusion coefficient

## potential to add viscosity here


###########
## HELPERS
###########

@ti.kernel
def initialize_fields():
    
    for i,j in density:
        #define center
        cx, cy =N/2, N/2
        
        ## initialize fields
        # simple stream function: like ripples or noise
        psi = ti.sin(0.10 * i) * ti.cos(0.10 * j)

        # finite differences to compute partial derivatives
        dpsi_dx = (ti.sin(0.10 * (i + 0.10/N)) * ti.cos(0.10 * j) - psi) * N
        dpsi_dy = (ti.sin(0.10 * i) * ti.cos(0.10 * (j + 1/N)) - psi) * N

        outward_force = (ti.Vector([i,j]) - ti.Vector([cx,cy]))*60
        # curl = (∂ψ/∂y, -∂ψ/∂x)
        # u[i, j] =  dpsi_dy/unit*10 
        # v[i, j] = -dpsi_dx/unit*10
        u[i, j] =  dpsi_dy/unit*50 + outward_force[0]
        v[i, j] = -dpsi_dx/unit*50 + outward_force[1]
        #u[i, j] =  outward_force[0]
        #v[i, j] =  outward_force[1]


        #u[i,j] = 10/unit
        #v[i, j] = 10/unit

        ##circle in center
        r = N//5
        dist = (ti.Vector([i,j]) - ti.Vector([cx,cy])).norm()
        if dist < r:
            density[i, j] = clamp(1 * (1-dist/r),0.0,1.0)

        
@ti.func
def set_p():
    for I in ti.grouped(p):
        p[I] = 0
        
@ti.kernel
def add_source(x: int,y: int):
    r = int(N*0.02)
    

    for i,j in ti.ndrange((x-r,x+r),(y-r,y+r)):
        dist = (ti.Vector([i,j]) - ti.Vector([x,y])).norm()
        vel = (ti.Vector([i,j]) - ti.Vector([x,y]))
        ##print("dist = ",1-dist/r)
        
        density[i, j] = clamp((density[i,j] + clamp((1-dist/r),0.0,1.0)),0.0,1.0)
        u[i, j] =  (vel[0]*N)*(1-dist/r)*10
        v[i, j] =  (vel[1]*N)*(1-dist/r)*10
    

@ti.kernel
def copy_field(field1: ti.template(), field2: ti.template()):
    for I in ti.grouped(field1):
        field1[I] = field2[I]


@ti.func
def set_bnd(field: ti.template()):
    N_ = N-1
    ##set left right wall
    for j in range(1,N_):
        field[0,j] = field[1,j]
        field[N_,j] = field[N_-1, j]
    ## set top and bottom wall
    for i in range(1,N_):
        field[i,0] = field[i,1]
        field[i,N_] = field[i,N_-1]

    ##set corners - average adjacent sides
    field[0,0] = 0.5 * (field[0,1] + field[1,0]) ## top left
    field[0,N_] = 0.5 * (field[0,N_-1] + field[1,N_]) ## bottom left
    field[N_,N_] = 0.5 * (field[N_,N_-1] + field[N_-1,N_]) ## bottom right
    field[N_,0] = 0.5 * (field[N_-1,0] + field[N_,1]) ## top right

@ti.func
def set_vel_bnd(u_field: ti.template(), v_field: ti.template()):
    N_ = N-1

    ## for horizontal flip the vel, for vertical grab tangential

    ##set left right wall
    for j in range(1,N_):
        u_field[0,j] = -u_field[1,j]
        u_field[N_,j] = -u_field[N_-1, j]
        v_field[0,j] = v_field[1,j]
        v_field[N_,j] = v_field[N_-1, j]
        
    ## set top and bottom wall
    for i in range(1,N_):
        u_field[i,0] = u_field[i,1]
        u_field[i,N_] = u_field[i,N_-1]
        v_field[i,0] = -v_field[i,1]
        v_field[i,N_] = -v_field[i,N_-1]

    ##set corners - average adjacent sides
    u_field[0,0] = 0.5 * (u_field[0,1] + u_field[1,0]) ## top left
    u_field[0,N_] = 0.5 * (u_field[0,N_-1] + u_field[1,N_]) ## bottom left
    u_field[N_,N_] = 0.5 * (u_field[N_,N_-1] + u_field[N_-1,N_]) ## bottom right
    u_field[N_,0] = 0.5 * (u_field[N_-1,0] + u_field[N_,1]) ## top right
    v_field[0,0] = 0.5 * (v_field[0,1] + v_field[1,0]) ## top left
    v_field[0,N_] = 0.5 * (v_field[0,N_-1] + v_field[1,N_]) ## bottom left
    v_field[N_,N_] = 0.5 * (v_field[N_,N_-1] + v_field[N_-1,N_]) ## bottom right
    v_field[N_,0] = 0.5 * (v_field[N_-1,0] + v_field[N_,1]) ## top right

##clamp
@ti.func
def clamp(value, min, max):
    return ti.max( min, ti.min( float(value), max ) )


##bilinear interpolation
@ti.func
def bilerp(bl, br, tr, tl, s_x, s_y):
    ## bilinear interpolation
            ## bottom_left        bottom right      top right     top left
    return (1-s_x)*(1-s_y)*bl + (s_x*(1-s_y)*br) + s_x*s_y*tr + (1-s_x)*s_y*tl

@ti.func
def get_value(field, x_l, x_r, y_b, y_t, s_x, s_y):
    ## corner samples
   
    bl_d = field[x_l,y_b]
    br_d = field[x_r,y_b]
    tl_d = field[x_l,y_t]
    tr_d = field[x_r,y_t]

    return bilerp(bl_d, br_d, tr_d, tl_d, s_x, s_y)
    
@ti.kernel
def clamp_values(field: ti.template()):
    for I in ti.grouped(field):
       
        field[I] = clamp(field[I],0.0,1.0)
       






###########
## FORCES 
###########



## ADD FORCE
    #NOTES:
    #w1(x) = w0(x) + delta T * f(x,t)
    #frame 2 vel = frame 1 vel + the force * time interval
@ti.kernel
def force():
    for I in ti.grouped(density):
        ## adding gravity force * units per timestep to vertical velocity
      ##  print(v[I]," + ", gravity*unit*dt, )
        v[I] = v[I] + gravity * unit
    
    

## ADVECT
    #NOTES:
    #particle path =  p(x, s)
    #w1(x) = w0(p(x0, -dt))
    #basically frame 2 particle vel = tracing vel of particle backwards to previous time step, 
    # sampling vel from that x,y coordinate, interpolate
    # multiply by some diminishing fraction so doesn't blow up? * 0.95?
@ti.kernel
def advect():
    ##grab the vertical and horizontal vector from the previous frame
    for i, j in ti.ndrange((1,N-1), (1,N-1)):

        ## velocity path at this point in units per timestep
        u_path = u_prev[i,j] * dt *unit
        v_path = v_prev[i,j] * dt * unit
        
        ## subtracting vel from current position to get where it was in previous position
        x_old = i - u_path
        y_old = j - v_path

        ## calculate corners with boundary enforcement
        x_l = int(ti.floor(clamp(x_old, 1, N-1)))
        x_r = int(ti.ceil(clamp(x_l+1, 1, N-1)))
        y_b = int(ti.floor(clamp(y_old, 1, N-1)))
        y_t = int(ti.ceil(clamp(y_b+1, 1, N-1)))
        
        
        # determine where in cell source is
        s_x = x_old - x_l
        s_y = y_old - y_b

        #Get values from prev pos
        density[i,j] = get_value(density_prev, x_l, x_r, y_b, y_t, s_x, s_y)
        u[i,j] = get_value(u_prev, x_l, x_r, y_b, y_t, s_x, s_y)
        v[i,j] = get_value(v_prev, x_l, x_r, y_b, y_t, s_x, s_y)
       

    set_bnd(density)
    set_vel_bnd(u, v)
    


## DIFFUSE
    ## NOTES: use gauss seidl relaxation, 20 iteration
    # w1(x,y) = w0(x,y) + a( w1(x-1,y) + w1(x+1,y) + w1(x,y-1) + w1(x,y+1)) / 1 + 4a
    #where a = diffusion coeff
    # gauss seidl interpolation, 20 iterations
    # calculate diffusion by averaging velocity/density for a given cell between the 4 or 8 neighbours, 
    # note: use the new values for the current iteration when available (the cells that came before)

@ti.kernel
def diffuse():

    ##diffusion co-efficient
    a = dt * diff * N * N

    for k in range(20):
        for i, j in ti.ndrange((1,N-1),(1,N-1)):
            ## don't diffuse on boundaries
            density[i,j] = (density_prev[i,j] + a*(density[i-1,j]+density[i,j-1]+ density[i+1,j] + density[i,j+1]))/(1+4*a) * decay
            u[i,j] = (u_prev[i,j] + a*(u[i-1,j]+u[i,j-1]+ u[i+1,j] + u[i,j+1]))/(1+4*a)
            v[i,j] = (v_prev[i,j] + a*(v[i-1,j]+v[i,j-1]+ v[i+1,j] + v[i,j+1]))/(1+4*a)
        
        set_bnd(density)
        set_vel_bnd(u,v)


## PROJECT
    ## NOTES:
    # solve for divergence: div[x,y] = (-1/2N)*( (r - l ) + (u - d)
    # iterate over pressure equation: p[i,j]= 1/4 * (div[i,j] + l + r + t + b)
    # resolve pressure : u[i,j] -= 0.5 * N * (p[i+1, j] - p[i-1,j] , v[i,j] -= 0.5 * N * (p[i, j+1] - p[i,j-1]
    # gauss seidl

@ti.kernel
def project():
    set_p()
    # compute divergence
    for i, j in ti.ndrange((1,N-1),(1,N-1)):
        div[i,j] = ((-0.5)) * ((u[i+1,j]-u[i-1,j]) + (v[i,j+1] - v[i,j-1]))

        # at boundaries
    set_bnd(div)
   


    for k in range(20):
        ## don't calculate on boundaries
        
        for i, j in ti.ndrange((1,N-1),(1,N-1)):
    
            ## compute pressure (0 placeholder for first iteration)
            p[i,j] = (1/4) * (div[i,j] + p[i-1,j] + p[i+1,j] + p[i,j+1] + p[i,j-1])

            #compute boundaries
        set_bnd(p)
    
    ## resolve pressure
    for i, j in ti.ndrange((1,N-1),(1,N-1)):
        #subtract horizontal and vertical pressure from velocity fields to resolve divergence
        u[i,j] -= 0.5 * (p[i+1, j] - p[i-1,j])*unit
        v[i,j] -= 0.5 * (p[i, j+1] - p[i,j-1])*unit

    #boundaries
    set_vel_bnd(u, v)






## TO DO:
## min_max helper function
## gauss_seidl helper function
## create random curl velocity in initial vel fields
## render logic
## drawing function
## make sure everything in proper units
## SOR instead of GS?







###########
## RENDER
###########



def substep():
    ##force() # gravity
    ##copy_field(v_prev, v)
    if window.is_pressed(ti.ui.LMB):
        x,y = window.get_cursor_pos()
        x = int(x*N)
        y = int(y*N)
        add_source(x,y)
        copy_field(density_prev, density)
        copy_field(u_prev, u)
        copy_field(v_prev, v)
        print("click detected")
    advect()
    copy_field(v_prev, v)
    copy_field(u_prev, u)
    copy_field(density_prev, density)

    diffuse()
    copy_field(v_prev, v)
    copy_field(u_prev, u)
   
    
    project()
    copy_field(v_prev, v)
    copy_field(u_prev, u)
    copy_field(density_prev, density)
    ##clamp_values(density)




## Display logic

window = ti.ui.Window("Stable Fluids - Steve Cutler", (1024, 1024), vsync=True)
canvas = window.get_canvas()
#set to white
canvas.set_background_color((0,0,1))
#create scene and camera
##scene = window.get_scene()
##camera = ti.ui.Camera()
## start time
current_t = 0.0
substeps = 5

##Initialize fields
initialize_fields()
copy_field(density_prev, density)
copy_field(u_prev, u)
copy_field(v_prev, v)  

##Render loop
while window.running:



    window.show()
    # reset every 1.5 seconds
    # if current_t > 2.5:
    #     initialize_fields()
    #     current_t=0
        
    # calculate all substeps and then update vertices
    for _ in range(substeps):
        substep()
    current_t += dt    
    canvas.set_image(density)


