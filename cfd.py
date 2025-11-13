##### Stable Fluids #####

## w0(x) -> add force -> w1(x) -> advect ->  w2(x) -> diffuse -> w3(x) -> project -> w4(x)

## make random splash functio for origin  thats different for each iteration - random number gen
## add noise to source add function
## make diffusion milky?
## add randomization to input level of add source
## add noise field - disturb? turubulence? that makes it smoky
## 2 modes: fluid and smoke

import random
import taichi as ti

###########
## CONFIG
##########

##use CPU for debugging with print
ti.init(arch=ti.cpu)

N = 512 #field measurments
unit = 1/N
gravity = -9.8
bouyancy_mult = 20
decay = 0.999 # decay rate
vel_decay = 0.999 # decay rate
dt = 0.05 #timestep
diff = 0.00001 #diffusion coefficient
curl_co = 300 #vortex coefficient
iterations = 4
substeps = 1
out_force = 25
source_r = 0.05
source_vel = 100
source_dens_mult = 0.5
current_t = 0.0

u = ti.field(dtype=ti.f32, shape=(N,N)) # x velocity field
v = ti.field(dtype=ti.f32, shape=(N,N)) # y velocity field
density = ti.field(dtype=ti.f32, shape=(N,N)) # density field
p = ti.field(dtype=ti.f32, shape=(N,N)) # pressure field
div = ti.field(dtype=ti.f32, shape=(N,N)) # divergence field
curl = ti.field(dtype=ti.f32, shape=(N,N)) # curl field
noise_scalar = ti.field(dtype=ti.f32, shape=(N,N))
noise = ti.Vector.field(2, dtype=ti.f32, shape=(N,N))
source_noise = 0


u_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step x velocity field
v_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step y velocity field
density_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step density field

prev_mouse = None


## potential to add viscosity here


###########
## HELPERS
###########

@ti.kernel
def initialize_fields():
    
    for i,j in density:
        #define center
        cx, cy =N/2, N/2
        outward_force = (ti.Vector([i,j]) - ti.Vector([cx,cy]))*out_force



        ##circle in center
        r = N//5
        dist = (ti.Vector([i,j]) - ti.Vector([cx,cy])).norm()
        if dist < r:
            density[i, j] = clamp(1 * (1-dist/r),0.0,1.0)
            u[i, j] = outward_force[0]
            v[i, j] = outward_force[1]

@ti.func
def noise(x,y,t,freq):
    return ti.sin(x*freq+t) * ti.cos(y*freq*1.1+t)
        
@ti.func
def set_p():
    for I in ti.grouped(p):
        p[I] = 0
        
@ti.kernel
def add_source(x: int, y: int, vx: int, vy: int):
    print("add source")
    r = int(N*source_r)
    
    for i,j in ti.ndrange((x-r,x+r),(y-r,y+r)):      
        dist = (ti.Vector([i,j]) - ti.Vector([x,y])).norm()


        if dist < r:
            falloff = 1.0 - dist / r
            w = falloff * falloff  # smooth falloff
            vel = (ti.Vector([i,j]) - ti.Vector([x,y])) 
            
            exp_w = ti.exp(-4*dist/r)

            
            perp = ti.Vector([-vel[1], vel[0]])  # perpendicular swirl
            mix = ti.random(ti.f32) * -0.1  # ±30% swirl mix
            noisy_dir = (vel + perp) * mix * source_noise
        
            rand = 0.1 + 0.9 * ti.random(ti.f32)

            # density[i, j] = clamp( (density[i,j] + (1-dist/r))*(1-dist/r),0.0,1.0)
            density[i, j] = density[i,j] + clamp(exp_w,0,1)*source_dens_mult* rand

            u[i, j] =  u[i,j]*0.9 + ((vel[0])*N)*clamp(w,0,1)*rand + noisy_dir[0]*10 + vx*5
            v[i, j] =  v[i,j]*0.9 + ((vel[1])*N)*clamp(w,0,1)*rand + noisy_dir[1]*10 + vy*5

            density_prev[i,j] = density[i,j]
            u_prev[i,j] = u[i,j]
            v_prev[i,j] = v[i,j]
    

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


## VEL DAMPEN
@ti.kernel
def vel_dampen():
    for I in ti.grouped(u):
        u[I] = u[I]* ti.min(density[I],1) * 0.9
        v[I] = v[I]* ti.min(density[I],1) * 0.9


## ADD FORCE
    #NOTES:
    #w1(x) = w0(x) + delta T * f(x,t)
    #frame 2 vel = frame 1 vel + the force * time interval
@ti.kernel
def gravity():
    for I in ti.grouped(v):

        v[I] = v[I] + gravity * unit

@ti.kernel
def up():
    for I in ti.grouped(v):
        v[I] = v[I] + bouyancy_mult * dt * N
        ##print(v[I])
    
    

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

        x_old = ti.max(0.5, ti.min(x_old, N - 1.5))
        y_old = ti.max(0.5, ti.min(y_old, N - 1.5))

        ## calculate corners with boundary enforcement
        x_l = int(ti.floor(x_old))
        x_r = x_l+1
        y_b = int(ti.floor(y_old))
        y_t = y_b+1
        
        
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
def diffuse_dens():

    ##diffusion co-efficient
    a = dt * diff * N * N

    for k in range(iterations):
        for i, j in ti.ndrange((1,N-1),(1,N-1)):
            ## don't diffuse on boundaries
            density[i,j] = (density_prev[i,j] + a*(density[i-1,j]+density[i,j-1]+ density[i+1,j] + density[i,j+1]))/(1+4*a) * decay
        
    set_bnd(density)

@ti.kernel
def diffuse_vel():

    ##diffusion co-efficient
    a = dt * diff * N * N

    for k in range(iterations):
        for i, j in ti.ndrange((1,N-1),(1,N-1)):
            ## don't diffuse on boundaries
            u[i,j] = (u_prev[i,j] + a*(u[i-1,j]+u[i,j-1]+ u[i+1,j] + u[i,j+1]))/(1+4*a) * vel_decay
            v[i,j] = (v_prev[i,j] + a*(v[i-1,j]+v[i,j-1]+ v[i+1,j] + v[i,j+1]))/(1+4*a) * vel_decay
        
    set_vel_bnd(u,v)

@ti.kernel
def decay_vel():

    ##diffusion co-efficient
    a = dt * diff * N * N

    for I in ti.grouped(density):
        u[I] = u[I] * 0.99
        v[I] = v[I] * 0.99


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
        div[i,j] = (-0.5) * ((u[i+1,j]-u[i-1,j]) + (v[i,j+1] - v[i,j-1]))

        # at boundaries
    ##set_bnd(div)
    for k in range(iterations):
        ## don't calculate on boundaries
        
        for i, j in ti.ndrange((1,N-1),(1,N-1)):
    
            ## compute pressure (0 placeholder for first iteration)
            p[i,j] = (1/4) * (div[i,j] + p[i-1,j] + p[i+1,j] + p[i,j+1] + p[i,j-1])

            #compute boundaries
    ##set_bnd(p)
    
    ## resolve pressure
    for i, j in ti.ndrange((1,N-1),(1,N-1)):
        #subtract horizontal and vertical pressure from velocity fields to resolve divergence
        u[i,j] -= 0.5 * (p[i+1, j] - p[i-1,j])*unit
        v[i,j] -= 0.5 * (p[i, j+1] - p[i,j-1])*unit

    #boundaries
    set_vel_bnd(u, v)

## VORTEX CONFINEMENT
    ##Notes: partial v/partial x - partial u/partial y
    ## how does vertical vel change as we x increases, and vice versa

@ti.kernel
def compute_curl():
    for i,j in ti.ndrange((1,N-1),(1,N-1)):
        curl[i,j] = 0.5*(v[i+1,j] - v[i-1,j]) - 0.5*(u[i,j+1]-u[i,j-1])
    ##set_bnd(curl)
    
    for i,j in ti.ndrange((2,N-2),(2,N-2)):
        grad_x = 0.5*(ti.abs(curl[i+1,j]) - ti.abs(curl[i-1,j]))
        grad_y = 0.5*(ti.abs(curl[i,j+1]) - ti.abs(curl[i,j-1]))
        mag = ti.Vector([grad_x,grad_y]).norm() + 1e-5 ## adding tiny amount to avoid dividing by 0
        Nx, Ny = grad_x/mag, grad_y/mag
        u[i,j] = u[i,j] + Ny * curl[i,j] * dt * curl_co
        v[i,j] = v[i,j] - Nx * curl[i,j] * dt * curl_co
    set_vel_bnd(u,v)
   
## TURBULENCE
@ti.kernel
def calc_noise(t: ti.f32, freq: ti.f32, amp: ti.f32, time_mult: ti.f32):
    for i, j in ti.ndrange((1,N-1),(1,N-1)):
        #scalar noise
        noise_scalar[i, j] = amp * noise(i, j, t*time_mult, freq)
        ##print('noise scalar = ',noise_scalar[i,j], 'i and j =', i, ' ', j)

    #set boundary cells
    set_bnd(noise_scalar)
        
@ti.kernel
def apply_turbulence(strength: ti.f32, amp: ti.f32):
    for i, j in ti.ndrange((1,N-1),(1,N-1)):
        #derivatives
        delta_nx = (noise_scalar[i+1, j] - noise_scalar[i-1,j])
        delta_ny = (noise_scalar[i, j+1] - noise_scalar[i,j-1])

        vec = ti.Vector([delta_ny, -delta_nx])
        mag = vec.norm() + 1e-5
        curl_vector = (vec / mag) * amp
        ##print("curl vec = ",curl_vector)

        #u[i,j] = u[i,j] + 10
        #v[i,j] = v[i,j] + 10
        # print('***')
        # print('u = ', u[i,j])
        # print(u[i,j] + curl_vector[0] * strength)
        # print('v = ', v[i,j])
        # print(v[i,j] + curl_vector[1] * strength)

        u[i,j] = u[i,j] + curl_vector[0] * strength
        v[i,j] = v[i,j] + curl_vector[1] * strength
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
    global u, v, u_prev, v_prev, density, density_prev, current_t


    ## GRAVITY
    ##force() # gravity
    ##copy_field(v_prev, v)
    # vel_dampen()
    # v_prev, v = v, v_prev
    # u_prev, u = u, u_prev


    up()
    v_prev, v = v, v_prev
    u_prev, u = u, u_prev

    ## ADVECT
    advect()
    v_prev, v = v, v_prev
    u_prev, u = u, u_prev
    density_prev, density = density, density_prev
    

    # DECAY
    decay_vel()
    v_prev, v = v, v_prev
    u_prev, u = u, u_prev
   
    # DIFFUSE
    diffuse_dens()
    #diffuse_vel()
    
    density_prev, density = density, density_prev
    v_prev, v = v, v_prev
    u_prev, u = u, u_prev



   

        ##TURBLUENCE
    calc_noise(current_t, freq=.1, amp=1, time_mult=1)
    apply_turbulence(strength=10, amp=1)
    current_t = current_t + dt/substeps
    # copy_field(v_prev, v)
    # copy_field(u_prev, u)

    ##CURL
    #compute_curl()
    # v_prev, v = v, v_prev
    # u_prev, u = u, u_prev


    ## PROJECT
    project()
    v_prev, v = v, v_prev
    u_prev, u = u, u_prev
    density_prev, density = density, density_prev

    ##ti.profiler.print_kernel_profiler_info()





## DISPLAY LOGIC

img = ti.Vector.field(3, dtype=ti.f32, shape=(N,N))

@ti.kernel
def make_display_image():
    for i, j in density:
        d = ti.min(density[i, j], 1.0)           # clamp high values
        d = ti.sqrt(ti.max(d, 0.0))              # gamma-like contrast
        img[i, j][0] = d * 0.9                   # red channel
        img[i, j][1] = d * 0.6 + 0.1             # green channel
        img[i, j][2] = 1.0 - d * 0.8             # blue channel


window = ti.ui.Window("Stable Fluids - Steve Cutler", (1024, 1024), vsync=True)
canvas = window.get_canvas()
#set to white
canvas.set_background_color((0,0,0))
#create scene and camera
##scene = window.get_scene()
##camera = ti.ui.Camera()
## start time


##Initialize fields
##initialize_fields()
##copy_field(density_prev, density)
##copy_field(u_prev, u)
##copy_field(v_prev, v)  

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

            ## CHECK FOR MOUSE CLICKS
    if window.is_pressed(ti.ui.LMB):
        x,y = window.get_cursor_pos()
        x = int(x*N)
        y = int(y*N)


        
        if prev_mouse is not None:
            px, py = prev_mouse
            vx = int((x - px) * 100)   # scale sensitivity
            vy = int((y - py) * 100)
            
            add_source(x,y,vx,vy)
            # copy_field(density_prev, density)
            # copy_field(u_prev, u)
            # copy_field(v_prev, v)
        else:
            print("click detected")
            add_source(x,y,0,0)
            # copy_field(density_prev, density)
            # copy_field(u_prev, u)
            # copy_field(v_prev, v)

        prev_mouse = (x, y)
    else:
        prev_mouse = None


    make_display_image()
    current_t += 0.01
    canvas.set_image(density)


