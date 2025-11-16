##### Stable Fluids #####

# https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf

## w0(x) -> add force -> w1(x) -> advect ->  w2(x) -> diffuse -> w3(x) -> project -> w4(x)


#########
## TO DO 
#########

## 2 modes: fluid and smoke

## optimize for gpu usage, currently gpu runs it super slowly
#fix burst popping out of existence
#burst button doesn't really work


import random
import taichi as ti
import taichi
import random


###########
## CONFIG
##########

##use CPU for debugging with print
ti.init(arch=ti.cpu)

N = 512 #field measurments
unit = 1/N
## switches

burst = True

up_force = True


## perlin vars

pnoise = ti.field(dtype=ti.f32, shape=(N,N)) # pnoise field
pnoise_birth_mix = 3


## variables

gravity = -9.8
#bouyancy_mult = 20
decay = 0.997 # decay rate
dt = 0.05 #timestep
diff = 0.00005 #diffusion coefficient
curl_co = 25 #vortex coefficient
iterations = 4
substeps = 1
substep_count = 1
out_force = 2000
r = N//4
source_vel = 150
source_dens_mult = 0.3
current_t = 0.0
step = 15 # vel display field space step (number of arrows)

# fields/states
u = ti.field(dtype=ti.f32, shape=(N,N)) # x velocity field
v = ti.field(dtype=ti.f32, shape=(N,N)) # y velocity field
density = ti.field(dtype=ti.f32, shape=(N,N)) # density field

u_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step x velocity field
v_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step y velocity field
density_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step density field

p = ti.field(dtype=ti.f32, shape=(N,N)) # pressure field
div = ti.field(dtype=ti.f32, shape=(N,N)) # divergence field
curl = ti.field(dtype=ti.f32, shape=(N,N)) # curl field
noise_scalar = ti.field(dtype=ti.f32, shape=(N,N))
noise = ti.Vector.field(2, dtype=ti.f32, shape=(N,N))

arrows_array = ti.Vector.field(2, dtype=ti.f32, shape=(N//step*N//step*2))
triangles = ti.Vector.field(2, dtype=ti.f32, shape=(N//step*N//step*3))


prev_mouse = None

####################
## SETUP GUI PARAMS
####################

params = {
    "bouyancy_mult": 20.0,
    "source_radius": 0.05,
    "turb_freq": 0.06,
    "turb_amp": 1.0,
    "turb_strength": 10.0,
    "turb_speed": 0.02,
    
    "turbulence": True,
    "arrows": False,
    "noise_display": False
}





###########
## HELPERS
###########

@ti.kernel
def initialize_fields():
    for I in ti.grouped(density):
        density[I] = 0
        u[I] = 0
        v[I] = 0
        density_prev[I] = 0
        u_prev[I] = 0
        v_prev[I] = 0

@ti.kernel
def add_burst(freq:ti.f32):
        seed = rand()
        for i,j in density:
        #define center
            cx, cy =N/2, N/2
           
            
            mod_r = (r * pnoise_birth_mix * abs(p_noise_calc(i*freq,j*freq, seed+substep_count,1)))
            #circle in center            
            dist = (ti.Vector([i,j]) - ti.Vector([cx,cy])).norm()
            if dist < mod_r :
                density[i, j] = density[i,j] + clamp(1 * (1-dist/mod_r),0.0,1.0)
                vel_fade = dist/mod_r
                outward_force = (ti.Vector([i,j]) - ti.Vector([cx,cy]))*out_force*vel_fade
                u[i, j] = outward_force[0]
                v[i, j] = outward_force[1]
                #print(vel_fade)



@ti.func
def rand():
    return random.randint(0, 10_000_000)

@ti.func
def noise(x,y,t,freq):
    return ti.sin(x*freq+t*0.3) * ti.cos(y*freq*1.1+t*0.2)
        
@ti.func
def set_p():
    for I in ti.grouped(p):
        p[I] = 0

@ti.kernel
def add_source(x: int, y: int, vx: int, vy: int, source_r:ti.f32, freq:ti.f32, speed:ti.f32):


    r = int(N*source_r)
    ##mod_r = abs(p_noise_calc(x*perlin_freq,y*perlin_freq, x))
    
    for i,j in ti.ndrange((x-r,x+r),(y-r,y+r)):      
        dist = (ti.Vector([i,j]) - ti.Vector([x,y])).norm()


        if dist < r:
            falloff = 1.0 - dist / r
            w = falloff * falloff  # smooth falloff
            vel_mult = abs((3 * p_noise_calc(i*freq,j*freq, i+j+substep_count, speed)))
            vel = (ti.Vector([i,j]) - ti.Vector([x,y]))
            
            exp_w = ti.exp(-4*dist/r)

            
            perp = ti.Vector([-vel[1], vel[0]])  # perpendicular swirl
            mix = ti.random(ti.f32) * -0.1  # ±30% swirl mix
            noisy_dir = (vel + perp) * mix
        
            rand = 0.1 + 0.3 * ti.random(ti.f32)

            # density[i, j] = clamp( (density[i,j] + (1-dist/r))*(1-dist/r),0.0,1.0)
            density[i, j] = density[i,j] + clamp(exp_w,0,1)*source_dens_mult * clamp(vel_mult/3,0.1,1)

            u[i, j] =  u[i,j]*0.9 + ((vel[0])*N)*clamp(w,0,1)*vel_mult*2 + vx*5
            v[i, j] =  v[i,j]*0.9 + ((vel[1])*N)*clamp(w,0,1)*vel_mult*2 + vy*5

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
       

@ti.func
def hash(x, y, rand):
    # deterministic hash
    hash = x * 374761393 + y * 668265263
    hash = (hash ^ (hash >> 13)) * 1274126177
    hash = hash ^ (hash >> 16)
    return hash

#fade function
@ti.func
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

#gradient function
@ti.func
def gradient(ix, iy, seed, speed):
    h = hash(ix, iy, seed)
    
    ##bit wise mask
    angle = (h & 0xffff) * (2 * 3.14159265 / 65536.0)
    return ti.Vector([ti.cos(angle+ seed*speed), ti.sin(angle+seed*speed)])

@ti.func
def p_noise_calc(i, j, rand, speed):




    ## calculate corners with boundary enforcement
    x_l = int(ti.floor(clamp(i, 0, N-1)))
    x_r = int(ti.ceil(clamp(x_l+1, 1, N)))
    y_b = int(ti.floor(clamp(j, 0, N-1)))
    y_t = int(ti.ceil(clamp(y_b+1, 1, N)))
    
    # determine where in cell source is
    s_x = i - x_l
    s_y = j - y_b

    ## hash random gradient for each grid corner
    bl_g = gradient(x_l,y_b, rand, speed)
    br_g = gradient(x_r,y_b, rand, speed)
    tr_g = gradient(x_r,y_t, rand, speed)
    tl_g = gradient(x_l,y_t, rand, speed)

    dot_bl = ti.Vector([s_x,s_y]).dot(bl_g)
    dot_br = ti.Vector([s_x-1,s_y]).dot(br_g)
    dot_tl = ti.Vector([s_x,1-s_y]).dot(tl_g)
    dot_tr = ti.Vector([s_x-1,1-s_y]).dot(tr_g)

    
    ## create fade coefficients
    x_fade = fade(s_x)
    y_fade = fade(s_y)

    ##noise value
    n = bilerp(dot_bl, dot_br, dot_tr, dot_tl, x_fade, y_fade)
    
    #print(n)
    return n

@ti.kernel
def vel_debug():
    # skipping grid points to only make N/step arrows_array
    nx= N // step
    ny= N // step

    for ni, nj in ti.ndrange((0,nx),(0,ny)):

        #real coords
        i = ni*step
        j = nj*step

        #norm coords
        norm_i = i/N
        norm_j = j/N
        
        #val values
        vel_x = u[i,j]
        vel_y = v[i,j]

        #calc mag
        mag = ti.sqrt(vel_x*vel_x + vel_y*vel_y) + 1e-5

        norm_vx = vel_x/mag/nx
        norm_vy = vel_y/mag/nx

        index = (ni * ny + nj)
        line_index = index * 2
        arrow_index = index * 3
       
        arrows_array[line_index] = ti.Vector([norm_i,norm_j])
        arrows_array[line_index+1] = ti.Vector([norm_i+norm_vx, norm_j+norm_vy])  

        triangles[arrow_index] = ti.Vector([norm_i+norm_vx, norm_j+norm_vy]) 

        ## arrow left tip
        # tip point - 0.1 of the direction vector to get there, plus this same distance, rotated 90 degree to the left

        ## arrow right tip
        triangles[arrow_index+1] = triangles[arrow_index] -  ti.Vector([norm_vx, norm_vy]) * 0.1 + ti.Vector([-norm_vy, norm_vx])*0.1
        triangles[arrow_index+2] = triangles[arrow_index] -  ti.Vector([norm_vx, norm_vy]) * 0.1 + ti.Vector([norm_vy, -norm_vx])*0.1

        









###########
## FORCES 
###########



## ADD FORCE
    #NOTES:
    #w1(x) = w0(x) + delta T * f(x,t)
    #frame 2 vel = frame 1 vel + the force * time interval
@ti.kernel
def gravity():
    for I in ti.grouped(v):

        v[I] = v[I] + gravity * unit

@ti.kernel
def up(up_force: ti.f32):
    for I in ti.grouped(v):
        v[I] = v[I] + up_force * dt * N * density[I]
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
            u[i,j] = (u_prev[i,j] + a*(u[i-1,j]+u[i,j-1]+ u[i+1,j] + u[i,j+1]))/(1+4*a) * decay
            v[i,j] = (v_prev[i,j] + a*(v[i-1,j]+v[i,j-1]+ v[i+1,j] + v[i,j+1]))/(1+4*a) * decay
        
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
def calc_noise(t: ti.f32, freq: ti.f32, amp: ti.f32, speed: ti.f32):
    for i, j in ti.ndrange((1,N-1),(1,N-1)):
        #scalar noise
        #noise_scalar[i, j] = amp * noise(i * freq, j * freq, t * 0.2, 0)
        noise_scalar[i, j] = 3 * amp * p_noise_calc(i * freq, j * freq, int(t), speed)
        ##print(noise_scalar[i,j])
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
        #print(curl_vector)
        #print(u[i,j])

        u[i,j] = u[i,j] + curl_vector[0] * strength
        v[i,j] = v[i,j] + curl_vector[1] * strength

        # u[i,j] = curl_vector[0] * strength
        # v[i,j] = curl_vector[1] * strength
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



def substep(params):
    global u, v, u_prev, v_prev, density, density_prev, current_t, substep_count


    ## GRAVITY
    ##force() # gravity
    ##copy_field(v_prev, v)

    if up_force : 
        up(params['bouyancy_mult'])
  
        v_prev, v = v, v_prev
        u_prev, u = u, u_prev
   
    ## ADVECT
    advect()
    v_prev, v = v, v_prev
    u_prev, u = u, u_prev
    density_prev, density = density, density_prev
    

    # DIFFUSE
    diffuse_dens()
    density_prev, density = density, density_prev
    diffuse_vel()
    v_prev, v = v, v_prev
    u_prev, u = u, u_prev
   
        ##TURBLUENCE
    if params['turbulence']: 
        
        calc_noise(substep_count, freq=params['turb_freq'], amp=params['turb_amp'], speed=params['turb_speed'])
        apply_turbulence(strength=params['turb_strength'], amp=params['turb_amp'])
        v_prev, v = v, v_prev
        u_prev, u = u, u_prev


    ## CURL
    compute_curl()
    #v_prev, v = v, v_prev
    #u_prev, u = u, u_prev


    

    ## PROJECT
    project()
    v_prev, v = v, v_prev
    u_prev, u = u, u_prev
    density_prev, density = density, density_prev

    if params['arrows']:
        vel_debug()

    current_t = current_t + dt/substeps
    substep_count = substep_count + 1
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
canvas.set_background_color((0,0,0))
gui = window.get_gui()



##Initialize fields
initialize_fields()
add_burst(params['turb_freq'])
copy_field(density_prev, density)
copy_field(u_prev, u)
copy_field(v_prev, v)  

##Render loop
while window.running:
    window.show()

    gui.begin("Controls", 0.02, 0.02, 0.3, 0.4)

    # sliders
    params["bouyancy_mult"] = gui.slider_float("Buoyancy", params["bouyancy_mult"], 0.0, 100.0)
    params["source_radius"] = gui.slider_float("Source Radius", params["source_radius"], 0.001, 0.2)
    params["turb_freq"] = gui.slider_float("Noise Freq", params["turb_freq"], 0.001, 0.2)
    params["turb_amp"]  = gui.slider_float("Noise Amp", params["turb_amp"], 0.0, 3.0)
    params["turb_strength"]  = gui.slider_float("Noise Strength", params["turb_strength"], 0.01, 50.0)
    params["turb_speed"]  = gui.slider_float("Noise Speed", params["turb_speed"], 0.00, 10.0)

    # checkbox
    params["turbulence"] = gui.checkbox("Turbulence", params["turbulence"])
    params["arrows"] = gui.checkbox("Display Vel Arrows", params["arrows"])
    params["noise_display"] = gui.checkbox("Display Noise Field", params["noise_display"])


    # buttons
    if gui.button("Reset"):
       initialize_fields()
    if gui.button("Burst"):
        add_burst(params['turb_freq'])

    gui.end()

    # calculate all substeps and then update
    for _ in range(substeps):
        substep(params)
        

    ## check for mouse clicks
    if window.is_pressed(ti.ui.LMB):
        x,y = window.get_cursor_pos()
        x = int(x*N)
        y = int(y*N)


        ## checking for previous click to add movement vectors
        if prev_mouse is not None:
            px, py = prev_mouse
            vx = int((x - px) * 100)   # scale sensitivity
            vy = int((y - py) * 100)
            
            add_source(x,y,vx,vy, params['source_radius'], params['turb_freq'], params['turb_speed'])

        else:
            print("adding source")
            add_source(x,y,0,0, params['source_radius'], params['turb_freq'], params['turb_speed'])


        prev_mouse = (x, y)
    else:
        #reset
        prev_mouse = None


    make_display_image()
    current_t += 0.01
    canvas.set_image(density)
    #displays noise field
    if params['noise_display'] and params['turbulence']:
        canvas.set_image(noise_scalar)
    #displays velocity arrows
    if params['arrows']:
        canvas.lines(arrows_array, width=0.001, color=(1.0,1.0,1.0))
        canvas.triangles(triangles, color=(1.0,1.0,1.0))