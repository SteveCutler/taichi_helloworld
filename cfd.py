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
gravity = -9.8

u_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step x velocity field
v_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step y velocity field
density_prev = ti.field(dtype=ti.f32, shape=(N,N)) # prev step density field

dt = 1/48 #timestep

diff = 0.0001 #diffusion coefficient

## potential to add viscosity here


###########
## HELPERS
###########

@ti.kernel
def initialize_fields():
    
    for i,j in density:
        
        ## initialize fields
        u[i,j] = 1.5
        v[i, j] = 3
       
        density[i, j] = ti.sin(i/N*1.0) * ti.cos(j/N*1.0)
        
        ## duplicate for previous frame fields
        u_prev[i,j] = 1.5
        v_prev[i,j] = 3
        density_prev[i,j] = ti.sin(i/N*1.0) * ti.cos(j/N*1.0)
        
        
@ti.Kernel
def copy_field(field1: ti.template(), field2: ti.template()):
    for I in ti.grouped(field1):
        field1[I] = field2[I]


###########
## FORCES 
###########



## ADD FORCE
    #NOTES:
    #w1(x) = w0(x) + delta T * f(x,t)
    #frame 2 vel = frame 1 vel + the force * time interval
@ti.kernel
def force():
    for I in density:
        ## adding gravity force * units per timestep to vertical velocity
        v[I] = v[I] + gravity * unit *dt
    
    

## ADVECT
    #NOTES:
    #particle path =  p(x, s)
    #w1(x) = w0(p(x0, -dt))
    #basically frame 2 particle vel = tracing vel of particle backwards to previous time step, 
    # sampling vel from that x,y coordinate, interpolate
    # multiply by some diminishing fraction so doesn't blow up? * 0.95?
@ti.kernel
def advect(u, v, density, u_prev, v_prev, density_prev, N, dt):
    ##grab the vertical and horizontal vector from the previous frame
    for i, j in density:

        ## velocity path at this point in units per timestep
        u_path = u_prev[i,j] * dt * unit
        v_path = v_prev[i,j] * dt * unit
        
        ## subtracting vel from current position to get where it was in previous position
        x_old = i - u_path
        y_old = j - v_path

        ## calculate corners with boundary enforcement
        x_l = ti.max(0, ti.min(int(ti.floor(x_old)), N-1))
        x_r = ti.max(0, ti.min(int(x_l+1), N-1))
        y_b = ti.max(0, ti.min(int(ti.floor(y_old)), N-1))
        y_t = ti.max(0, ti.min(int(y_b+1), N-1))
        
        # determine where in cell source is
        s_x = x_old - x_l
        s_y = y_old - y_b

        ## corner samples
        bl_d = density_prev[x_l,y_b]
        br_d = density_prev[x_r,y_b]
        tl_d = density_prev[x_l,y_t]
        tr_d = density_prev[x_r,y_t]

        bl_u = u_prev[x_l,y_b]
        br_u = u_prev[x_r,y_b]
        tl_u = u_prev[x_l,y_t]
        tr_u = u_prev[x_r,y_t]

        bl_v = v_prev[x_l,y_b]
        br_v = v_prev[x_r,y_b]
        tl_v = v_prev[x_l,y_t]
        tr_v = v_prev[x_r,y_t]

        ## bilinear interpolation
                      ## bottom_left            bottom right         top right        top left
        density[i,j] = (1-s_x)*(1-s_y)*bl_d + (s_x*(1-s_y)*br_d) + s_x*s_y*tr_d + (1-s_x)*s_y*tl_d
        u[i,j] = (1-s_x)*(1-s_y)*bl_u + (s_x*(1-s_y)*br_u) + s_x*s_y*tr_u + (1-s_x)*s_y*tl_u
        v[i,j] = (1-s_x)*(1-s_y)*bl_v + (s_x*(1-s_y)*br_v) + s_x*s_y*tr_v + (1-s_x)*s_y*tl_v
       


## DIFFUSE
    ## NOTES: use gauss seidl relaxation, 20 iteration
    # w1(x,y) = w0(x,y) + a( w1(x-1,y) + w1(x+1,y) + w1(x,y-1) + w1(x,y+1)) / 1 + 4a
    #where a = diffusion coeff

    # gauss seidl interpolation, 20 iterations
    # calculate diffusion by averaging velocity/density for a given cell between the 4 or 8 neighbours, 
    # note: use the new values for the current iteration when available (the cells that came before)

    @ti.kernel
    def diffuse():
        ##Diffuse using gauss seidl, 20 iterations

        for i in range(20):
            for i, j in density:
                ## don't diffuse on boundaries
                if( i == 0 or i == N-1 or j == 0 or j == N-1 ):
                    continue
                density[i,j] = density_prev[i,j] + (diff*(density[i-1,j]+density[i,j-1]+ density_prev[i+1,j] + density_prev[i,j+1]))/(1+4*diff)

            copy_field(density_prev, density)
            
            ##enforce boundaries constraints with set_bnd


## PROJECT
    # poisson solver?
    # gauss seidl

## BOUNDARY CONDITIONS
    ## create set bnd helper for setting boundary pixel values per stam
    ##diffuse

## TO DO:
## min_max helper function
## gauss_seidl helper function
## set_bnd function
## create random curl velocity in initial vel fields
## render logic





@ti.kernel
def main():
    
    initialize_fields()
    force()
    advect(u, v, density, u_prev, v_prev, density_prev, N)

main()

#### RENDER ######


## Display logic

##window = ti.ui.Window("Stable Fluids - Steve Cutler", (1024, 1024), vsync=True)
# canvas = window.get_canvas()
# #set to white
# canvas.set_background_color((1,1,1))
# #create scene and camera
# ##scene = window.get_scene()
# camera = ti.ui.Camera()
# #start time
# current_t = 0.0

##Render loop
##while window.running:
   ## main()
    ##current_t += dt
    ##canvas.scene(scene)
    ##window.show()
    #reset every 1.5 seconds
    ##if current_t > 2.5:
        # initialize_fields()
        # current_t=0
        
    ## calculate all substeps and then update vertices
    # for i in range(substeps):  
    #     substep()
    #     current_t += dt    
    # update_vertices()

    # camera.position(0,0,3)
    # camera.lookat(0,0,0)
    # scene.set_camera(camera)

    # scene.point_light(pos=(0,1,2), color=(1,1,1))
    # scene.ambient_light((0.5,0.5,0.5))
    # scene.mesh(vertices,
    #            indices,
    #            per_vertex_color=colors,
    #            two_sided=True)
    
    #  # Draw a smaller ball to avoid visual penetration
    
    # ##scene.particles(ball_center, radius=ball_radius*0.9, color=(0.5, 0.42, 0.8))

    # #test with sphere mesh:
    # update_sphere_vertices(ball_center[0], ball_radius)
    # scene.mesh(
    #     sphere_vertices,
    #     sphere_indices,
    #     color=(0.5, 0.42, 0.8),
    #     two_sided=True,
    # )


