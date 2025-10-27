import taichi as ti
from taichi.math import sin, cos, pi


### CLOTH SIMULATION ### TAICHI LANG #### https://docs.taichi-lang.org/docs/cloth_simulation



##### CONFIG #####

ti.init(arch=ti.metal)

# POINTS have 2 arrays:
    #position
    #velocity

n = 150
##time step
dt = 2e-2 / n
## set sunsteps per timestep
substeps = int(1 / 60 / dt)
#x=positions
x = ti.Vector.field(3, dtype=float, shape=(n,n))
#v=velocities
v = ti.Vector.field(3,dtype=float, shape=(n,n))

## n x n grid is normalized to 1 unit along each side, so each quad edge is 1/n, the number of quads along that side
quad_size = 1.5/n

## BALL
ball_radius = 0.3
## ball_center field, vec3 and the field has only one slot
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
##put ball at origin
ball_center[0] = [0,.25,0]

## creative vertices for rendering
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

## CLOTH AFFECTED BY 4 FACTORS: gravity, elastic spring force, damping, col with ball
# gravity

gravity = ti.Vector([0,-5.8,0])

#elastic coefficient of springs
#Young's modulus: stiffness
spring_Y = 3e4
#damping coefficient, at most spring can have 12 influential points
dashpot_damping = 1e4
##create drag
drag_damping = 1

#spring offsets is a list of influential points
spring_offsets = []

bending_springs = False



#### INITIALIZE CLOTH #####

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)




@ti.kernel
def initialize_mass_points():
    ## effectively creates a random vector offset that can be added to point positions, between (0.05 x 0.05) and (-0.05 x -0.05)
    random_offset = ti.Vector([ti.random(float) - 0.5, ti.random(float) - 0.5]) * 0.1

    # creating cloth array of points, y is constant at 0.6 floating right above where ball will be, i and j are x and z point positions
    # subtracting 0.5 because grid size is normalized to 1
    for i, j in x:
        x[i,j] = [
            i * quad_size - 0.75 + random_offset[0], #random_offset[0] is like v@P.x
            0.6,                                    #constant 0.6
            j * quad_size - 0.75 + random_offset[1]
        ]
    
    #set vel for each point to 0
    v[i,j] = [0,0,0]

## reset velocities properly
@ti.kernel
def reset_velocities():
    for I in ti.grouped(v):
        v[I] = ti.Vector([0.0, 0.0, 0.0])




#### FORCES #####

## create list of neighbouring point indices, relative to whichever point is being looped over
spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))


@ti.kernel
def substep():
    ## ti.grouped loops over every point in cloth in 1d array, although each point is in x,y vector form
    ## i is the absolute index relative to this ti.grouped method
    
    ## FIRST we add gravity to vel for each point. scale by timestep
    for i in ti.grouped(x):
        v[i] += gravity * dt

    ## THEN we calculate spring force and dampening
    for i in ti.grouped(x):
        #intializing force of 0 per point every step
        force = ti.Vector([0.0,0.0,0.0])
        ## add gravity effect
        

        #traverse the surrounding points for each point using spring_offsets list
        for spring_offset in ti.static(spring_offsets):
            ## j is the neighbouring point we're examining right now: i + the offset vector in the spring_offsets list we made earlier
            j = i + spring_offset

            ## if point is in cloth domain - between 0 and n inclusive - it has an effect on the point:
            if 0 <= j[0] < n and 0 <= j[1] < n:
                
                ## relative difference between points
                x_ij = x[i] - x[j]

                ## relative difference between velocities
                v_ij = v[i] - v[j]

                ## dir vector = normalized unit vector difference between positions
                dir = x_ij.normalized()

                ## calculate length of distance between the two points (.norm() = .length() in vex)
                current_dist = x_ij.norm()

                ## rest pos distance (quad size * ((0,0) - (1,0)).length )
                original_dist = quad_size * float(i-j).norm()

                ##adding spring force, hooke's law:
                ## spring_Y = stiffness
                ## current_dist/original_dist - 1 = the strain (stretch or compression)
                ## -dir pushes points together if distance > rest, and away if distance < rest
                force += spring_Y * -dir * (current_dist / original_dist - 1)
                
                ## add spring damping force
                ##v_ij.dot(dir) = how much vel is pushing in spring direction
                # * -dir vector magnitude of this force in correct direction 
                # *dashpot damping = dampening coefficient
                # quad size for relative length between two points of the cloth
                force += v_ij.dot(dir) * -dir * dashpot_damping * quad_size
        v[i] += force * dt

    ##NEXT we add drag
    for i in ti.grouped(x):
        ## simulating velocity diffusion over each frame
        v[i] *= ti.exp(-drag_damping * dt)
    
    
    ##FINALLY we add col force
    for i in ti.grouped(x):
        ## calculate distance from ball center point
        offset_to_center = x[i] - ball_center[0]
        ## check if this is less than radius of ball
        if offset_to_center.norm() <= ball_radius:
            ## calculate collision vector, normalized
            normal = offset_to_center.normalized()
            ## calculate the dot product between the point vel and the collision normal, if negative use that
            ## point is actively penetrating colision
            ## if positive point is touching but moving away, so default to 0
            ## multiply by the collision normal
            
            v[i] -= min(v[i].dot(normal),0)*normal
            ## add this collision velocity to the point, scaled by time step
            ## basically just ensures that point will not pentrate collision object
            x[i] = ball_center[0] + normal * ball_radius
        
        ##update positions
        x[i] += v[i] * dt

        
### TEST WITH SPHERE COL INSTEAD OF PARTICLE COL - COLLISION ACCURACY MUCH BETTER

# Simple procedural sphere generator
def make_sphere(res=40):
    vertices = []
    indices = []
    for i in range(res + 1):
        theta = pi * i / res
        for j in range(res * 2 + 1):
            phi = 2 * pi * j / (res * 2)
            x = sin(theta) * cos(phi)
            y = cos(theta)
            z = sin(theta) * sin(phi)
            vertices.append([x, y, z])
            if i < res and j < res * 2:
                a = i * (res * 2 + 1) + j
                b = a + (res * 2 + 1)
                indices += [a, b, a + 1, b, b + 1, a + 1]
    return ti.Vector.field(3, dtype=float, shape=len(vertices)), \
           ti.field(int, shape=len(indices)), \
           vertices, indices

# Create the sphere fields and fill them
sphere_vertices, sphere_indices, verts, inds = make_sphere()
for i in range(len(verts)):
    sphere_vertices[i] = verts[i]
for i in range(len(inds)):
    sphere_indices[i] = inds[i]

# Keep a CPU copy of the unit sphere vertices
verts_buf = ti.Vector.field(3, dtype=float, shape=len(verts))
for i in range(len(verts)):
    verts_buf[i] = verts[i]


@ti.kernel
def update_sphere_vertices(center: ti.types.vector(3, float), radius: float):
    for i in range(sphere_vertices.shape[0]):
        sphere_vertices[i] = verts_buf[i] * radius + center

        


#### RENDER ######

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


## Display logic

window = ti.ui.Window("Cloth Sim Tutorial - Steve Cutler", (1024, 1024), vsync=True)
canvas = window.get_canvas()
#set to white
canvas.set_background_color((1,1,1))
#create scene and camera
scene = window.get_scene()
camera = ti.ui.Camera()
#start time
current_t = 0.0

#intialize cloth
initialize_mesh_indices()
initialize_mass_points()

##Render loop
while window.running:
    #reset every 1.5 seconds
    if current_t > 2.3:
        initialize_mass_points()
        reset_velocities()
        current_t=0
        
    ## calculate all substeps and then update vertices
    for i in range(substeps):  
        substep()
        current_t += dt    
    update_vertices()

    camera.position(0,0,3)
    camera.lookat(0,0,0)
    scene.set_camera(camera)

    scene.point_light(pos=(0,1,2), color=(1,1,1))
    scene.ambient_light((0.5,0.5,0.5))
    scene.mesh(vertices,
               indices,
               per_vertex_color=colors,
               two_sided=True)
    
     # Draw a smaller ball to avoid visual penetration
    
    ##scene.particles(ball_center, radius=ball_radius*0.9, color=(0.5, 0.42, 0.8))

    #test with sphere mesh:
    update_sphere_vertices(ball_center[0], ball_radius)
    scene.mesh(
        sphere_vertices,
        sphere_indices,
        color=(0.5, 0.42, 0.8),
        two_sided=True,
    )

    canvas.scene(scene)
    window.show()


