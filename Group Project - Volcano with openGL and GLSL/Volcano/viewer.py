#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""

import sys                          # for system arguments

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np                  # all matrix manipulations & OpenGL args
import glfw                         # lean window system wrapper for OpenGL

from core import Shader, Mesh, Viewer, Node, load, RotationControlNode, TranslateControlNode, Cylinder, RotationSpeedNode, Sphere
from transform import translate, identity, rotate, scale
from transform import (lerp, quaternion_slerp, quaternion_matrix, translate,
                       scale, identity, vec, quaternion, quaternion_from_euler, quaternion_from_axis_angle)
from animation import KeyFrameControlNode, KeyFrames
from texture import Texture, Textured
from itertools import cycle

# -------------- Example textured plane class ---------------------------------
class TexturedTriangle(Textured):
    """ Simple first textured object """
    def __init__(self, shader, coords, tex_file, uniforms):
        assert coords.shape[0] == 3
        # prepare texture modes cycling variables for interactive toggling
        self.wraps = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                            GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filters = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                              (GL.GL_LINEAR, GL.GL_LINEAR),
                              (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap, self.filter = next(self.wraps), next(self.filters)
        self.file = tex_file

        # setup plane mesh to be textured
        #base_coords = ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
        #scaled = 100 * np.array(base_coords, np.float32)
        indices = np.array((0, 1, 2), np.uint32)
        tex_coord = np.array(((0.5,0),(1,1),(0,1)), np.float32)
        mesh = Mesh(shader, attributes=dict(position=coords, tex_coord = tex_coord), index=indices)

        # setup & upload texture to GPU, bind it to shader name 'diffuse_map'
        texture = Texture(tex_file, self.wrap, *self.filter)
        super().__init__(mesh, uniforms, diffuse_map=texture)

    def key_handler(self, key):
        # cycle through texture modes on keypress of F6 (wrap) or F7 (filtering)
        self.wrap = next(self.wraps) if key == glfw.KEY_F6 else self.wrap
        self.filter = next(self.filters) if key == glfw.KEY_F7 else self.filter
        if key in (glfw.KEY_F6, glfw.KEY_F7):
            texture = Texture(self.file, self.wrap, *self.filter)
            self.textures.update(diffuse_map=texture)

class TexturedPlane(Textured):
    """ Simple first textured object """
    def __init__(self, shader, tex_file, uniforms):
        # prepare texture modes cycling variables for interactive toggling
        self.wraps = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                            GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filters = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                              (GL.GL_LINEAR, GL.GL_LINEAR),
                              (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap, self.filter = next(self.wraps), next(self.filters)
        self.file = tex_file

        # setup plane mesh to be textured
        base_coords = ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
        scaled = np.array(base_coords, np.float32)/10.
        indices = np.array((0, 1, 2, 0, 2, 3), np.uint32)
        tex_coord = np.array(((1,1),(0,1),(0,0),(1,0)), np.float32)
        mesh = Mesh(shader, attributes=dict(position=scaled, tex_coord = tex_coord), index=indices)

        # setup & upload texture to GPU, bind it to shader name 'diffuse_map'
        texture = Texture(tex_file, self.wrap, *self.filter)
        super().__init__(mesh, uniforms, diffuse_map=texture)


class Triangle(Mesh):
    """Hello triangle object"""
    def __init__(self, shader, uniforms = dict()):
        position = np.array(((0, -0.5, 0), (0, .5, 0), (1, -0.5, 0), (0, -0.5, 0), (1, -0.5, 0), (0, .5, 0)), 'f')
        color = np.array(((1, 1, 1), (1, 1, 1), (1, 1, 1)), 'f')
        self.color = (1, 1, 1)
        attributes = dict(position=position, color=color)
        super().__init__(shader, attributes=attributes, **uniforms)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.color, **uniforms)

    def key_handler(self, key):
        if key == glfw.KEY_C:
            self.color = (0, 0, 0)

class SquareGrid(Mesh):
    def __init__(self, shader, position, index, normals, uniforms, textures = dict()):
        self.position = position
        self.index = index
        self.normals = normals

        self.textures = textures

        attributes = dict(position=self.position, normal=self.normals)                            
        super().__init__(shader, attributes=attributes, index=self.index, usage=GL.GL_STATIC_DRAW, **uniforms)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        for index, (name, texture) in enumerate(self.textures.items()):
            GL.glActiveTexture(GL.GL_TEXTURE0 + index)
            GL.glBindTexture(texture.type, texture.glid)
            uniforms[name] = index
        super().draw(primitives=primitives, **uniforms)

def random_projections(N, uniforms):
    ''' Generates N random projections and integrates their positions over time '''
    nodes = []
    for i in range(N):
        # Random initial time
        t = np.random.rand() * 10

        # Initial position
        pos = np.array((0.,0.5,0.), np.float32)
        # Initial angle in XZ plane
        angle = 2 * np.pi * np.random.rand()

        # Initial direction vector
        d = np.array((0.3*np.cos(angle), 1, 0.3*np.sin(angle)), np.float32)
        d /= np.linalg.norm(d, 2)

        # Initial velocity vector
        v = (np.random.rand() + 2.) * d

        # Step for numerical integration
        step = 0.1

        # Gravity
        g = np.array((0,-1,0), np.float32)

        # Scale and scale decrease per second
        scale = 0.1
        delta = 0.01

        # Identity quaternion
        Id = quaternion()

        # First keyframe
        translate_keys = {t: pos}
        rotate_keys = {t: Id}
        scale_keys = {t: scale}

        # Integrate until they've fallen under the terrain
        while pos[1] > -1:
            # Step in time
            t += step

            # Update position
            pos = pos + v * step

            # Update speed
            v += g * step
            
            # Update scale
            scale -= delta * step

            # Insert transforms values
            translate_keys[t] = pos
            rotate_keys[t] = Id
            scale_keys[t] = scale

        # Projection node
        node = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys, uniforms)

        # All projections
        nodes.append(node)
    return nodes

# Inflating bubble keyframes
def KeyFrameBubble(pos, scale, uniforms):
    t_ini = np.random.rand() * 3
    Id = quaternion()

    translate_keys = {t_ini: pos}
    rotate_keys = {t_ini: Id}
    scale_keys = {t_ini: 0}

    max_t = scale*10 + 2

    for t in np.arange(t_ini + 1, t_ini + max_t, 1):
        translate_keys[t] = pos
        rotate_keys[t] = Id
        scale_keys[t] = scale * max_t/5

    #print(translate_keys)
    #print(scale_keys)

    return KeyFrameControlNode(translate_keys, rotate_keys, scale_keys, uniforms)

# Helicopter blade rotation
def KeyFrameBlade():
    translate_keys = {0: 0}
    rotate_keys = {0: quaternion()}
    scale_keys = {0: 1}

    for t in np.arange(0.1,1,0.1):
        translate_keys[t] = 0
        rotate_keys[t] = quaternion_from_axis_angle(axis = vec(0,1,0), degrees = 2*360/t) 
        scale_keys[t] = 1

    return KeyFrameControlNode(translate_keys, rotate_keys, scale_keys)

# Square grid generator
def grid(size, N):
    ''' Builds the position, index and normals array for a square grid of dimensions N*N'''
    X = np.linspace(-size/2, size/2, N, endpoint = True)
    Z = np.copy(X)

    #position = np.zeros((N*N, 3), 'f')
    position = np.array([(x, 0, z) for x in X for z in Z], np.float32)

    index = np.array([(i + j*N, i+1 + j*N, i + (j+1)*N, i+1 + j*N, i+1 + (j+1)*N, i + (j+1) * N) for i in range(N-1) for j in range(N-1)], np.uint32)
    #self.index = self.index.reshape((self.index.size, 1))
    index.flatten('C')

    normals = np.array([(0,1,0) for i in range(index.size)])

    return position, index, normals

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()
    shader = Shader("volcano.vert", "volcano.frag")

    # Terrain size and number of points along each axis
    size = 10
    N = 1000
    mesh_arrays = grid(size, N) # Generate the grid once, reuse for different elements

    # Surface types, passed as uniforms
    terrain_type = 0
    water_type = 1
    lava_type = 2
    projection_type = 3
    smoke_type = 4
    bubble_type = 5
    cloud_type = 6
    wood_type = 7
    cactus_type = 8
    skybox_type = -1

    # Surface materials and textures
    terrain_material = {"k_a" : np.array((0.,0.,0.)),
                        "k_d" : np.array((0.6,0.5,0.4)),
                        "k_s" : np.array((0.,0,0)),
                        "s" : 20,
                        "surface_type" : terrain_type}
    terrain_textures = {"grass" : Texture("textures/grass.png"),
                        "rock" : Texture("textures/rock.png"),
                        "sand" : Texture("textures/sand.jpg"),
                        "normalmap": Texture("textures/normalmap.jpg")}
    
    water_material =   {"k_a" : np.array((0.,0.,0.)),
                        "k_d" : np.array((0.1,0.1,0.4)),
                        "k_s" : np.array((0.8,0.8,0.8)),
                        "s" : 200,
                        "surface_type" : water_type}
    water_textures = {"normalmap" : Texture("textures/normalmap.jpg")}
    
    lava_material =    {"k_a" : np.array((0.,0.,0.)),
                        "k_d" : np.array((0.1,0.1,0.1)),
                        "k_s" : np.array((0.1,0.,0.)),
                        "s" : 100,
                        "surface_type" : lava_type}
    
    cloud_material =    {"k_a" : np.array((0.05,0.05,0.05)),
                        "k_d" : np.array((0.5,0.5,0.7)),
                        "k_s" : np.array((0.,0.,0.)),
                        "s" : 100,
                        "surface_type" : cloud_type}    
    
    wood_material = {"k_a" : np.array((0.,0.,0.)),
                        "k_d" : np.array((0.2,0.1,0.)),
                        "k_s" : np.array((0.,0.,0.)),
                        "s" : 100,
                        "surface_type" : wood_type}  
    wood_textures = {"wood" : Texture("textures/wood.png")}

    cactus_material = {"k_a" : np.array((0.,0.,0.)),
                        "k_d" : np.array((0.07,0.41,0.07)),
                        "k_s" : np.array((0.2,0.1,0.1)),
                        "s" : 30,
                        "surface_type" : cactus_type}  

    projection_material = lava_material.copy()
    projection_material["k_a"] = np.array((1,0,0))
    projection_material["surface_type"] = projection_type
    
    smoke_material =    {"k_a" : np.array((0.,0.,0.)),
                        #"k_d" : np.array((0.,0.,0.)),
                        #"k_s" : np.array((0.,0.,0.)),
                        #"s" : 1,
                         "surface_type": smoke_type}


    # Constant uniforms    
    viewer.uniforms = {"terrain_offset" : -0.5,
                       "water_offset" : 0,
                       "lava_offset" : 0,
                       "sun_axis" : vec(0.,1.,-0.5),
                       "terrain_type" : terrain_type,
                       "water_type" : water_type,
                       "lava_type" : lava_type,
                       "bubble_type" : bubble_type,
                       "cloud_type" : cloud_type,
                       "projection_type" : projection_type,
                       "smoke_type" : smoke_type,
                       "skybox_type" : skybox_type,
                       "wood_type" : wood_type,
                       "cactus_type" : cactus_type
                        }
  
                        
                       #"perlin" : perlin_noise}

    # Skybox
    viewer.add(*[mesh for mesh in load("skybox/cube.obj", shader, surface_type = skybox_type)])

    # Terrain
    plane = SquareGrid(shader, *mesh_arrays, terrain_material, terrain_textures)
    viewer.add(plane)

    # Lava
    lava = SquareGrid(shader, *mesh_arrays, lava_material)
    viewer.add(lava)

    # Water
    water = SquareGrid(shader, *mesh_arrays, water_material, water_textures)
    viewer.add(water)

    # Lava projections
    keyframe_nodes = random_projections(20, projection_material)
    for node in keyframe_nodes:
        node.add(Sphere(shader, radius = 0.3))
        viewer.add(node)

    # Log boat
    cylinder = Cylinder(shader, scale = 1, uniforms = wood_material)#, textures = wood_textures)
    centered_cylinder = Node((cylinder,), transform = scale(0.05,1,0.05) @ translate(0,-0.5,0))
    logs = [Node((centered_cylinder,), transform = translate(0.1*(i - 3), 0, 0) @ rotate((1,0,0), angle=90)) for i in range(7)]
    mast = Node((centered_cylinder,), transform = scale(0.5,1,0.5)@translate(0,0.5,0))
    body = Node(logs)
    triangle = Triangle(shader, cloud_material)
    square = TexturedPlane(shader, tex_file = "textures/flag.png", uniforms = cloud_material)
    sail = Node((triangle,), transform=rotate((0,1,0), 90) @ scale(0.5,0.7,1) @ translate(0,0.7,0))
    flag = Node((square,), transform=rotate((0,1,0), 90)  @ translate(0.1,0.9,0))
    boat = Node((body, mast, sail, flag), transform=translate(3,0.03,3) @ scale(0.3,0.3,0.3))
    viewer.add(boat)

        # Cacti
    cylinder2 = Cylinder(shader,scale = 1,uniforms=cactus_material)

    trunk = Node(transform = scale (0.02,0.6,0.02)@translate(0,0,0))
    trunk.add(cylinder2)
    trunk_arm = Node(transform = scale (0.012,0.08,0.012)@translate(0,0,0))
    trunk_arm.add(cylinder2)

    trunk_arm2 = Node(transform = scale(0.01,0.06,0.01)@translate(0,0,0))
    trunk_arm2.add(cylinder2)

    trunk_arm22 = Node(transform = scale(0.009,0.07,0.009)@translate(0,0,0))
    trunk_arm22.add(cylinder2)


    theta = 45.0
    phi1 = 20.0

    transform_arm22 = Node(transform = translate(0,0.05,0)@rotate((1,0,0),phi1))
    transform_arm22.add(trunk_arm22)

    transform_arm2 = Node(transform = translate(0,0.4,0)@rotate((1,0,0),(-theta)))
    transform_arm2.add(trunk_arm2,transform_arm22)

    transform_arm = Node(transform = translate(0,0.5,0)@rotate((1,0,0),(theta)))
    transform_arm.add(trunk_arm)

    transform_trunk = Node(transform = translate(0,0,0)@rotate((1,0,0),(0)))
    transform_trunk.add(trunk, transform_arm,transform_arm2)

    for i in range(10):
        transform_trunk = Node(transform = translate((rotate((0,1,0),radians=np.random.rand() * 2 * np.pi) @ vec(np.random.rand() * 0.5 + 1.5, 0, 0, 1))[:3])@rotate((0,1,0),radians=np.random.rand() * 2 * np.pi))
        transform_trunk.add(trunk, transform_arm,transform_arm2)
        viewer.add(transform_trunk)

 

    # Smoke billboard
    smoke = TexturedTriangle(shader, np.array(((0.,0, 0), (4., 5, 0), (-4., 5, 0))) + np.array((0,0.5,0)), "textures/grass.png", smoke_material)
    viewer.add(smoke)

    controls = """
    Controls: 
    left click and move the mouse to rotate the camera
    scroll to zoom in/out
    'ESCAPE' quits
    'SPACE' pauses
    'W' changes the draw mode
    'RIGHT'/'LEFT' changes the fog density
    'UP'/'DOWN' steps time manually
    'F5' shows/hides the cacti
    """
    print(controls)    

    # start rendering loop
    viewer.run()




if __name__ == '__main__':
    main()                     # main function keeps variables locally scoped
