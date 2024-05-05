# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle         # allows easy circular choice list
import atexit                       # launch a function at exit

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import assimpcy                     # 3D resource loader

# our transform functions
from transform import Trackball, identity, rotate, translate, vec, ortho, camera_offset_from_mouse_pos, lookat, perspective, quaternion_from_axis_angle, quaternion_matrix, normalized

# initialize and automatically terminate glfw on exit
glfw.init()
atexit.register(glfw.terminate)


# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            os._exit(1)
        return shader

    def __init__(self, vertex_source, fragment_source, debug=False):
        """ Shader can be initialized with raw strings or source file names """
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                os._exit(1)

        # get location, size & type for uniform variables using GL introspection
        self.uniforms = {}
        self.debug = debug
        get_name = {int(k): str(k).split()[0] for k in self.GL_SETTERS.keys()}
        for var in range(GL.glGetProgramiv(self.glid, GL.GL_ACTIVE_UNIFORMS)):
            name, size, type_ = GL.glGetActiveUniform(self.glid, var)
            name = name.decode().split('[')[0]   # remove array characterization
            args = [GL.glGetUniformLocation(self.glid, name), size]
            # add transpose=True as argument for matrix types
            if type_ in {GL.GL_FLOAT_MAT2, GL.GL_FLOAT_MAT3, GL.GL_FLOAT_MAT4}:
                args.append(True)
            if debug:
                call = self.GL_SETTERS[type_].__name__
                print(f'uniform {get_name[type_]} {name}: {call}{tuple(args)}')
            self.uniforms[name] = (self.GL_SETTERS[type_], args)

    def set_uniforms(self, uniforms):
        """ set only uniform variables that are known to shader """
        for name in uniforms.keys() & self.uniforms.keys():
            set_uniform, args = self.uniforms[name]
            set_uniform(*args, uniforms[name])

    def __del__(self):
        GL.glDeleteProgram(self.glid)  # object dies => destroy GL object

    GL_SETTERS = {
        GL.GL_UNSIGNED_INT:      GL.glUniform1uiv,
        GL.GL_UNSIGNED_INT_VEC2: GL.glUniform2uiv,
        GL.GL_UNSIGNED_INT_VEC3: GL.glUniform3uiv,
        GL.GL_UNSIGNED_INT_VEC4: GL.glUniform4uiv,
        GL.GL_FLOAT:      GL.glUniform1fv, GL.GL_FLOAT_VEC2:   GL.glUniform2fv,
        GL.GL_FLOAT_VEC3: GL.glUniform3fv, GL.GL_FLOAT_VEC4:   GL.glUniform4fv,
        GL.GL_INT:        GL.glUniform1iv, GL.GL_INT_VEC2:     GL.glUniform2iv,
        GL.GL_INT_VEC3:   GL.glUniform3iv, GL.GL_INT_VEC4:     GL.glUniform4iv,
        GL.GL_SAMPLER_1D: GL.glUniform1iv, GL.GL_SAMPLER_2D:   GL.glUniform1iv,
        GL.GL_SAMPLER_3D: GL.glUniform1iv, GL.GL_SAMPLER_CUBE: GL.glUniform1iv,
        GL.GL_FLOAT_MAT2: GL.glUniformMatrix2fv,
        GL.GL_FLOAT_MAT3: GL.glUniformMatrix3fv,
        GL.GL_FLOAT_MAT4: GL.glUniformMatrix4fv,
    }


class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects."""
    def __init__(self, shader, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = {}  # we will store buffers in a named dict
        nb_primitives, size = 0, 0

        # load buffer per vertex attribute (in list with index = shader layout)
        for name, data in attributes.items():
            loc = GL.glGetAttribLocation(shader.glid, name)
            if loc >= 0:
                # bind a new vbo, upload its data to GPU, declare size and type
                self.buffers[name] = GL.glGenBuffers(1)
                data = np.array(data, np.float32, copy=False)  # ensure format
                nb_primitives, size = data.shape
                GL.glEnableVertexAttribArray(loc)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[name])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
                GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers['index'] = GL.glGenBuffers(1)
            index_buffer = np.array(index, np.int32, copy=False)  # good format
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers['index'])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

    def execute(self, primitive, attributes=None):
        """ draw a vertex array, either as direct array or indexed array """

        # optionally update the data attribute VBOs, useful for e.g. particles
        attributes = attributes or {}
        for name, data in attributes.items():
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[name])
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data)

        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), list(self.buffers.values()))


# ------------  Mesh is the core drawable -------------------------------------
class Mesh:
    """ Basic mesh class, attributes and uniforms passed as arguments """
    def __init__(self, shader, attributes, index=None,
                 usage=GL.GL_STATIC_DRAW, **uniforms):
        self.shader = shader
        self.uniforms = uniforms
        self.vertex_array = VertexArray(shader, attributes, index, usage)

    def draw(self, primitives=GL.GL_TRIANGLES, attributes=None, **uniforms):
        GL.glUseProgram(self.shader.glid)
        self.shader.set_uniforms({**self.uniforms, **uniforms})
        self.vertex_array.execute(primitives, attributes)

class Pyramid(Mesh):
    def __init__(self, shader):
        self.position = np.array(((0,1,0), (-.5,0,-.5), (.5,0,-.5), (.5,0,.5), (-.5,0,.5)), np.float32)
        self.index = np.array(((0,2,1),(0,3,2),(0,4,3),(0,1,4),(1,2,3),(3,4,1)), np.uint32)
        self.color = np.array(((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)), 'f')
        self.global_color = (1, 1, 0)
        attributes = dict(position=self.position, color=self.color)                             # EXACT SAME NAMES AS IN color.vert
        super().__init__(shader, attributes=attributes, index=self.index)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.global_color, **uniforms)

    def key_handler(self, key):
        if key == glfw.KEY_C:
            self.global_color = (0, 0, 0) if self.global_color != (0,0,0) else (1,1,0)

def spherical_cartesian(theta, phi):
    return (np.sin(theta)*np.cos(phi), np.cos(theta), np.sin(theta)*np.sin(phi))

class Sphere(Mesh):
    def __init__(self, shader, radius = 1, spokes = 20, pos = vec(0,0,0), uniforms = dict()):
        #self.uniforms = uniforms
        angle_step = np.pi/spokes
        quad_list = []
        triangles_list = []

        for theta in np.arange(angle_step, np.pi, angle_step):
            for phi in np.arange(0, 2*np.pi, angle_step):
                quad_list.append((spherical_cartesian(theta + angle_step, phi), spherical_cartesian(theta + angle_step, phi + angle_step),
                                  spherical_cartesian(theta, phi + angle_step), spherical_cartesian(theta, phi)))
 
        for quad in quad_list:
            triangles_list.append((quad[0], quad[1], quad[3]))
            triangles_list.append((quad[1], quad[2], quad[3]))

        for phi in np.arange(0, 2*np.pi, angle_step):
            triangles_list.append(((0,1,0),spherical_cartesian(angle_step, phi),spherical_cartesian(angle_step, phi + angle_step)))
            triangles_list.append(((0,-1,0),spherical_cartesian(np.pi - angle_step, phi + angle_step),spherical_cartesian(np.pi - angle_step, phi)))

        position_list = []
        index_list = []
        for i in np.arange(0, len(triangles_list)):
            position_list.append(triangles_list[i][0])
            position_list.append(triangles_list[i][1])
            position_list.append(triangles_list[i][2])
            index_list.append((i*3, i*3 + 1, i*3 + 2))

        self.position = radius * (np.array(position_list, np.float32))
        self.index = np.array(index_list, np.uint32)
        self.normals = np.array(position_list, np.float32)
        self.color = np.copy(self.normals)
        attributes = dict(position=self.position, normals = self.normals, color=self.color)                             # EXACT SAME NAMES AS IN color.vert
        super().__init__(shader, attributes=attributes, index=self.index, **uniforms)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, **uniforms)#, **self.uniforms)
        
class Cylinder(Mesh):
    def __init__(self, shader, height = 1, spokes = 20, scale = 1, uniforms = dict()):
        #self.uniforms = uniforms
        angle_step = 2*np.pi/spokes
        quad_list = []
        quad_normals_list = []
        triangles_list = []
        triangles_normals_list = []
        for i in np.arange(0, spokes):
            quad_list.append(((np.cos((i+1) * angle_step), 0, np.sin((i+1) * angle_step)), (np.cos(i * angle_step), 0, np.sin(i * angle_step)),
                              (np.cos(i * angle_step), height, np.sin(i * angle_step)), (np.cos((i+1) * angle_step), height, np.sin((i+1) * angle_step))))
            quad_normals_list.append(((np.cos((i+1) * angle_step), 0, np.sin((i+1) * angle_step)), (np.cos(i * angle_step), 0, np.sin(i * angle_step)),
                                      (np.cos(i * angle_step), 0, np.sin(i * angle_step)), (np.cos((i+1) * angle_step), 0, np.sin((i+1) * angle_step))))

        for quad in quad_list:
            triangles_list.append((quad[0], quad[1], quad[3]))
            triangles_list.append((quad[1], quad[2], quad[3]))
        
        for quad_normal in quad_normals_list:
            triangles_normals_list.append((quad_normal[0], quad_normal[1], quad_normal[2]))
            triangles_normals_list.append((quad_normal[1], quad_normal[2], quad_normal[3]))

        for i in np.arange(0, spokes):
            triangles_list.append(((0,0,0), (np.cos(i * angle_step), 0, np.sin(i * angle_step)), (np.cos((i+1) * angle_step), 0, np.sin((i+1) * angle_step))))
            triangles_list.append(((0, height, 0), (np.cos((i+1) * angle_step), height, np.sin((i+1) * angle_step)), (np.cos(i * angle_step), height, np.sin(i * angle_step))))

            triangles_normals_list.append(((0,1,0), (0,1,0), (0,1,0)))
            triangles_normals_list.append(((0,-1,0), (0,-1,0), (0,-1,0)))

        position_list = []
        index_list = []
        normals_list = []
        for i in np.arange(0, len(triangles_list)):
            position_list.append(triangles_list[i][0])
            position_list.append(triangles_list[i][1])
            position_list.append(triangles_list[i][2])
            index_list.append((i*3, i*3 + 1, i*3 + 2))
            normals_list.append(triangles_normals_list[i][0])
            normals_list.append(triangles_normals_list[i][1])
            normals_list.append(triangles_normals_list[i][2])

        self.position = np.array(position_list, np.float32) * scale
        self.index = np.array(index_list, np.uint32)
        #self.color = np.copy(self.position)
        self.global_color = (0, 0, 0)
        self.normals = np.array(normals_list, np.float32)
        #self.normals /= 2
        #self.normals += np.array((.5,.5,.5))
        self.color = np.copy(self.normals)
        attributes = dict(position=self.position, normal = self.normals, color=self.color)                             # EXACT SAME NAMES AS IN color.vert
        super().__init__(shader, attributes=attributes, index=self.index, **uniforms)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.global_color, **uniforms)#, **self.uniforms)

# ------------  Node is the core drawable for hierarchical scene graphs -------
class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, children=(), transform=identity()):
        self.transform = transform
        self.world_transform = identity()
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, model=identity(), **other_uniforms):
        """ Recursive draw, passing down updated model matrix. """
        self.world_transform = model @ self.transform; #identity()   # TODO: compute model matrix
        for child in self.children:
            child.draw(model=self.world_transform, **other_uniforms)

    def key_handler(self, key):
        """ Dispatch keyboard events to children with key handler """
        for child in (c for c in self.children if hasattr(c, 'key_handler')):
            child.key_handler(key)

class RotationControlNode(Node):
    def __init__(self, key_up, key_down, axis, angle=0, speed=0):
        super().__init__(transform=rotate(axis, angle))
        self.speed = speed
        self.angle, self.axis = angle, axis
        self.key_up, self.key_down = key_up, key_down

    def key_handler(self, key):
        self.angle += 5 * int(key == self.key_up)
        self.angle -= 5 * int(key == self.key_down)
        self.transform = rotate(self.axis, self.angle)
        super().key_handler(key)

    def draw(self, model=identity(), **uniforms):
        #""" When redraw requested, interpolate our node transform from keys """
        #self.transform = self.keyframes.value(glfw.get_time())
        super().draw(model=model, **self.uniforms)

class RotationSpeedNode(Node):
    def __init__(self, axis, speed=0):
        super().__init__(transform=identity())
        self.speed = speed
        self.axis = axis

    def draw(self, model=identity(), **uniforms):
        self.transform = rotate(self.axis, self.speed*glfw.get_time())
        super().draw(model = model, **uniforms)

class TranslateControlNode(Node):
    def __init__(self, key_up, key_down, key_left, key_right, position, uniforms = dict()):
        super().__init__(transform=translate(position), **uniforms)
        self.uniforms = uniforms
        self.position = position
        self.key_up, self.key_down, self.key_left, self.key_right = key_up, key_down, key_left, key_right

    def key_handler(self, key):
        self.position += 0.05 * vec(1,0,0) * int(key == self.key_up)
        self.position -= 0.05 * vec(1,0,0) * int(key == self.key_down)
        self.position += 0.05 * vec(0,0,1) * int(key == self.key_right)
        self.position -= 0.05 * vec(0,0,1) * int(key == self.key_left)
        self.transform = translate(self.position)
        super().key_handler(key)

    def draw(self, model=identity(), **uniforms):
        #""" When redraw requested, interpolate our node transform from keys """
        #self.transform = self.keyframes.value(glfw.get_time())
        super().draw(model=model, **self.uniforms)


# -------------- 3D resource loader -------------------------------------------
MAX_BONES = 128

# optionally load texture module
try:
    from texture import Texture, Textured
except ImportError:
    Texture, Textured = None, None

# optionally load animation module
try:
    from animation import KeyFrameControlNode, Skinned
except ImportError:
    KeyFrameControlNode, Skinned = None, None


def load(file, shader, tex_file=None, **params):
    """ load resources from file using assimp, return node hierarchy """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_JoinIdenticalVertices | pp.aiProcess_FlipUVs
        flags |= pp.aiProcess_OptimizeMeshes | pp.aiProcess_Triangulate
        flags |= pp.aiProcess_GenSmoothNormals
        flags |= pp.aiProcess_ImproveCacheLocality
        flags |= pp.aiProcess_RemoveRedundantMaterials
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # ----- Pre-load textures; embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if tex_file:
            tfile = tex_file
        elif 'TEXTURE_BASE' in mat.properties:  # texture token
            name = mat.properties['TEXTURE_BASE'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            tfile = next((os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)), None)
            assert tfile, 'Cannot find texture %s in %s subtree' % (name, path)
        else:
            tfile = None
        if Texture is not None and tfile:
            mat.properties['diffuse_map'] = Texture(tex_file=tfile)

    # ----- load animations
    def conv(assimp_keys, ticks_per_second):
        """ Conversion from assimp key struct to our dict representation """
        return {key.mTime / ticks_per_second: key.mValue for key in assimp_keys}

    # load first animation in scene file (could be a loop over all animations)
    transform_keyframes = {}
    if scene.HasAnimations:
        anim = scene.mAnimations[0]
        for channel in anim.mChannels:
            # for each animation bone, store TRS dict with {times: transforms}
            transform_keyframes[channel.mNodeName] = (
                conv(channel.mPositionKeys, anim.mTicksPerSecond),
                conv(channel.mRotationKeys, anim.mTicksPerSecond),
                conv(channel.mScalingKeys, anim.mTicksPerSecond)
            )

    # ---- prepare scene graph nodes
    nodes = {}                                       # nodes name -> node lookup
    nodes_per_mesh_id = [[] for _ in scene.mMeshes]  # nodes holding a mesh_id

    def make_nodes(assimp_node):
        """ Recursively builds nodes for our graph, matching assimp nodes """
        keyframes = transform_keyframes.get(assimp_node.mName, None)
        if keyframes and KeyFrameControlNode:
            node = KeyFrameControlNode(*keyframes, assimp_node.mTransformation)
        else:
            node = Node(transform=assimp_node.mTransformation)
        nodes[assimp_node.mName] = node
        for mesh_index in assimp_node.mMeshes:
            nodes_per_mesh_id[mesh_index] += [node]
        node.add(*(make_nodes(child) for child in assimp_node.mChildren))
        return node

    root_node = make_nodes(scene.mRootNode)

    # ---- create optionally decorated (Skinned, Textured) Mesh objects
    for mesh_id, mesh in enumerate(scene.mMeshes):
        # retrieve materials associated to this mesh
        mat = scene.mMaterials[mesh.mMaterialIndex].properties

        # initialize mesh with args from file, merge and override with params
        index = mesh.mFaces
        uniforms = dict(
            k_d=mat.get('COLOR_DIFFUSE', (1, 1, 1)),
            k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
            k_a=mat.get('COLOR_AMBIENT', (0, 0, 0)),
            s=mat.get('SHININESS', 16.),
        )
        attributes = dict(
            position=mesh.mVertices,
            normal=mesh.mNormals,
        )

        # ---- optionally add texture coordinates attribute if present
        if mesh.HasTextureCoords[0]:
            attributes.update(tex_coord=mesh.mTextureCoords[0])

        # --- optionally add vertex colors as attributes if present
        if mesh.HasVertexColors[0]:
            attributes.update(color=mesh.mColors[0])

        # ---- compute and add optional skinning vertex attributes
        if mesh.HasBones:
            # skinned mesh: weights given per bone => convert per vertex for GPU
            # first, populate an array with MAX_BONES entries per vertex
            vbone = np.array([[(0, 0)] * MAX_BONES] * mesh.mNumVertices,
                             dtype=[('weight', 'f4'), ('id', 'u4')])
            for bone_id, bone in enumerate(mesh.mBones[:MAX_BONES]):
                for entry in bone.mWeights:  # need weight,id pairs for sorting
                    vbone[entry.mVertexId][bone_id] = (entry.mWeight, bone_id)

            vbone.sort(order='weight')   # sort rows, high weights last
            vbone = vbone[:, -4:]        # limit bone size, keep highest 4

            attributes.update(bone_ids=vbone['id'],
                              bone_weights=vbone['weight'])

        new_mesh = Mesh(shader, attributes, index, **{**uniforms, **params})

        if Textured is not None and 'diffuse_map' in mat:
            new_mesh = Textured(new_mesh, diffuse_map=mat['diffuse_map'])
        if Skinned and mesh.HasBones:
            # make bone lookup array & offset matrix, indexed by bone index (id)
            bone_nodes = [nodes[bone.mName] for bone in mesh.mBones]
            bone_offsets = [bone.mOffsetMatrix for bone in mesh.mBones]
            new_mesh = Skinned(new_mesh, bone_nodes, bone_offsets)
        for node_to_populate in nodes_per_mesh_id[mesh_id]:
            node_to_populate.add(new_mesh)

    nb_triangles = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded', file, '\t(%d meshes, %d faces, %d nodes, %d animations)' %
          (scene.mNumMeshes, nb_triangles, len(nodes), scene.mNumAnimations))
    return [root_node]


# ------------  Viewer class & window management ------------------------------
class Viewer(Node):
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):
        super().__init__()

        self.uniforms = dict()

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # initialize trackball
        self.trackball = Trackball(min_height=0.5)
        self.mouse = (0, 0)

        # Time management
        glfw.set_time(0)
        self.time = glfw.get_time()
        self.pause = False

        # Fog
        self.fog = 0.1

        self.draw_cactus = 0

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)
        glfw.set_window_size_callback(self.win, self.on_size)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glEnable(GL.GL_CULL_FACE)   # backface culling enabled (TP2)
        GL.glEnable(GL.GL_DEPTH_TEST)  # depth test now enabled (TP2)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            win_size = glfw.get_window_size(self.win)

            if not self.pause:
                self.time = glfw.get_time()

            # move the light in both directions alternatively
            if (10*self.time // 140)%2 == 0: 
                q = quaternion_from_axis_angle(self.uniforms["sun_axis"], degrees = 10*self.time % 140 + 20.)
                m = quaternion_matrix(q)
                light_dir = (m @ np.array([-1,0,0,1]))[:3]
            else:
                q = quaternion_from_axis_angle(self.uniforms["sun_axis"], degrees = -(10*self.time % 140 + 20.))
                m = quaternion_matrix(q)
                light_dir = (m @ np.array([1,0,0,1]))[:3]

            # light color depends on light orientation (red sunrise/set)
            abscos = abs(np.dot(vec(-1,0,0),light_dir))
            light_color = vec(1,0.5 + 0.5*(1-abscos**3),0.5 + 0.5*(1-abscos**3))

            # draw our scene objects
            cam_pos = np.linalg.inv(self.trackball.view_matrix())[:, 3]
            self.draw(view=self.trackball.view_matrix(),
                      projection=self.trackball.projection_matrix(win_size),
                      skybox_view=self.trackball.matrix(),
                      model=identity(),
                      w_camera_position=cam_pos,
                      time=self.time,
                      light_dir=light_dir,
                      light_color=light_color,
                      fog_coef = self.fog,
                      draw_cactus = self.draw_cactus,
                      **self.uniforms)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_Z:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key == glfw.KEY_SPACE:
                self.pause = not self.pause
                if not self.pause:
                    glfw.set_time(self.time)
            if key == glfw.KEY_RIGHT:
                self.fog += 0.01
            if key == glfw.KEY_LEFT and self.fog - 0.01 >= 0:
                self.fog -= 0.01
            if key == glfw.KEY_UP:
                self.pause = True
                self.time += 0.1
            if key == glfw.KEY_DOWN:
                self.pause = True
                self.time -= 0.1
            if key == glfw.KEY_F5:
                self.draw_cactus = 1 - self.draw_cactus

            #if key == glfw.KEY_R:
            #    self.vertex_shadows = not self.vertex_shadows
            #if key == glfw.KEY_UP:
            #    self.camera.move_z(True)
            #if key == glfw.KEY_DOWN:
            #    self.camera.move_z(False)
            #if key == glfw.KEY_RIGHT:
            #    self.camera.move_x(True)
            #if key == glfw.KEY_LEFT:
            #    self.camera.move_x(False)

            # call Node.key_handler which calls key_handlers for all drawables
            self.key_handler(key)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.trackball.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.trackball.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.trackball.zoom(deltay, glfw.get_window_size(win)[1])

    def on_size(self, _win, _width, _height):
        """ window size update => update viewport to new framebuffer size """
        GL.glViewport(0, 0, *glfw.get_framebuffer_size(self.win))
