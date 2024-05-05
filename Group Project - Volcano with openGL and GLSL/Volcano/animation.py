# Python built-in modules
from bisect import bisect_left      # search sorted keyframe lists

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args

from core import Node
from transform import (lerp, quaternion_slerp, quaternion_matrix, translate,
                       scale, identity)


# -------------- Keyframing Utilities TP6 ------------------------------------
class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        if time < self.times[0]:
            return self.values[0]
        if time > self.times[-1]: # looping animation
            time = ((time - self.times[0]) % (self.times[-1] - self.times[0])) + self.times[0]
        
        index_plus = bisect_left(self.times, time)
        index = index_plus - 1

        f = (time - self.times[index])/(self.times[index_plus] - self.times[index])

        #print(self.values)

        #value = self.interpolate(self.values[index], self.values[index_plus], f)

        # 1. ensure time is within bounds else return boundary keyframe

        # 2. search for closest index entry in self.times, using bisect_left

        # 3. using the retrieved index, interpolate between the two neighboring
        # values in self.values, using the stored self.interpolate function
        return self.interpolate(self.values[index], self.values[index_plus], f)


class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translate_keys = KeyFrames(translate_keys)
        self.rotate_keys = KeyFrames(rotate_keys, quaternion_slerp)
        self.scale_keys = KeyFrames(scale_keys)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        translation = translate(self.translate_keys.value(time))
        scaling = scale(self.scale_keys.value(time))
        rotation = quaternion_matrix(self.rotate_keys.value(time))
        #print(translation)
        #print(rotation)
        #print(scaling)
        return translation @ rotation @ scaling
        #return translation @ scaling


class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, trans_keys, rotate_keys, scale_keys, uniforms = dict(), transform=identity()):
        self.uniforms = uniforms
        super().__init__(transform=transform)
        self.keyframes = TransformKeyFrames(trans_keys, rotate_keys, scale_keys)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(primitives=primitives, **self.uniforms)


# -------------- Linear Blend Skinning : TP7 ---------------------------------
class Skinned:
    """ Skinned mesh decorator, passes bone world transforms to shader """
    def __init__(self, mesh, bone_nodes, bone_offsets):
        self.mesh = mesh

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = np.array(bone_offsets, np.float32)

    def draw(self, **uniforms):
        world_transforms = [node.world_transform for node in self.bone_nodes]
        uniforms['bone_matrix'] = world_transforms @ self.bone_offsets
        self.mesh.draw(**uniforms)
