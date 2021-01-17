import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec

class ReflectionPadding2D(Layer):
    def __init__(self):
        self.padding = tuple((1,1))
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__()

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
