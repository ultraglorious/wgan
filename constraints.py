import tensorflow as tf


class ClipConstraint(tf.keras.constraints.Constraint):
    """Clip weight values within a range"""
    def __init__(self, clip_value: float):
        self.clip_value = clip_value

    def __call__(self, weights: tf.Tensor, *args, **kwargs):
        return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {"clip value": self.clip_value}
