"""Export joint models to SavedModel, then TF.js graph format is done via CLI."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

WORK = r"C:\Users\BRAVO\AppData\Local\Temp\opencode\sibi-train"

ours = tf.keras.models.load_model(os.path.join(WORK, "joint_model", "sibi_joint_best.h5"))
ours.export(os.path.join(WORK, "joint_sm"))
print("ours exported:", ours.input_shape, "->", ours.output_shape)

base = tf.keras.models.load_model(os.path.join(WORK, "upstream_model.h5"), compile=False)
base.export(os.path.join(WORK, "baseline_sm"))
print("baseline exported:", base.input_shape, "->", base.output_shape)
