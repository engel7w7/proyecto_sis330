import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb
import flatbuffers

def export_dual_output_and_weights(
    model_path="android_app/app/src/main/assets/experto_vision_int8.tflite",
    weights_path="android_app/app/src/main/assets/gradcam_weights.bin"
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    with open(model_path, "rb") as f:
        buf = bytearray(f.read())

    model = schema_fb.Model.GetRootAsModel(buf, 0)
    model_obj = schema_fb.ModelT.InitFromObj(model)

    model_obj.subgraphs[0].outputs = [486, 483]

    b = flatbuffers.Builder(1024 * 1024 * 10)
    b.Finish(model_obj.Pack(b), b"TFL3")
    new_buf = b.Output()

    with open(model_path, "wb") as f:
        f.write(new_buf)
    print(f"Modelo TFLite de vision actualizado con doble salida: {model_path} ({len(new_buf)} bytes)")

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()

    weights_details = interp.get_tensor_details()[12]
    w_raw = interp.get_tensor(12)
    scales = weights_details["quantization_parameters"]["scales"]
    zero_points = weights_details["quantization_parameters"]["zero_points"]
    w_float = (w_raw.astype(np.float32) - zero_points[:, None]) * scales[:, None]
    w_fake = w_float[0].astype(np.float32)

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    with open(weights_path, "wb") as f:
        f.write(w_fake.tobytes())
    print(f"Pesos Grad-CAM exportados: {weights_path} ({len(w_fake.tobytes())} bytes)")

if __name__ == "__main__":
    export_dual_output_and_weights()
