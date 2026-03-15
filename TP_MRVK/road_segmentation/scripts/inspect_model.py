import onnxruntime as ort
import numpy as np
import sys

model_path = "/home/yehor/work_disk/catkin_ws/src/TP_MRVK/road_segmentation/models/road.onnx"

try:
    session = ort.InferenceSession(model_path)
    print("Inputs:")
    for inp in session.get_inputs():
        print(f" Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
    
    print("\nOutputs:")
    for outp in session.get_outputs():
        print(f" Name: {outp.name}, Shape: {outp.shape}, Type: {outp.type}")
        
    # Run a test inference on zeros
    input_info = session.get_inputs()[0]
    shape = [s if isinstance(s, int) else 1 for s in input_info.shape]
    dummy_input = np.zeros(shape, dtype=np.float32)
    
    outputs = session.run(None, {input_info.name: dummy_input})
    print("\nTest Output Shape:", outputs[0].shape)
    print("Test Output Stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(
        np.min(outputs[0]), np.max(outputs[0]), np.mean(outputs[0])
    ))
    
    if len(outputs[0].shape) == 4:
        print("Output Values (first few channels):", outputs[0][0, :, 0, 0])

except Exception as e:
    print(f"Error: {e}")
