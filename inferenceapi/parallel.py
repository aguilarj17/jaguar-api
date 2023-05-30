import multiprocessing
import onnxruntime
# n_threads = min(4, multiprocessing.cpu_count() - 1)
# print(f"n_threads={n_threads}")


def visualize_model():
    sess1 = onnxruntime.InferenceSession('models/model_bbox_regression_and_classification_m1_vf.onnx', providers=["CPUExecutionProvider"])
    for i in sess1.get_inputs():
        print(f"input {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
        input_name = i.name
        input_shape = list(i.shape)
        if input_shape[0] in [None, "batch_size", "N"]:
            input_shape[0] = 1

    output_name = None
    for i in sess1.get_outputs():
        print(f"output {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
        if output_name is None:
            output_name = i.name

    print(f"input_name={input_name!r}, output_name={output_name!r}")


def multiple_threads():
    n_threads = min(4, multiprocessing.cpu_count() - 1)
    
    sess1 = [onnxruntime.InferenceSession('models/model_bbox_regression_and_classification_m1_vf.onnx', providers=["CPUExecutionProvider"])for i in range(n_threads)]



if __name__=='__main__':
    #try_model()
    multiple_threads()