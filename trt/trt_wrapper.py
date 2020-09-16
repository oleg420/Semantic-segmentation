import tensorrt as trt
import pycuda.driver as cuda

import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class Deeplabv3TRT:
    def __init__(self, engine_path, num_classes, size, ctx, stream):
        self.ctx = ctx
        self.stream = stream

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.h_input_tmp = np.zeros((1, 3, size, size), dtype=np.float32)
        self.h_output = np.zeros((1, num_classes, size, size), dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.h_input_tmp.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

    def __call__(self, h_input):
        assert self.h_input_tmp.shape != h_input, 'Invalid input shape'

        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod_async(self.d_input, h_input, self.stream)

            context.execute_async(bindings=[int(self.d_input),
                                            int(self.d_output)],
                                  stream_handle=self.stream.handle)

            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)

            self.stream.synchronize()

        return self.h_output
