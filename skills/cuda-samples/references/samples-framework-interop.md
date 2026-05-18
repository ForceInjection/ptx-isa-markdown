# Framework Interop Samples

## 8.1 customPyTorchKernel

- **Path**: `python/3_FrameworkInterop/customPyTorchKernel/customPyTorchKernel.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/3_FrameworkInterop/customPyTorchKernel/customPyTorchKernel.py>
- **Pattern**: Full PyTorch custom CUDA op via `torch.autograd.Function` with forward and backward kernels compiled through cuda.core. Kernel caching avoids recompilation.
- **Arch**: All
- **Lines**: ~390

```python
class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.empty_like(input)
        launch(torch_stream, config, square_kernel, input_buf, output_buf, n)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.empty_like(input)
        launch(torch_stream, config, square_backward_kernel, input_buf, grad_output_buf, grad_input_buf, n)
        return grad_input
```

## 8.2 customTensorFlowKernel

- **Path**: `python/3_FrameworkInterop/customTensorFlowKernel/customTensorFlowKernel.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/3_FrameworkInterop/customTensorFlowKernel/customTensorFlowKernel.py>
- **Pattern**: TensorFlow custom CUDA ReLU op via `tf.py_function` + `@tf.custom_gradient` wrapping cuda.core kernels
- **Arch**: All
- **Lines**: ~430

```python
@tf.custom_gradient
def custom_relu(x):
    y = tf.py_function(func=lambda x_np: launch_relu_kernel(x_np), inp=[x], Tout=x.dtype)
    def grad(dy):
        return tf.py_function(
            func=lambda dy_np, x_np: launch_relu_backward_kernel(dy_np, x_np),
            inp=[dy, x], Tout=x.dtype)
    return y, grad
```
