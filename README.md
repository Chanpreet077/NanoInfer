# NanoInfer

NanoInfer is a lightweight neural-network inference runtime built from scratch in C++ with CUDA acceleration and INT8 weight quantization.

It implements core inference components manually, including tensors, linear layers, ReLU, softmax, model loading, CPU matrix multiplication, custom CUDA kernels, benchmarking, and quantized weight inference.

## Features

- Custom C++ Tensor implementation
- CPU FP32 inference
- CUDA FP32 inference
- INT8 weight quantization
- Linear, ReLU, and Softmax layers
- MNIST model loading and inference
- Naive CUDA matrix multiplication kernel
- Shared-memory tiled CUDA kernel
- Runtime CUDA kernel selection
- CPU vs CUDA benchmarking
- FP32 vs INT8 weight-memory comparison

## Model Architecture

```text
784 input pixels
      |
      v
Linear 784 -> 128
      |
      v
     ReLU
      |
      v
Linear 128 -> 10
      |
      v
   Softmax
      |
      v
10 digit probabilities
```

Each linear layer performs:

```text
output = input x weights + bias
```

## CUDA Acceleration

NanoInfer includes custom CUDA matrix multiplication instead of using cuBLAS.

Two kernels are implemented:

- Naive kernel: each GPU thread computes one output element
- Tiled kernel: uses shared memory to reuse matrix data

For the current MNIST batch-size-1 workload, the naive kernel performs better. The tiled kernel is kept for larger matrix shapes where shared-memory reuse can be more useful.

## INT8 Quantization

NanoInfer supports symmetric weight-only INT8 quantization.

FP32 weights use 4 bytes per weight, while INT8 weights use 1 byte per weight.

The quantization scale is:

```text
scale = max(abs(weight)) / 127
```

Weights are quantized approximately using:

```text
quantized_weight = round(weight / scale)
```

During inference, the INT8 value is reconstructed using the stored scale.

This implementation focuses on reducing weight storage rather than performing full integer arithmetic.

## Results

CPU FP32, CUDA FP32, and INT8 inference all preserved the MNIST prediction:

```text
CPU prediction: 7
CUDA prediction: 7
INT8 prediction: 7
```

INT8 probabilities changed slightly because of quantization, but the predicted class stayed the same.

## Performance

Benchmark over 1000 inference runs:

```text
CPU average inference time:  1.84478 ms
CUDA average inference time: 0.48488 ms
CPU/CUDA speedup:            3.8046x
```

Results may vary depending on hardware and runtime environment.

## Weight Memory

The network contains:

```text
784 x 128 = 100,352 weights
128 x 10  =   1,280 weights

Total = 101,632 weights
```

FP32 storage:

```text
101,632 x 4 bytes = 406,528 bytes
```

INT8 storage:

```text
101,632 x 1 byte = 101,632 bytes
```

Result:

```text
75% reduction in weight storage
~4x smaller weight representation
```

This refers to weight storage only. Biases remain FP32, and the current runtime still keeps the original FP32 weights alongside the quantized weights.

## Project Structure

```text
NanoInfer/
|
|-- README.md
|-- CMakeLists.txt
|
|-- include/
|   |-- tensor.h
|   |-- linear.h
|   |-- relu.h
|   |-- softmax.h
|   |-- model.h
|   |-- model_loader.h
|   |-- cuda_matmul.h
|   `-- quantization.h
|
|-- src/
|   |-- main.cpp
|   |-- tensor.cpp
|   |-- linear.cpp
|   |-- relu.cpp
|   |-- softmax.cpp
|   |-- model.cpp
|   |-- model_loader.cpp
|   |-- cuda_matmul.cu
|   `-- quantization.cpp
|
|-- models/
|   |-- mnist_weights.txt
|   `-- mnist_test_image.txt
|
`-- python/
    `-- train_model.py
```

## Running NanoInfer

CUDA execution requires an NVIDIA GPU and CUDA toolkit.

If your local computer does not support CUDA, the easiest option is Google Colab.

### Running on Google Colab

1. Zip the entire NanoInfer project folder.

You should have:

```text
NanoInfer.zip
```

2. Open a new Google Colab notebook.

3. Enable a GPU runtime:

```text
Runtime
-> Change runtime type
-> Hardware accelerator
-> GPU
```

4. Upload `NanoInfer.zip` using the Files panel on the left side of Colab.

5. Unzip the project:

```python
!unzip NanoInfer.zip
```

6. Enter the project folder:

```python
%cd NanoInfer
```

7. Check that a GPU is available:

```python
!nvidia-smi
```

8. Check that the CUDA compiler is available:

```python
!nvcc --version
```

9. Compile NanoInfer:

```python
!nvcc -std=c++17 -Iinclude \
src/main.cpp \
src/tensor.cpp \
src/linear.cpp \
src/relu.cpp \
src/softmax.cpp \
src/model.cpp \
src/model_loader.cpp \
src/quantization.cpp \
src/cuda_matmul.cu \
-o nanoinfer_cuda
```

10. Run NanoInfer:

```python
!./nanoinfer_cuda
```

Example output:

```text
CPU NanoInfer prediction: 7
GPU NanoInfer prediction: 7
INT8 NanoInfer prediction: 7
```

The program also reports:

- FP32 vs INT8 weight memory
- CPU inference time
- CUDA inference time
- CPU/CUDA speedup

## Training

The MNIST model is trained separately using PyTorch.

The training script is located at:

```text
python/train_model.py
```

PyTorch is used for training and exporting the trained weights.

NanoInfer then loads those weights and performs inference using its own C++ and CUDA implementations.

```text
PyTorch training
      |
      v
Export weights
      |
      v
NanoInfer
      |
      v
C++ / CUDA inference
```

## What This Project Demonstrates

NanoInfer explores the systems side of machine-learning inference.

It demonstrates:

- Neural-network inference fundamentals
- Tensor representation
- Matrix multiplication
- Model weight loading
- CPU inference
- CUDA programming
- GPU memory transfers
- Shared-memory optimization
- Kernel-selection tradeoffs
- CPU vs GPU benchmarking
- INT8 quantization
- Memory and accuracy tradeoffs

## Current Limitations

NanoInfer is intentionally small and experimental.

Current limitations include:

- Supports only a small set of neural-network operations
- Primarily supports fully connected models
- CUDA memory is allocated and copied during individual matrix multiplication calls
- INT8 uses weight-only quantization
- INT8 weights are reconstructed to floating point during multiplication
- FP32 weights are still retained alongside INT8 weights
- No persistent GPU model storage
- No automatic computation graph
- No standard model format such as ONNX

## Future Improvements

Possible future extensions include:

- Persistent GPU buffers
- Batched inference
- More tensor operations
- Convolution layers
- Additional activation functions
- Full INT8 activation quantization
- INT32 accumulation
- CUDA INT8 kernels
- Binary model serialization
- Automated kernel benchmarking
- More advanced kernel dispatch
- Support for larger neural-network architectures