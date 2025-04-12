import torch
import numpy as np


tensor_0d = torch.tensor(1)
tensor_1d = torch.tensor([1, 2, 3])
tensor_2d = torch.tensor(
    [[1, 2],
     [3, 4]]
)
tensor_3d = torch.tensor(
    [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ]
)

np_3d = np.array(
    [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ]
)

print(f"tensor_0d: {tensor_0d}, {tensor_0d.shape}")
print(f"tensor_1d: {tensor_1d}, {tensor_1d.shape}")
print(f"tensor_2d: {tensor_2d}, {tensor_2d.shape}")
print(f"tensor_3d: {tensor_3d}, {tensor_3d.shape}")
print(f"np_3d: {np_3d}, {np_3d.shape}")

tensor3d_2 = torch.tensor(np_3d)  # ディープコピー
tensor3d_3 = torch.from_numpy(np_3d)  # シャロウコピー
np_3d[0, 0, 0] = 999

print(f"tensor3d_2: {tensor3d_2}")
print(f"tensor3d_3: {tensor3d_3}")

int_tensor = torch.tensor([1, 2, 3])
float_tensor = torch.tensor([1.0, 2.0, 3.0])
float_from_int_tensor = int_tensor.to(torch.float32)
print(f"int_tensor.dtype: {int_tensor.dtype}")
print(f"float_tensor.dtype: {float_tensor.dtype}")
print(f"float_from_int_tensor.dtype: {float_from_int_tensor.dtype}")

tensor_cal = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor_cal.view(3, 2))
print(tensor_cal.T)
print(tensor_cal.matmul(tensor_cal.T))
print(tensor_cal @ tensor_cal.T)
