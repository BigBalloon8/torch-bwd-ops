import torch
import inspect
import ctypes

def func(x,y):
    return x + y

capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
capsule_pointer.restype = ctypes.c_void_p  # Generic void pointer
capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

if __name__ == '__main__':
    pointer = capsule_pointer(torch.ops.aten.convolution_backward._op, b"expected_name")
    print(hex(pointer))