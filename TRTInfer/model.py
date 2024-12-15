import numpy as np
import time
import tensorrt as trt
from typing import OrderedDict
import torch
import pycuda.autoinit
import pycuda.driver as cuda
import contextlib


# 装载数据的字典，通过张量名称装载张量的地址头
class MyOrderedDict:
    def __init__(self):
        self.data = OrderedDict()
    def add(self, key:str, value):
        self.data[key] = value
    def get_value(self, key):
        return self.data.get(key, None)
    def values_as_list(self):
        return list(self.data.values())
    def __setitem__(self, key, value):
        self.add(key,value)
    def __getitem__(self, key):
        return self.get_value(key)
# 时间记录器
class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
    def __enter__(self, ):
        self.start = self.time()
        return self 
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    def reset(self, ):
        self.total = 0
    def time(self, ):
        # 记录时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

# ***** tensorrt + cuda 的推理类 *****
class TRTInference(object):
    def __init__(self, engine_path,  verbose=False):
        # 设置类型的各种参数
        self.engine_path = engine_path # engine 的路径
        # 日志
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)  
        # 权重引擎
        self.engine = self.load_engine(engine_path)
        # 上下文
        self.context = self.engine.create_execution_context()
        # 约束指定
        self.bindings = self.get_bindings(self.engine)
        # 输入和输出的名称
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        # cuda 后端的流数据
        self.stream = cuda.Stream()
        # 计时器
        self.time_profile = TimeProfiler()
    
    # 下载权重
    def load_engine(self, path):
        '''load engine
        '''
        # 初始化部件
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    # 获取输入名称
    def get_input_names(self, ):
        # 获取输入名称
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                print(" name : " ,name , ", tensor shape : ",self.engine.get_tensor_shape(name) , ", tensor type : ",self.engine.get_tensor_dtype(name),", tensor format : ",self.engine.get_tensor_format_desc(name))
                names.append(name)
        return names
    
    # 获取输出名称
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                print(" name : " ,name , ", tensor shape : ",self.engine.get_tensor_shape(name) , ", tensor type : ",self.engine.get_tensor_dtype(name),", tensor format : ",self.engine.get_tensor_format_desc(name))
                names.append(name)
        return names
    
    # 获取约束张量的列表
    def get_bindings(self, engine) -> list:
        bindings = MyOrderedDict()
        for _ , name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            print(" tensor name : ",name,",expect dtype : ",engine.get_tensor_dtype(name))
            # set the data
            data = np.random.randn(*shape).astype(dtype)
            ptr = cuda.mem_alloc(data.nbytes)
            # 添加约束
            bindings.add(name,ptr)
        return bindings

	# 异步执行cuda加速
    def async_run_cuda(self, blob):
        output_blob = {}
        # 设置输入
        for name in self.input_names:
            cuda.memcpy_htod_async(self.bindings[name], blob[name], self.stream)
            # 设置张量地址
            self.context.set_tensor_address(name,self.bindings[name])
        # 设置输出
        for name in self.output_names:
            # 设置张量地址
            self.context.set_tensor_address(name,self.bindings[name])
            output_blob[name] = np.zeros(self.engine.get_tensor_shape(name),dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
        # 异步执行
        self.context.execute_async_v3(self.stream.handle)
        # 得到输出
        for name in self.output_names:
            cuda.memcpy_dtoh_async(output_blob[name], self.bindings[name], self.stream)
        # 关闭流
        self.stream.synchronize()
        return output_blob
    # ( ) 重载
    def __call__(self, blob):
        return self.async_run_cuda(blob)   
    # 测量n次，计算平均值作为运行时间
    def speed(self, blob, n):
        self.time_profile.reset()
        for _ in range(n):
            with self.time_profile:
                _ = self(blob)

        return self.time_profile.total / n 