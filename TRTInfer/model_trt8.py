import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from collections import OrderedDict
import time
class MyOrderedDict8(OrderedDict):
    def add(self, key, value):
        self[key] = value

class TimeProfiler8:
    def __init__(self):
        self.total = 0
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.total += time.time() - self._start
        self._start = None

    def reset(self):
        self.total = 0

class TRTInference8(object):
    def __init__(self, engine_path, verbose=False):
        # 设置类型的各种参数
        self.engine_path = engine_path  # engine 的路径
        
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
        self.time_profile = TimeProfiler8()
    
    def load_engine(self, path):
        '''load engine'''
        # 初始化部件
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            engine_data = f.read()
            return runtime.deserialize_cuda_engine(engine_data)
    
    def get_input_names(self):
        # 获取输入名称
        names = []
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                name = self.engine.get_binding_name(idx)
                print(f" name: {name}, shape: {self.engine.get_binding_shape(idx)}")
                names.append(name)
        return names
    
    def get_output_names(self):
        names = []
        for idx in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(idx):
                name = self.engine.get_binding_name(idx)
                print(f" name: {name}, shape: {self.engine.get_binding_shape(idx)}")
                names.append(name)
        return names
    
    def get_bindings(self, engine):
        bindings = MyOrderedDict8()
        for idx in range(engine.num_bindings):
            name = engine.get_binding_name(idx)
            shape = engine.get_binding_shape(idx)
            dtype = trt.nptype(engine.get_binding_dtype(idx))
            print(f" tensor name: {name}, expect dtype: {engine.get_binding_dtype(idx)}")
            
            # set the data
            data = np.random.randn(*shape).astype(dtype)
            ptr = cuda.mem_alloc(data.nbytes)
            
            # 添加约束
            bindings.add(name, ptr)
        return bindings

    def async_run_cuda(self, blob):
        output_blob = {}
        
        # 设置输入
        for name in self.input_names:
            cuda.memcpy_htod_async(self.bindings[name], blob[name], self.stream)
        
        # 准备输出
        for name in self.output_names:
            output_blob[name] = np.zeros(self.engine.get_binding_shape(self.engine.get_binding_index(name)), 
                                          dtype=trt.nptype(self.engine.get_binding_dtype(self.engine.get_binding_index(name))))
        
        # 异步执行
        self.context.execute_async_v2(
            bindings=[int(self.bindings[name]) for name in (self.input_names + self.output_names)],
            stream_handle=self.stream.handle
        )
        
        # 得到输出
        for name in self.output_names:
            cuda.memcpy_dtoh_async(output_blob[name], self.bindings[name], self.stream)
        
        # 关闭流
        self.stream.synchronize()
        return output_blob
    
    def __call__(self, blob):
        return self.async_run_cuda(blob)
    
    def speed(self, blob, n):
        self.time_profile.reset()
        for _ in range(n):
            with self.time_profile:
                _ = self(blob)
        return self.time_profile.total / n