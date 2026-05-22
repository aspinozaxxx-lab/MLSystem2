import os
import time
import pickle
import requests
import numpy as np
from typing import Optional, Any, Callable, Literal, Union
from loguru import logger
from pydantic import Field
from aeronet_raster import BandCollectionSample

from ..base.registry_object import RegistryObject
from ..base.registry import Registry, CLASS_REGISTRY

# models, registered in this key-value storage will be accessible for deserialization
MODEL_REGISTRY = Registry()

# constants from aeronet_serving
# must comply with the ones in our min-tfs-client 
# and that used while saving models
# later should think about where to store them properly

IMAGE_INPUT_KEY = 'input'
IMAGE_OUTPUT_KEY = 'output'

# input image samples format for every adapter must be (C, H, W)
# adapter __call__ invokes following methods sequentially:
#   - handle_input_type(): cast arbitrary input (usually, it is BandSample) to np.ndarray
#   - preprocess_fn(): apply model-specific transformations to the array, such as add batch dim, transpose axes, etc.
#   - predict(): feed data to the model and return result.
#   - request(): wrapper around predict() to handle server exceptions, multiple retries, etc.
#   - postprocess_fn(): apply transformations to the result, such as remove batch dim, transpose, etc.
# By design, only predict() method should be changed in subclasses, others should be inherited from superclass


class ModelAccessError(RuntimeError):
    pass


def get_preprocess_fn(transpose=None, ndim=None, input_dtype=None):
    if not transpose and not ndim and not input_dtype:
        return None
    if isinstance(transpose, str):
        transpose = tuple(int(i.strip()) for i in transpose.split(','))

    def preprocess(x):
        while ndim and x.ndim < ndim:
            x = np.expand_dims(x, 0)
        if transpose:
            x = x.transpose(*transpose)
        if input_dtype:
            x = x.astype(input_dtype)
        return x

    return preprocess


def get_postprocess_fn(transpose=None, ndim=None, output_dtype=None):
    if not transpose and not ndim and not output_dtype:
        return None
    if isinstance(transpose, str):
        transpose = tuple(int(i.strip()) for i in transpose.split(','))

    def postprocess(x):
        while ndim is not None and x.ndim > ndim:
            x = x[0]
        while ndim is not None and x.ndim < ndim:
            x = np.expand_dims(x, 0)
        if transpose:
            x = x.transpose(*transpose)
        if output_dtype:
            x = x.astype(output_dtype)
        return x
    return postprocess


class ModelAdapter(RegistryObject):
    """
    Singleton-like objects with unique names
    Args:
        name (str): unique name of the adapter, used do store model in cache
                    (cache is needed to prevent reloading and initialization each time)
        lazy: if True, init model only upon being called
        input_transpose: parameter of np.transpose before feeding to the model if needed
                         transpose comes after changing ndim,
                         so len(transpose) must be equal to input_ndim
                         Ignored in preprocess_fn is not None
        output_transpose: parameter of np.transpose of model result if needed
                          transpose comes after changing output ndim,
                          so len(transpose) must be equal to output_ndim
                          Ignored in postprocess_fn is not None
        input_ndim: required ndim of model input (e.g. adding extra batch dimension)
                    Ignored in preprocess_fn is not None
        output_ndim: required ndim of result (e.g. reducing extra batch dimension)
                     Ignored in postprocess_fn is not None
        input_dtype: dtype to cast before feeding to the model to
        output_dtype: dtype to cast output to
        """
    _model: Any
    _preprocess_fn: Callable
    _postprocess_fn: Callable
    brick_class: str
    name: str
    lazy: bool = Field(False, description='If True, the model will be initialized upon first call')
    input_transpose: Optional[tuple] = Field(None,
                                             description='parameter of np.transpose before feeding to the model if needed'
                                                         'transpose comes after changing ndim,'
                                                         'so len(transpose) must be equal to input_ndim')
    output_transpose: Optional[tuple] = Field(None,
                                              description='parameter of np.transpose of model result if needed'
                                                          'transpose comes after changing output ndim,'
                                                          'so len(transpose) must be equal to output_ndim')
    input_ndim: Optional[int] = Field(None,
                                      description='required ndim of model input (e.g. adding extra batch dimension)')
    output_ndim: Optional[int] = Field(None,
                                       description='required ndim of model result (e.g. reducing extra batch dimension)')
    input_dtype: Optional[str] = Field('uint8', description='dtype to cast before feeding to the model to')
    output_dtype: Optional[str] = Field('uint8', description='dtype to cast output to')
    verbose: bool = Field(False)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._model = None
        self.set_preprocess_fn()
        self.set_postprocess_fn()
        if not self.lazy:
            self.init_model()

    def init_model(self):
        raise NotImplementedError

    @staticmethod
    def handle_input_type(x: Any) -> np.ndarray:
        if isinstance(x, BandCollectionSample):
            return x.numpy()
        return x

    def set_preprocess_fn(self):
        self._preprocess_fn = get_preprocess_fn(self.input_transpose, self.input_ndim, self.input_dtype)

    def set_postprocess_fn(self):
        self._postprocess_fn = get_postprocess_fn(self.output_transpose, self.output_ndim, self.output_dtype)

    def __call__(self, x: Any, *args, **kwargs) -> np.ndarray:
        """ Handle_input_dtype -> preprocess -> request -> postprocess.
            This method shouldn't be rewritten in subclasses, rewrite request() or predict() instead"""
        if self._model is None:
            self.init_model()
        
        x = self.handle_input_type(x)
        if isinstance(x, np.ndarray):
            logger.trace(f"Got {x.shape}, {x.dtype} before preprocessing")

        if self._preprocess_fn is not None:
            x = self._preprocess_fn(x)
            if isinstance(x, np.ndarray):
                logger.trace(f"Got {x.shape}, {x.dtype} after preprocessing")
        try:
            x = self.request(x)
        except Exception as e:
            raise ModelAccessError(f"Error while calling the model {self.name}: {str(e)}") from e

        if isinstance(x, np.ndarray):
            logger.trace(f"Got {x.shape}, {x.dtype} as response")

        if self._postprocess_fn is not None:
            x = self._postprocess_fn(x)
            if isinstance(x, np.ndarray):
                logger.trace(f"Got {x.shape}, {x.dtype} after postprocessing")

        return x

    def request(self, x):
        return self.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def from_config(config: dict):
        cls_name = config.pop('_class')
        force_restart = config.pop('_force_restart', True)
        name = config['name']

        if name in MODEL_REGISTRY and not force_restart:
            return MODEL_REGISTRY[name]
        else:
            cls = CLASS_REGISTRY[cls_name]
            obj = cls(**config)
            MODEL_REGISTRY[name] = obj
            return obj

    @staticmethod
    def from_name(name):
        return MODEL_REGISTRY[name]

    @property
    def exceptions(self):
        """Should return list of exceptions which are raised when model fail"""
        return []


class MockAdapter(ModelAdapter):
    """For auto tests, allows us to use mock-model (some simple python function) in test pipelines
    Args:
        path (str): path to model *.py file
    """

    path: str
    brick_class: Literal['MockAdapter'] = Field('MockAdapter')

    def init_model(self):
        import importlib.util
        import sys
        import os
        module_name = os.path.splitext(os.path.basename(self.path))[0]
        spec = importlib.util.spec_from_file_location(module_name, self.path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[self.name] = model_module
        spec.loader.exec_module(model_module)
        self._model = model_module.Model()

    def predict(self, x: np.ndarray):
        return self._model(x)


class TorchJitModelAdapter(ModelAdapter):
    """Interface for local PyTorch JIT (serialized) model
    
    Args:
        path (str, optional): path to model *.pt file
        device (str, optional): location of the model, e.g. "cpu" or "cuda:0"
    """
    brick_class: Literal['TorchJitModelAdapter']
    path: str
    device: Literal['cpu', 'gpu'] = Field('cpu')

    def init_model(self):
        import torch
        self._model = torch.jit.load(self.path, map_location=self.device)
        self._model.eval()

    def predict(self, x):
        import torch
        # with torch.no_grad(): !! This leads to a bug with some models
        x = torch.from_numpy(x).to(self.device)
        x = self._model(x)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, (tuple, list)):
            return [i.detach().cpu().numpy() for i in x]
        else:
            return x


class RemoteServerModelAdapter(ModelAdapter):
    """Abstract class implements request() to a remote host with n_retries"""
    brick_class: Literal['RemoteServerModelAdapter']
    host: str = Field(description='Model serving host address')  #Todo: validation?
    port: Union[str, int] = Field(description='Model serving port')  #Todo: validation?
    timeout: int = Field(20, gt=0, description='Timeout for a single request in seconds')
    n_retries: int = Field(1, ge=0, description='Number of retries. 0 means one try')
    retry_sleep: int = Field(10, gt=0, description='Wait between retries in seconds')

    def request(self, x: np.ndarray, *args, **kwargs):
        for i in range(self.n_retries+1):
            try:
                result = self.predict(x)
                break
            except Exception as e:
                if i == self.n_retries:
                    logger.exception(e)
                    raise e
                else:
                    logger.warning(str(e))
                    logger.warning(f'Sleeping {self.retry_sleep} seconds before {i+1}-th retry')
                    time.sleep(self.retry_sleep)
        return result


class TFServingModelAdapter(RemoteServerModelAdapter):
    """Interface for remote Tensorflow Serving models 
    (use GRPC to send and receive data)
    
    Args:
        name (str): unique name of the adapter, used do store model in cache 
            (cache is needed to prevent reloading and initialization each time)
        host (str): host address
        port (str): port number of TF Serving
        timeout (int, optional): time to wait response from serving. Defaults to 60.
        n_retries (int, optional): number of retries in case serving is not respond. Defaults to 10.
        retry_sleep (int, optional): sleep time between retries. Defaults to 20.
    """
    brick_class: Literal['TFServingModelAdapter']
    input_transpose: Optional[tuple] = Field((0, 2, 3, 1))
    output_transpose: Optional[tuple] = Field((2, 0, 1))
    input_ndim: int = Field(4)
    output_ndim: int = Field(3)
    input_dtype: str = Field('uint8')
    output_dtype: str = Field('uint8')

    def init_model(self):
        from min_tfs_client.requests import TensorServingClient
        self._model = TensorServingClient(self.host, self.port)

    def predict(self, x: np.ndarray):
        from min_tfs_client.tensors import tensor_proto_to_ndarray
        logger.trace(f'Sending request to {self._model}, name={self.name}')
        input_dict = {IMAGE_INPUT_KEY: x}
        raw_result = self._model.predict_request(model_name=self.name, input_dict=input_dict, timeout=self.timeout)
        result = tensor_proto_to_ndarray(raw_result.outputs[IMAGE_OUTPUT_KEY])
        return result


class TorchServingModelAdapter(RemoteServerModelAdapter):
    """Interface for remote Torch Serve models
    (pickle input data and send to server, then unpickle the result)
    
    Args:
        version (str): model version. Defaults to None.
    """
    brick_class: Literal['TorchServingModelAdapter']
    input_transpose: Optional[tuple] = Field((1, 2, 0))
    output_transpose: Optional[tuple] = Field((2, 0, 1))
    input_ndim: int = Field(3)
    output_ndim: int = Field(3)
    input_dtype: str = Field('uint8')
    output_dtype: str = Field('uint8')
    version: Optional[float] = Field(None)

    def init_model(self):
        pass
    
    @property
    def url(self):
        host_port = "{}:{}".format(self.host, self.port) if self.port is not None else self.host
        url = "http://{host_port}/predictions/{model_name}/".format(host_port=host_port, model_name=self.name)
        if self.version is not None:
            url += (str(self.version) + "/")
        return url
    
    @staticmethod
    def encode(data: np.array) -> bytes:
        return pickle.dumps(data)
        
    @staticmethod
    def decode(data: bytes) -> np.array:
        return pickle.loads(data)

    def predict(self, x: np.ndarray):
        logger.trace(f'Sending request to {self.url}, name={self.name}')
        response = requests.post(self.url, files={"data": self.encode(x)}, timeout=self.timeout)
        response.raise_for_status()
        result = self.decode(response.content)
        return result


class TritonAdapter(RemoteServerModelAdapter):
    """Interface for remote Nvidia Triton service models
    Args:
        version (str): model version. Defaults to None.
        protocol(str, optional): HTTP or GRPC protocol type. Defaults HTTP.
    """
    brick_class: Literal['TritonAdapter']
    version: str = Field('')
    protocol: Literal['http', 'grpc'] = Field('http')
    classes: int = Field(0)
    _client: Any
    _input_metadata: Any
    _output_metadata: Any

    def model_post_init(self, __context):
        super().model_post_init(__context)
        # A way to substitute host for inference container to use it over pipeline defined
        predef_host = os.getenv("TRITON_MODELS_HOST")
        predef_port = os.getenv("TRITON_MODELS_PORT")
        if predef_host and predef_port:
            self.host = predef_host
            self.port = predef_port
    
    def init_model(self):
        if self.protocol == "http":
            import tritonclient.http as httpclient
            # Specify large enough concurrency to handle the number of requests.
            self._client = httpclient
            self._model = httpclient.InferenceServerClient(url=self.url,
                                                           verbose=self.verbose,
                                                           concurrency=1,
                                                           connection_timeout=self.timeout,
                                                           network_timeout=self.timeout)
        else:
            raise ValueError('Unsupported protocol, supported type is http')
        try:
            model_metadata = self._model.get_model_metadata(model_name=self.name, model_version=self.version)
        except Exception as e:
            raise ModelAccessError(f"Error while initializing the model {self.name}: {str(e)}") from e

        self._input_metadata = model_metadata['inputs']
        self._output_metadata = model_metadata['outputs']
        logger.trace(f'Model {self.name} initialized,\n'
                     f'inputs={self._input_metadata}\n'
                     f'output={self._output_metadata}\n')

    @property
    def url(self):
        return "{}:{}".format(str(self.host), str(self.port))
        
    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = [x]

        inputs = []
        for i, inp in enumerate(self._input_metadata):
            infer_input = self._client.InferInput(inp['name'], x[i].shape, inp['datatype'])
            infer_input.set_data_from_numpy(x[i])
            inputs.append(infer_input)

        outputs = [self._client.InferRequestedOutput(outp['name'],
                                                     class_count=self.classes) for outp in self._output_metadata]
        response = self._model.infer(self.name,
                                     inputs,
                                     model_version=self.version,
                                     outputs=outputs)
        results = [response.as_numpy(outp['name']) for outp in self._output_metadata]
        if len(results) == 1:
            results = results[0]
            
        return results

    @property
    def exceptions(self):
        """Should return list of exceptions which are raised when model fail"""
        from tritonclient.utils import InferenceServerException
        return [InferenceServerException]

ALL_ADAPTER_TYPES = MockAdapter | TorchServingModelAdapter | TritonAdapter