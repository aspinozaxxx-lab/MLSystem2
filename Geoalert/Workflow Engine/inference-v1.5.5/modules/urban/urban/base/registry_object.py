# all classes for serialisation/deserialization should be inherited from RegistryObject
# RegistryObject inherited classes automatically register in Registry
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from .registry import CLASS_REGISTRY


class _ClassRegistryMeta(ModelMetaclass):

    def __new__(mcs, name, bases, attrs):
        new_cls = super().__new__(mcs, name, bases, attrs)
        CLASS_REGISTRY[name] = new_cls
        return new_cls


class RegistryObject(BaseModel, metaclass=_ClassRegistryMeta):
    def get_config(self) -> dict:
        config = self.model_dump()
        config['brick_class'] = self.__class__.__name__
        return config
