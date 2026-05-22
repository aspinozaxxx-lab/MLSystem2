class Registry(dict):

    def __getitem__(self, key):
        if key not in self:
            raise KeyError('`{}` not in Registry'.format(key))

        return super().__getitem__(key)

    def add(self, name, value):
        self[name] = value


# all classes registered in this key-value storage to be deserializable
CLASS_REGISTRY = Registry()
