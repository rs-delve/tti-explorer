
class Registry:
    "Case insensitive registry"
    def __init__(self):
        self._register = dict()
    
    def __getitem__(self, key):
        return self._register[key.lower()]
    
    def __call__(self, name):
        def add(thing):
            self._register[name.lower()] = thing
            return thing
        return add
