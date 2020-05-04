

class Registry:
    def __init__(self):
        self._register = dict()
    
    def __getitem__(self, key):
        return self._register[key]
    
    def __call__(self, name):
        def add(thing):
            self._register[name] = thing
            return thing
        return add
