import pickle
import os

class OutputLoader():
    def save(obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as handle:
            #pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(obj, handle)

    def load(path):
        with open(path, 'rb') as handle:
            obj = pickle.load(handle)
        return obj
