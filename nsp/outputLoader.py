import pickle
import os

class OutputLoader():
    """
    Store and load intermediate results.
    Especially useful for storing `NNGraph` and their transformers
        `inv_F` or `lu_piv` since their computations take long and
        they can be used to transform multiple `Activations`.
    """
    def save(obj, path):
        """Store python object like `NNGrpah` or `Activations`.

        Parameters
        ----------
        obj : NNGraph, Activation
            Object to save.
        path : str
            Path to desired location of `obj`.

        Returns
        -------
        None
        """
        os.makedirs(os.path.dirname('./'+path), exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(path):
        """Load python object like `NNGrpah` or `Activation`.

        Parameters
        ----------
        path: str
            Path to location of the object.

        Returns
        -------
            The object stored at `path`.

        """
        with open(path, 'rb') as handle:
            obj = pickle.load(handle)
        return obj
