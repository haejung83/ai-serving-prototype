from tensorflow.python.keras.engine import saving
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import model_from_json
import h5py
import os


_POSTFIX_NAME_ARCH_FILE = '_arch.json'
_POSTFIX_NAME_FREEZED_FILE = '_freezed.h5'
_POSTFIX_NAME_TUNED_FILE = '_tuned.h5'


def thin_model_from_json(root_path=None, filename=None, custom_objects=None):
    if not filename:
        raise ValueError('[%s] Invalid filename for constructing ThingModel fro file' %(__name__))

    filename = _extract_file_name(filename)

    if root_path:
        filename = os.path.join(root_path, filename)

    json_file = open(filename + _POSTFIX_NAME_ARCH_FILE, 'r')
    loaded_json = json_file.read()
    json_file.close()

    _custom_objects = {'ThinModel': ThinModel}

    if custom_objects:
        _custom_objects.update(custom_objects)

    # Build model from an architecture file(json)
    loaded_model = model_from_json(loaded_json, custom_objects=_custom_objects)

    # Load freeze weights into the model
    loaded_model.load_weights(
        filename + _POSTFIX_NAME_FREEZED_FILE, by_name=True)
    # Load tuned weights into the model
    loaded_model.load_weights(
        filename + _POSTFIX_NAME_TUNED_FILE, by_name=True)

    return loaded_model


def _extract_file_name(filepath=None):
    if not filepath:
        raise ValueError('[%s] Invalid filepath for extracting filename' %(__name__))

    if '.h5' in filepath:
        filename = filepath[:filepath.find('.h5')]
    else:
        filename = filepath

    return filename


class ThinModel(Model):

    def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
        # Signature detection
        if (len(args) == 2 or
            len(args) == 1 and 'outputs' in kwargs or
                'inputs' in kwargs and 'outputs' in kwargs):
            # Graph network
            self._init_graph_network(*args, **kwargs)
        else:
            # Subclassed network
            self._init_subclassed_network(**kwargs)

        self._internal_init(args, kwargs)

    def _internal_init(self, *args, **kwargs):
        self._separation_marked_layer = None

    def save_weights_separately(self, root_path, filename, overwrite=True):
        if not self.built:
            raise ValueError('[%s] This model has not built layers' %(__name__))

        if not self._separation_marked_layer:
            raise ValueError('[%s] There is no seperation mark layer. '
                             'set_separation_mark should be called '
                             'before save own weights separately.' %(__name__))

        filename = _extract_file_name(filename)
        _absolute_path = os.path.join(root_path, filename)       

        # Save own model's architecture to json
        model_json = self.to_json()
        with open(_absolute_path + _POSTFIX_NAME_ARCH_FILE, 'w') as json_file:
            json_file.write(model_json)

        seperation_layer_index = self._layers.index(
            self._separation_marked_layer)

        print("[%s] The separation layer is %s at [%d]"
              % (__name__, self._separation_marked_layer.name, seperation_layer_index))

        # Save non-trainable layers
        freezed_layers = self._layers[:seperation_layer_index]
        with h5py.File(_absolute_path + _POSTFIX_NAME_FREEZED_FILE, 'w') as f:
            saving.save_weights_to_hdf5_group(f, freezed_layers)

        # Save trainable layers
        fine_tuned_layers = self._layers[seperation_layer_index+1:]
        with h5py.File(_absolute_path + _POSTFIX_NAME_TUNED_FILE, 'w') as f:
            saving.save_weights_to_hdf5_group(f, fine_tuned_layers)

        return (
            filename + _POSTFIX_NAME_ARCH_FILE,
            filename + _POSTFIX_NAME_FREEZED_FILE,
            filename + _POSTFIX_NAME_TUNED_FILE
            )

    def set_separation_mark(self, layer=None):
        if layer is None or layer.name is None:
            raise ValueError('[%s] Invalid layer for marking.' %(__name__))

        if not isinstance(layer, Layer):
            raise ValueError('[%s] Invalid type of layer for marking.' %(__name__))

        if self.get_layer(name=layer.name) is None:
            raise ValueError('[%s] This model has not contain given layer.' %(__name__))

        self._separation_marked_layer = layer
