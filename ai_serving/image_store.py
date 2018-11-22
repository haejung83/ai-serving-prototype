from ODM.odm_image_store import ODMImageStore, create_odm_image_store
from ODM.odm_image_store import ODMImageLabel, create_odm_image_label
import ai_config as config
import os


class ImageLabel(object):
    def __init__(self, odm):
        self._odm = odm

    @staticmethod
    def create_from_odm(odm_image_label):
        return ImageLabel(odm_image_label)

    @staticmethod
    def delete_by_odm(odm_image_label):
        if isinstance(odm_image_label, ODMImageLabel):
            odm_image_label.delete()
        else:
            raise ValueError(
                "The object is not ODMImageLabel instnace. So couldn't delete it")

    @staticmethod
    def create(image_name, label):
        return ImageLabel.create_from_odm(create_odm_image_label(image_name, label))

    def get_image_name(self):
        return self._odm.image_name

    def get_label(self):
        return self._odm.label

    @property
    def odm(self):
        return self._odm


class ImageStore(object):
    def __init__(self, odm):
        self._odm = odm
        self._path = None
        self._image_labels = []
        self._label_lut = []
        self._build()

    @staticmethod
    def create_from_odm(odm_image_store):
        return ImageStore(odm_image_store)

    @staticmethod
    def delete_by_odm(odm_image_store):
        if isinstance(odm_image_store, ODMImageStore):
            if odm_image_store.image_labels:
                for odm_image_label in odm_image_store.image_labels:
                    ImageLabel.delete_by_odm(odm_image_label)

            odm_image_store.delete()
        else:
            raise ValueError('[%s] %s' %(__name__,
                "The object is not ODMImageStore instnace. So couldn't delete it"))

    @staticmethod
    def create(hash_key, label_lut):
        return ImageStore.create_from_odm(create_odm_image_store(hash_key, label_lut))

    def _build(self):
        self._make_essential()

        if self._odm.image_labels:
            for odm_image_label in self._odm.image_labels:
                self._image_labels.append(
                    ImageLabel.create_from_odm(odm_image_label))

        if self._odm.label_lut:
            self._label_lut = self.get_label_lut_list()

    def _make_essential(self):
        _image_store_path = os.path.join(
            config.SERVING_PROPERTIES[config.SERVING_PROP_PROJECT_ROOT],
            self._odm.hash_key,
            config.SERVING_PROPERTIES[config.SERVING_PROP_IMAGE_STORE_PATH])

        if not os.path.exists(_image_store_path):
            os.makedirs(_image_store_path)
            print('[%s] %s' %(__name__,
                'Make dir ' + _image_store_path))

        self._path = _image_store_path

    def add_image(self, image_name, label):
        _odm_image_label = create_odm_image_label(image_name, label)
        self._odm.image_labels.append(_odm_image_label)
        self._odm.save()
        self._image_labels.append(ImageLabel.create_from_odm(_odm_image_label))

    def add_images(self, image_list, label_list):
        for image, label in zip(image_list, label_list):
            self.add_image(image, label)

    def get_image_label(self, image_name=None, image_index=None):
        if image_index is not None and image_index < len(self._image_labels):
            return self._image_labels[image_index]
        elif image_name:
            # FIXME(haejung): Should make more efficient way to retrieve child.
            _found_image = None
            for image_label in self._image_labels:
                if image_label.image_name == image_name:
                    _found_image = image_label
                    break
            return _found_image
        else:
            raise ValueError(
                "[%s] Empty parameter, Please check parameters" % (__name__))

    def get_image_path(self, image_name=None, image_index=None):
        _image_label = self.get_image_label(image_name, image_index)
        if _image_label:
            return os.path.join(self._path, _image_label.get_image_name())
        else:
            return None

    def get_image_label_list(self):
        # FIXME(haejung): Typically, It should return non-editable object.
        return self._image_labels

    def remove_image(self, image_name=None, image_index=None):
        _image_label = self.get_image_label(image_name, image_index)
        if _image_label:
            self._image_labels.remove(_image_label)
            _odm_image_label = _image_label.odm
            self._odm.image_labels.remove(_odm_image_label)
            self._odm.save()
            ImageLabel.delete_by_odm(_odm_image_label)

    def remove_all_image(self):
        for odm_image_label in self._odm.image_labels:
            ImageLabel.delete_by_odm(odm_image_label)
        self._odm.image_labels = []
        self._odm.save()
        self._image_labels = []

    def get_path(self):
        return self._path

    def add_label_lut(self, label_index, label_description):
        self._odm.label_lut.append('%s,%s' % (label_index, label_description))
        self._odm.save()
        self._label_lut = self.get_label_lut_list()

    def add_label_lut_list(self, sorted_label_description):
        for index, label in enumerate(sorted_label_description):
            self._odm.label_lut.append('%s,%s' % (index, label))
        self._odm.save()
        self._label_lut = self.get_label_lut_list()

    def get_label_description(self, label_index=None):
        if label_index is not None:
            return self._label_lut[label_index]
        else:
            raise ValueError(
                "[%s] Empty parameter, Please check parameters" % (__name__))

    def get_label_lut_list(self):
        _label_lut = []
        for encoded_lut in self._odm.label_lut:
            decoded_lut = encoded_lut.split(',')
            _label_lut.insert(int(decoded_lut[0]), decoded_lut[1])
        return _label_lut
