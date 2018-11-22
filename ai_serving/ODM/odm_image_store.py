from mongoengine import Document, StringField, ListField, IntField, ReferenceField
import time

class ODMImageLabel(Document):
    image_name = StringField(max_length=128, required=True)
    label = IntField(min_value=0, required=True)


def create_odm_image_label(image_name, label):
    _new_image_label = ODMImageLabel()
    _new_image_label.image_name = image_name
    _new_image_label.label = label
    _new_image_label.save()
    return _new_image_label


class ODMImageStore(Document):
    hash_key = StringField(max_length=36, required=True)
    # Lookup Table[LUT] = (index,label) ex(10000,flower)
    label_lut = ListField(StringField(max_length=64))
    image_labels = ListField(ReferenceField(ODMImageLabel, required=False))


def create_odm_image_store(hash_key, label_lut=None):
    _new_image_store = ODMImageStore()
    _new_image_store.hash_key = hash_key
    if label_lut:
        _new_image_store.label_lut = label_lut
    _new_image_store.save()
    return _new_image_store
