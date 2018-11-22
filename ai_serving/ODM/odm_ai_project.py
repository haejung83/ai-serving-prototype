from mongoengine import Document, StringField, ListField, IntField, ReferenceField
from ODM.odm_serve_model import ODMServeModel, create_odm_serve_model
from ODM.odm_image_store import ODMImageStore, create_odm_image_label
from ODM.odm_image_store import create_odm_image_store


class ODMAIProject(Document):
    name = StringField(max_length=128, required=True)
    hash_key = StringField(max_length=36, required=True)
    serve_model = ReferenceField(ODMServeModel, required=False)
    image_store = ReferenceField(ODMImageStore, required=False)


def create_odm_ai_project(name, hash_key, model_type):
    _new_project = ODMAIProject()
    _new_project.name = name
    _new_project.hash_key = hash_key
    # FIXME(haejung): Check these code's location
    _new_project.image_store = create_odm_image_store(hash_key)
    _new_project.serve_model = create_odm_serve_model(hash_key, model_type)
    _new_project.save()
    return _new_project
