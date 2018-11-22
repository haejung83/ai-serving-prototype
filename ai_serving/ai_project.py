from ODM.odm_ai_project import ODMAIProject, create_odm_ai_project
from image_store import ImageStore
from serve_model import ServeModel, ServeModelType
import ai_config as config
import os


class AIProject(object):
    def __init__(self, odm):
        self._odm = odm
        self._serve_model = None
        self._image_store = None
        self._build()

    @staticmethod
    def create_from_odm(odm_ai_project):
        return AIProject(odm_ai_project)

    @staticmethod
    def delete_by_odm(odm_ai_project):
        if isinstance(odm_ai_project, ODMAIProject):
            ImageStore.delete_by_odm(odm_ai_project.image_store)
            ServeModel.delete_by_odm(odm_ai_project.serve_model)
            odm_ai_project.delete()
        else:
            raise ValueError('[%s] %s' %(__name__,
                "The object is not ODMAIProject instnace. So couldn't delete it"))

    @staticmethod
    def create(name, hash_key, model_type):
        model_type = model_type if model_type else ServeModelType.Generic
        # FIXME(haejung): Check locations to create both ODM object at create_odm_ai_project()
        return AIProject.create_from_odm(
            create_odm_ai_project(name, hash_key, model_type))

    def _build(self):
        self._make_essential()
        self._serve_model = ServeModel.create_from_odm(self._odm.serve_model)
        self._image_store = ImageStore.create_from_odm(self._odm.image_store)

    def _make_essential(self):
        _project_path = os.path.join(
            config.SERVING_PROPERTIES[config.SERVING_PROP_PROJECT_ROOT],
            self._odm.hash_key)

        if not os.path.exists(_project_path):
            os.makedirs(_project_path)
            print('[%s] %s' %(__name__,
                'Make dir ' + _project_path))

    def get_name(self):
        return self._odm.name

    def get_hash_key(self):
        return self._odm.hash_key

    def get_serve_model(self):
        return self._serve_model

    def get_image_store(self):
        return self._image_store
