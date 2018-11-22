from ai_project import AIProject
from ai_project_loader import AIProjectLoader
import ai_config as config
import mongoengine as me
import os


class AIServiceImpl(object):

    def __init__(self):
        self._project_loader = AIProjectLoader()
        self._prepare()

    def create_project(self, project_name, model_type):
        if not self._project_loader.exist_project(project_name):
            return self._project_loader.create_project(project_name, model_type)
        else:
            return self._project_loader.get_project(project_name)

    def remove_project(self, project_name):
        self._validate_project(project_name)
        return self._project_loader.remove_project(project_name)

    def add_image(self, project_name, image, label):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().add_image(image, label)

    def add_images(self, project_name, image_list, label_list):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().add_images(image_list, label_list)

    def get_image_label_list(self, project_name):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().get_image_label_list()

    def get_image_label(self, project_name, image_name, image_index):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().get_image_label(image_name, image_index)

    def remove_image(self, project_name, image_name, image_index):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().remove_image(image_name, image_index)

    def remove_all_image(self, project_name):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().remove_all_image()

    def get_image_store_path(self, project_name):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().get_path()

    def add_label_lut(self, project_name, label_index, label_description):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().add_label_lut(label_index, label_description)

    def add_label_lut_list(self, project_name, sorted_label_description):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().add_label_lut_list(sorted_label_description)

    def get_label_description(self, project_name, label_index):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().get_label_description(label_index)

    def get_label_lut_list(self, project_name):
        self._validate_project(project_name)
        return self._project_loader.get_project(
            project_name).get_image_store().get_label_lut_list()

    def train(self, project_name, **kwargs):
        self._validate_project(project_name)
        _ai_project = self._project_loader.get_project(project_name)
        _ai_project.get_serve_model().train(
            image_store=_ai_project.get_image_store(), kwargs=kwargs)
        _ai_project.get_serve_model().save()

    def predict(self, project_name, image_name, image_index, **kwargs):
        self._validate_project(project_name)
        _ai_project = self._project_loader.get_project(project_name)
        _ai_project.get_serve_model().predict(
            image_store=_ai_project.get_image_store(),
            image_name=image_name,
            image_index=image_index,
            **kwargs)

    def get_score(self, project_name, score_type, **kwargs):
        self._validate_project(project_name)
        self._project_loader.get_project(
            project_name).get_serve_model().get_score(kwargs=kwargs)

    def _validate_project(self, project_name):
        if not self._project_loader.exist_project(project_name):
            raise ValueError(
                '[%s] %s is not exist.' % (__name__, project_name))

    def _prepare(self):
        self._make_essential()
        alias = me.connect(
            db=config.DB_PROPERTIES[config.DB_PROP_DB_NAME],
            host=config.DB_PROPERTIES[config.DB_PROP_ADDRESS],
            port=config.DB_PROPERTIES[config.DB_PROP_PORT],
            username=config.DB_PROPERTIES[config.DB_PROP_USER_NAME],
            password=config.DB_PROPERTIES[config.DB_PROP_PASSWORD],
        )
        print(alias)

    def _make_essential(self):
        # Make a directory for projects
        if not os.path.exists(config.SERVING_PROPERTIES[config.SERVING_PROP_PROJECT_ROOT]):
            os.makedirs(config.SERVING_PROPERTIES[config.SERVING_PROP_PROJECT_ROOT])
            print('[%s] %s' %(__name__,
                'Make dir ' + config.SERVING_PROPERTIES[config.SERVING_PROP_PROJECT_ROOT]))

