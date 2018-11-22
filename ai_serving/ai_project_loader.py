from ai_project import AIProject
from ODM.odm_ai_project import ODMAIProject
from utiils import get_random_hash_key


class AIProjectLoader(object):
    def __init__(self):
        self._cache = dict()

    def create_project(self, name, model_type):
        if not self._cache.get(name):
            self._cache[name] = AIProject.create(
                name, get_random_hash_key(), model_type)
        return self._cache[name]

    def get_project(self, name):
        if not self._cache.get(name):
            self._cache[name] = AIProject.create_from_odm(
                self._get_odm_ai_project(name))
        return self._cache[name]

    def remove_project(self, name):
        AIProject.delete_by_odm(self._get_odm_ai_project(name))
        if self._cache.get(name):
            del self._cache[name]

    def exist_project(self, name):
        return len(ODMAIProject.objects(name=name)) is not 0

    def _get_odm_ai_project(self, name):
        _odm_ai_projects = ODMAIProject.objects(name=name)
        if _odm_ai_projects and len(_odm_ai_projects) > 0:
            return _odm_ai_projects[0]
        else:
            raise ValueError(
                '[%s] There is not exist a project with a given name [%s]' %(__name__, name))
