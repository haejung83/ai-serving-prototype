from serve_model import ServeModelType, ServeModel
from utiils import get_random_hash_key

class ModelBuilder(object):
    
    @staticmethod
    def create_model(serve_model_type=None):
        if not serve_model_type:
            raise ValueError('[%s] %s' %(__name__,
             'Must give a model type before call create_model method'))

        new_hash_key = get_random_hash_key()
        return ServeModel.create(new_hash_key, serve_model_type)
   