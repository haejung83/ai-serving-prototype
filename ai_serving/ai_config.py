import os


# Properties about Database
DB_PROP_ADDRESS = 'address'
DB_PROP_PORT = 'port'
DB_PROP_DB_NAME = 'dbname'
DB_PROP_USER_NAME = 'user'
DB_PROP_PASSWORD = 'password'
DB_PROPERTIES = {
    DB_PROP_ADDRESS: '127.0.0.1',
    DB_PROP_PORT: 27017,
    DB_PROP_DB_NAME: 'test',
    DB_PROP_USER_NAME: 'haejung',
    DB_PROP_PASSWORD: '1qaz2wsx',
}


# Properties about Serving
SERVING_PROP_MAX_MODEL_COUNT = 'count_serve_model_max'
SERVING_PROP_PROJECT_ROOT = 'project_root'
SERVING_PROP_IMAGE_STORE_PATH = 'image_store_path'
SERVING_PROP_SERVE_MODEL_PATH = 'serve_model_path'
SERVING_PROPERTIES = {
    SERVING_PROP_MAX_MODEL_COUNT: 2,
    SERVING_PROP_PROJECT_ROOT: os.path.join(
        os.path.dirname(__file__), 'projects'),
    SERVING_PROP_IMAGE_STORE_PATH: 'image_store',
    SERVING_PROP_SERVE_MODEL_PATH: 'serve_model',
}


def load_from_file(filepath):
    raise NotImplementedError('load_from_file method not implemented yet')
