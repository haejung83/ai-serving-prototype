from mongoengine import Document, StringField, ListField, IntField, ReferenceField


class ODMServeModel(Document):
    hash_key = StringField(max_length=36, required=True)
    model_type = IntField(required=True)
    num_classes = IntField(required=False)
    architecture_path = StringField(max_length=128, required=False)
    freezed_weight_path = StringField(max_length=128, required=False)
    tuned_weight_path = StringField(max_length=128, required=False)


def create_odm_serve_model(
        hash_key,
        model_type,
        architecture_path=None,
        freezed_weight_path=None,
        tuned_weight_path=None):
    _new_odm_serve_model = ODMServeModel()
    _new_odm_serve_model.hash_key = hash_key
    _new_odm_serve_model.model_type = model_type

    if architecture_path:
        _new_odm_serve_model.architecture_path = architecture_path
    if freezed_weight_path:
        _new_odm_serve_model.freezed_weight_path = freezed_weight_path
    if tuned_weight_path:
        _new_odm_serve_model.tuned_weight_path = tuned_weight_path

    _new_odm_serve_model.save()
    return _new_odm_serve_model
