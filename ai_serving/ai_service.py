from ai_service_impl import AIServiceImpl


def _check_validation(**kwargs):
    for key, value in kwargs.items():
        if value is None:
            raise ValueError('[%s] The %s is not acceptable' %(__name__, key))


class AIService(object):

    def __init__(self):
        # Using bridge pattern for extension
        self._impl = AIServiceImpl()

    def create_project(self, project_name=None, model_type=None):
        _check_validation(**({'project_name': project_name}))
        return self._impl.create_project(
            project_name=project_name,
            model_type=model_type)

    def remove_project(self, project_name=None):
        _check_validation(**({'project_name': project_name}))
        return self._impl.remove_project(
            project_name=project_name)

    def add_image(self, project_name=None, image=None, label=None):
        _check_validation(
            **({'project_name': project_name, 'image': image, 'label': label}))
        return self._impl.add_image(
            project_name=project_name,
            image=image,
            label=label)

    def add_images(self, project_name=None, image_list=None, label_list=None):
        _check_validation(
            **({'project_name': project_name, 'image_list': image_list}))
        return self._impl.add_images(
            project_name=project_name,
            image_list=image_list,
            label_list=label_list)

    def get_image_lable_list(self, project_name=None):
        _check_validation(**({'project_name': project_name}))
        return self._impl.get_image_label_list(
            project_name=project_name)

    def get_image_label(self, project_name=None, image_name=None, image_index=None):
        _check_validation(
            **({'project_name': project_name, 'image_name': image_name}))
        return self._impl.get_image_label(
            project_name=project_name,
            image_name=image_name,
            image_index=image_index)

    def remove_image(self, project_name=None, image_name=None, image_index=None):
        _check_validation(
            **({'project_name': project_name, 'image_name': image_name}))
        return self._impl.remove_image(
            project_name=project_name,
            image_name=image_name,
            image_index=image_index)

    def remove_all_image(self, project_name=None):
        _check_validation(**({'project_name': project_name}))
        return self._impl.remove_all_image(
            project_name=project_name)

    def get_image_store_path(self, project_name=None):
        _check_validation(**({'project_name': project_name}))
        return self._impl.get_image_store_path(
            project_name=project_name)

    def add_label_lut(self, project_name=None, label_index=None, label_description=None):
        _check_validation(
            **({'project_name': project_name, 'label_index': label_index, 'label_desc': label_description}))
        return self._impl.add_label_lut(
            project_name=project_name,
            label_index=label_index,
            label_description=label_description)

    def add_label_lut_list(self, project_name=None, sorted_label_description=None):
        _check_validation(
            **({'project_name': project_name, 'sorted_label_desc': sorted_label_description}))
        return self._impl.add_label_lut_list(
            project_name=project_name,
            sorted_label_description=sorted_label_description)

    def get_label_description(self, project_name=None, label_index=None):
        _check_validation(
            **({'project_name': project_name, 'label_index': label_index}))
        return self._impl.get_label_description(
            project_name=project_name,
            label_index=label_index)

    def get_label_lut_list(self, project_name=None):
        _check_validation(**({'project_name': project_name}))
        return self._impl.get_label_lut_list(
            project_name=project_name)

    def train(self, project_name=None, **kwargs):
        _check_validation(**({'project_name': project_name}))
        return self._impl.train(
            project_name=project_name,
            **kwargs)

    def predict(self, project_name=None, image_name=None, image_index=None, **kwargs):
        _check_validation(**({'project_name': project_name}))
        if not image_name and image_index is None:
            _check_validation(
                **({'image_name': image_name, 'image_index': image_index}))
        return self._impl.predict(
            project_name=project_name,
            image_name=image_name,
            image_index=image_index,
            **kwargs)

    # Availiable Score Type: Precision, Recall, F1score
    def get_score(self, project_name=None, score_type=None, **kwargs):
        _check_validation(
            **({'project_name': project_name, 'score_type': score_type}))
        return self._impl.get_score(project_name=project_name,
                                    score_type=score_type,
                                    **kwargs)
