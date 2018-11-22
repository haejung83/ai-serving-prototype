
from ai_service import AIService
from serve_model import ServeModelType
from ODM.odm_ai_project import ODMAIProject
from ODM.odm_image_store import ODMImageLabel, ODMImageStore
from ODM.odm_serve_model import ODMServeModel

from utiils import get_random_hash_key
import ai_config as config
import mongoengine as me
import os
import argparse


TRAIN_IMAGE_LUT = [
    (0, 'rose'),
    (1, 'iris'),
    (2, 'sunflower'),
]
TRAIN_IMAGE_LIST = [
    ('0000.jpg', 0),
    ('0001.jpg', 1),
    ('0002.jpg', 2),
    ('0003.jpg', 0),
    ('0004.jpg', 1),
    ('0005.jpg', 2),
    ('0006.jpg', 0),
    ('0007.jpg', 1),
    ('0008.jpg', 2),
    ('0009.jpg', 0),
    ('0010.jpg', 1),
    ('0011.jpg', 2),
]


def test_ai_service(args):
    print('Test the AI Service\n')
    project_name = 'cvt_net'

    if args.project_name:
        project_name = str(args.project_name)

    print("Project Name: " + project_name)

    _service = AIService()
    print('Create AIService object\n')

    test_ai_service_creation(_service, project_name)

    # Check validation of added images
    test_ai_service_validation_images(_service, project_name)

    # Check validation of training, prediction, get_score
    if args.test_type and not "image" in args.test_type:
        test_ai_service_validation_serve_model(_service, project_name, args.test_type)

    if args.with_deletion.lower() == "true":
        test_ai_service_deletion(_service, project_name)


def test_ai_service_creation(service, project_name):
    # Create a project
    hash_key = service.create_project(
        project_name=project_name,
        model_type=ServeModelType.Generic)
    print('Create AIProject successfully %s\n' % (hash_key))

    # Check the root path of image storage
    image_store_path = service.get_image_store_path(project_name)
    print('Image Store Path: %s\n' % (image_store_path))

    # Add label LUT
    label_lut_list = service.get_label_lut_list(project_name)
    if label_lut_list and len(label_lut_list):
        print('Label LUT already exist in ImageStore')
    else:
        print('Label LUT is empty in ImageStore')
        for _label_index, _label_desc in TRAIN_IMAGE_LUT:
            print('Add label LUT index: %d, desc %s' % (_label_index, _label_desc))
            service.add_label_lut(project_name, _label_index, _label_desc)

    # Add images if not exist
    image_label_list = service.get_image_lable_list(project_name)
    if image_label_list and len(image_label_list) > 0:
        print('Images already exist in ImageStore')
    else:
        print('The ImageStore is empty')
        for image_name, label in TRAIN_IMAGE_LIST:
            print('Add image %s, label %d' % (image_name, label))
            service.add_image(project_name, image_name, label)

        # Prepare for adding images
        image_list = list()
        label_list = list()
        for _image, _label in TRAIN_IMAGE_LIST:
            image_list.append(_image)
            label_list.append(_label)
        
        # service.add_images(project_name, image_list, label_list) 


def test_ai_service_validation_images(service, project_name):
    label_lut_list = service.get_label_lut_list(project_name)
    label_lut_dict = dict()

    # Rearrange the label LUT
    if not label_lut_list or len(label_lut_list) == 0:
        print('Label LUT is not exist in ImageStore')
    else:
        for index, lut in enumerate(label_lut_list):
            label_lut_dict[index] = lut
            print('Loaded label LUT [%d] = %s' %(index, lut))

    image_label_list = service.get_image_lable_list(project_name)

    if not image_label_list or len(image_label_list) == 0:
        print('Image label is not exist in ImageStore')
    else:
        for index, image_label in enumerate(image_label_list):
            print("[%d] %s [%s]"
            %(index,
            image_label.get_image_name(),
            label_lut_dict[image_label.get_label()]))


def test_ai_service_validation_serve_model(service, project_name, test_type):
    if "train" in test_type:
        service.train(project_name)
    elif "prediction" in test_type:
        service.predict(
            project_name,
            image_index=1,
            desc=True,
            callback=callback_predict
        )
    elif "get_score" in test_type:
        score = service.get_score(project_name)
        print("Get Score " + score)
    else:
        raise ValueError("This serve model couldn't support test type of " + test_type)

def callback_predict(result):
    print(result)

def test_ai_service_deletion(service, project_name):
    service.remove_all_image(project_name)
    print('Remove all added image successfully %s\n' % (project_name))

    service.remove_project(project_name=project_name)
    print('Remove AIProject successfully %s\n' % (project_name))


def test_mongodb():
    print('Test the MongoDB\n')
    alias = me.connect(
        db=config.DB_PROPERTIES[config.DB_PROP_DB_NAME],
        host=config.DB_PROPERTIES[config.DB_PROP_ADDRESS],
        port=config.DB_PROPERTIES[config.DB_PROP_PORT],
        username=config.DB_PROPERTIES[config.DB_PROP_USER_NAME],
        password=config.DB_PROPERTIES[config.DB_PROP_PASSWORD],
    )
    print(alias)

    _ai_project = ODMAIProject()
    _ai_project.name = 'leanmaker_prediction'
    _ai_project.hash_key = get_random_hash_key()
    print(_ai_project.hash_key)
    print(len(_ai_project.hash_key))

    _ai_project.serve_model = ODMServeModel()
    _ai_project.serve_model.model_type = 0
    _ai_project.serve_model.save()

    _ai_project.image_store = ODMImageStore()
    _ai_project.image_store.label_lut = ['0,car', '1,ship', '2,house']
    _ai_project.image_store.save()

    _ai_project.save()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--test_type", default="image")
    arg_parser.add_argument("--project_name")
    arg_parser.add_argument("--with_deletion", default="False")
    args = arg_parser.parse_args()

    try:
        test_ai_service(args)
    except Exception as e: 
        print("[Exception] An exception occured as below, please check it")
        print(e)
    
    # test_mongodb()
