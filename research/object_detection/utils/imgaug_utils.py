import imgaug.augmenters as iaa
import numpy as np
import tensorflow.compat.v1 as tf
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tensorflow.python.framework.ops import EagerTensor
from object_detection.protos import preprocessor_pb2


def _proto2dict(msg):
    hsh_val = dict(msg.ListFields())
    return dict(
        (k, hsh_val[hsh])
        for k, hsh in msg.DESCRIPTOR.fields_by_name.items()
        if hsh in hsh_val
    )


def _get_imgaug_seq(imgaug_seq_msg: dict):
    imgaug_config_dict = _proto2dict(imgaug_seq_msg)
    func_names = map(
        lambda key: "".join(x.capitalize() or "_" for x in key.split("_")),
        imgaug_config_dict.keys(),
    )
    func_parameter_dicts = map(_proto2dict, imgaug_config_dict.values())
    imgaug_func_list = list(
        map(
            lambda f_name, parameters: getattr(iaa, f_name)(**parameters),
            func_names,
            func_parameter_dicts,
        )
    )
    return iaa.Sequential(imgaug_func_list, random_order=True)


def get_augment_func(imgaug_seq_msg: preprocessor_pb2.AugmentFuncs):
    imgaug_seq = _get_imgaug_seq(imgaug_seq_msg)

    def augment(image, locations, labels):
        image_np, locations_np, labels_np = map(
            lambda tensor: tensor.numpy(), [image, locations, labels]
        )
        h, w, c = image_np.shape
        locations_np = np.concatenate(
            (locations_np[:, 1::2] * w, locations_np[:, ::2] * h), axis=1
        )[:, [0, 2, 1, 3]]
        bbox_np = np.concatenate(
            (locations_np, np.expand_dims(labels_np, -1)),
            axis=1
        )
        bboxes = BoundingBoxesOnImage(
            [BoundingBox(*bbox) for bbox in bbox_np], shape=(h, w, c)
        )
        image_aug_np, bboxes_aug = imgaug_seq(
            image=image_np.astype(np.uint8), bounding_boxes=bboxes
        )
        bboxes_aug_np = np.array(
            [
                *map(
                    lambda box: [*vars(box).values()],
                    bboxes_aug.remove_out_of_image().clip_out_of_image(),
                )
            ]
        )
        if bboxes_aug_np.size != 0:
            locations_aug_np = np.concatenate(
                (bboxes_aug_np[:, :-1:2] / w, locations_np[:, 1::2] / h), axis=1
            )[:, [2, 0, 3, 1]]
            labels_aug_np = bboxes_aug_np[:, -1]
        else:
            locations_aug_np = np.reshape(bboxes_aug_np, [0, 4])
            labels_aug_np = np.reshape(bboxes_aug_np, [0, 1])
        return (
            tf.convert_to_tensor(image_aug_np, dtype=tf.float32),
            tf.convert_to_tensor(locations_aug_np, dtype=tf.float32),
            tf.convert_to_tensor(labels_aug_np, dtype=tf.int64),
        )

    return augment
