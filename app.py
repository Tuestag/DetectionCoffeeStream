import cv2
import numpy as np
import streamlit as st

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


#PROBANDO
import requests
requests.get("https://github.com/Tuestag", headers = {'User-agent': 'your bot 0.1'})

#response = requests.get("https://api.github.com/repos/Tuestag/DetectionCoffeeStream/releases/latest")

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)
@st.cache(persist=True)
def initialization():
    """Loads configuration and model for the prediction.
    
    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.
        
    """
    cfg = get_cfg()
    # Force model to operate within CPU, erase if CUDA compatible devices ara available
    cfg.MODEL.DEVICE = 'cpu'
    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
  #  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    # Set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well

    #cfg.MODEL.WEIGHTS = os.path.join("https://api.github.com/repos/Tuestag/DetectionCoffeeStream/releases/assets/49141342")
    cfg.MODEL.WEIGHTS = os.path.join("https://github.com/Tuestag/DetectionCoffeeStream/releases/download/CoffeeDetector/DA_faster_rcnn_R_50_FPN_1x.pth")
    
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("my_dataset_test", {}, "/test/_annotations.coco.json", "test")
    MetadataCatalog.get("my_dataset_test").thing_classes = ['CafPrueba', 'Pinton', 'Rojo', 'Sobremaduro', 'Verde']
    MetadataCatalog.get("my_dataset_test").thing_dataset_id_to_contiguous_id = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    # Initialize prediction model
    predictor = DefaultPredictor(cfg)

    return cfg, predictor


@st.cache
def inference(predictor, img):
    return predictor(img)


@st.cache
def output_image(cfg, img, outputs):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("my_dataset_test"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image()

    return processed_img


@st.cache
def discriminate(outputs, classes_to_detect):
    """Select which classes to detect from an output.
    Get the dictionary associated with the outputs instances and modify
    it according to the given classes to restrict the detection to them
    Args:
        outputs (dict):
            instances (detectron2.structures.instances.Instances): Instance
                element which contains, among others, "pred_boxes", 
                "pred_classes", "scores" and "pred_masks".
        classes_to_detect (list: int): Identifiers of the dataset on which
            the model was trained.
    Returns:
        ouputs (dict): Same dict as before, but modified to match
            the detection classes.
    """
    pred_classes = np.array(outputs['instances'].pred_classes)
    # Take the elements matching *classes_to_detect*
    mask = np.isin(pred_classes, ['CafPrueba', 'Pinton', 'Rojo', 'Sobremaduro', 'Verde'])
    # Get the indexes
    idx = np.nonzero(mask)

    # Get the current Instance values
    pred_boxes = outputs['instances'].pred_boxes
    pred_classes = outputs['instances'].pred_classes
    pred_masks = outputs['instances'].pred_masks
    scores = outputs['instances'].scores
    st.title(pred_classes)

    # Get them as a dictionary and leave only the desired ones with the indexes
    out_fields = outputs['instances'].get_fields()
    out_fields['pred_boxes'] = pred_boxes[idx]
    out_fields['pred_classes'] = pred_classes[idx]
    out_fields['pred_masks'] = pred_masks[idx]
    out_fields['scores'] = scores[idx]

    return outputs


def main():
    # Initialization
    cfg, predictor = initialization()

    # Streamlit initialization
    st.title("Instance Segmentation")

    # Retrieve image
    uploaded_img = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # Detection code
        outputs = inference(predictor, img)
        out_image = output_image(cfg, img, outputs)
        st.image(out_image, caption='Processed Image', use_column_width=True)        


if __name__ == '__main__':
    main()
