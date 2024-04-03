#imports
from tokenize import Double
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import glob
import argparse
import pathlib
import os
from tqdm import tqdm
from re import X
from random import sample
from multiprocessing import Pool

# detectron2 utils
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config
from detectron2.utils.visualizer import ColorMode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

# validation set
val = []

# size of traning set
trainSize = 0

# prepares data for any task
def get_wood_dicts(i_dir, ratio):

    # load jsons ground truth
    img_files = glob.glob(os.path.join(i_dir,'jsons','*.json'))
    dataset_dicts = []
    print("loading dataset")

    global val, trainSize

    # creating validation set
    i2 = str(i_dir).replace("d", "t")
    val = glob.glob(os.path.join(i2,'jsons','*.json'))

    # creating training set
    img_files =  [i for i in img_files if i not in val]
    trainSize = len(img_files)

    # for each jsons
    for img_dir in tqdm(img_files):
        json_file = os.path.join(img_dir)

        # open json
        with open(json_file) as f:
            imgs_anns = json.load(f)

        # get fields
        for idx, v in enumerate(imgs_anns.values()):
            record = {}
            
            # get associated image path
            filename = os.path.join(i_dir, 'org', v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            
            # path
            record["file_name"] = filename

            # doesn't matter
            record["image_id"] = idx

            # height
            record["height"] = height

            # width
            record["width"] = width
        
            # mask
            annos = v["regions"]
            objs = []

            # get mask properties
            for _, anno in annos.items():
                assert not anno["region_attributes"]

                # polygon
                anno = anno["shape_attributes"]

                # list of points values on x axis
                px = anno["all_points_x"]

                # list of points values on y axis
                py = anno["all_points_y"]

                # creation of points
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                # box attribute (relevant for instance segmentation)
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs

            # add to training set
            dataset_dicts.append(record)
    return dataset_dicts

# registers the dataset
def register(d, ratio): 
    DatasetCatalog.register("wood_train", lambda d=d: get_wood_dicts(d, ratio))
    MetadataCatalog.get("wood_train").set(thing_classes=["wood"])

# create pointRend model ready to be trained with given starting weight
def PointRend(args):
    global trainSize

    # creation of cfg
    cfg = get_cfg()

    # mandatory for pointRend
    add_pointrend_config(cfg)

    # get specs
    cfg.merge_from_file("projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml")
    
    # train dataset (relevant only to train)
    cfg.DATASETS.TRAIN = ("wood_train",)
    cfg.DATASETS.TEST = ()

    # nb GPU to put on the upcomming task
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    # loading pretrained weights
    cfg.MODEL.WEIGHTS = "projects/PointRend/model_final_ba17b9.pkl"

    # set batch size
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size

    # set learning rate
    cfg.SOLVER.BASE_LR = args.base_lr

    # set max number of iter per epoch
    cfg.SOLVER.MAX_ITER = (trainSize//args.batch_size)+1
    cfg.SOLVER.STEPS = []        

    # ROI head size
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size_per_img

    # nb class
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1

    os.makedirs(args.logdir, exist_ok=True)
    # checkpoints dir
    cfg.OUTPUT_DIR = os.path.join(args.logdir, args.model+args.logname)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

def write_res(out):
    # get mask
    res = out[0]
    img_orig = out[2]
    name = out[1]

    if(res.shape[0]>0):

        res = np.swapaxes(res,0,2)
        res = np.swapaxes(res,0,1)
        res = res[:, :, :1]
        res = np.squeeze(res)

        res = res.astype(np.uint8)

        output = cv2.connectedComponentsWithStats(res, 8, cv2.CV_32S)
        
        labels = output[1]
        stats = output[2]

        maxi = 0
        maxL = 0
        for i in range(1, len(stats), 1):
            if(stats[i][4]>maxi):
                maxi = stats[i][4]
                maxL = i

        #print(maxL)
        res[labels!=maxL]=0

        res = (np.ones(res.shape)-res).astype(np.uint8)

        output = cv2.connectedComponentsWithStats(res, 8, cv2.CV_32S)

        labels = output[1]
        stats = output[2]

        xa = stats[0][0]
        xA = xa + stats[0][2]
        ya = stats[0][1]
        yA = ya + stats[0][3]

        supp = []

        for i in range(1, len(stats), 1):

            x1 = stats[i][0]
            x2 = x1 + stats[i][2]
            y1 = stats[i][1]
            y2 = y1 + stats[i][3]

            if(xa<x1 and ya<y1 and xA>x2 and yA>y2):
                supp.append(i)
        
        for s in supp:
            res[labels==s]=0

        #print(stats)
        res = (np.ones(res.shape)-res).astype(np.uint8)

        # mask segmentation
        img_orig[res==0] = (0, 0, 0)

        # save both outputs
        cv2.imwrite("database/imgs/"+name.split('/')[len(name.split('/'))-1], img_orig)
       

# inference
def inference(cfg, args):

    # path
    dir = args.datadir

    # load model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.pth_name)

    # threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    # create predictor
    predictor = DefaultPredictor(cfg)

    # no point
    wood_metadata = MetadataCatalog.get("wood_train")
    print("inferences")

    # get every image in the folder
    img_files = glob.glob(os.path.join(dir,'*.jpg'))

    # for each not annotated images
    for d in tqdm(img_files):    
        im = cv2.imread(d)

        # compute mask
        outputs = predictor(im)

        # get mask
        res = outputs['instances'][outputs['instances'].pred_classes==0].pred_masks.cpu().numpy().astype(int)
        
        write_res([res, d, im])

# inference
def batch_inference(cfg, args):

    # path
    dir = args.datadir

    # load model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.pth_name)

    # threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    model = build_model(cfg)
    model_dict = torch.load(os.path.join(cfg.OUTPUT_DIR, args.pth_name))
    model.load_state_dict(model_dict['model'] )
    model.train(False) 

    # no point
    wood_metadata = MetadataCatalog.get("wood_train")
    print("inferences")

    # get every image in the folder
    img_files = glob.glob(os.path.join(dir,'testicle','*'))

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    #print(torch.cuda.memory_allocated())
    for j in tqdm(range(0, len(img_files), args.batch_size)):
        img_origs = []
        inputs = []
        # for each not annotated images
        #print("load")

        torch.cuda.empty_cache()
        #print(torch.cuda.memory_allocated())


        for welp in range(j, min(j+args.batch_size, len(img_files)), 1):   
            d = img_files[welp] 
            im = cv2.imread(d)
            img_origs.append((d,im))
            h, w = im.shape[:2]
            im = aug.get_transform(im).apply_image(im)

            img = np.transpose(im,(2,0,1))

            img_tensor = torch.from_numpy(img)
            inputs.append({"image":img_tensor, "height":h, "width":w})
            del img_tensor


        # compute mask
        #print(inputs)
        out = model(inputs)
        out2 = []

        i=0
        for outputs in out:
            res = outputs['instances'][outputs['instances'].pred_classes==0].pred_masks.cpu().numpy().astype(int)
            out2.append((res, img_origs[i][0], img_origs[i][1]))
            i+=1

        #print(os.cpu_count())
        pool = Pool(min(args.batch_size,os.cpu_count()))
        pool.map(write_res, out2)
        #print("write")

        for outputs in out:
            del outputs
        for inp in inputs:
            del inp
        for outputs in out2:
            del outputs
        del out
        del out2
        del inputs
        
        

if __name__ == "__main__":

    # list of supported models in this program
    available_models = {"PointRend" : PointRend, "Mask_R-CNN" : MRCNN}

    parser = argparse.ArgumentParser()

    # choice of function to execute 
    parser.add_argument("command", choices=["train", "test", "debug", "inference", "batch_inference"], default="train")

    # dir to save model
    parser.add_argument("--logdir", type=pathlib.Path, default="logs")

    # dir of dataset
    parser.add_argument("--datadir", type=pathlib.Path, default='DATA2')

    # nb of GPU on task
    parser.add_argument("--num_workers", type=int, default=0)

    # batch size
    parser.add_argument("--batch_size", type=int, default=2)

    # model
    parser.add_argument("--model", choices=available_models, default="PointRend")

    # Training parameters

    # name of model
    parser.add_argument("--logname", type=str, default="modeleTest1")

    # ratio training Val
    parser.add_argument("--ratio", type=float, default=0)

    # nb of epochs 
    parser.add_argument("--nepochs", type=int, default=300)

    # learning rate
    parser.add_argument("--base_lr", type=float, default=0.00025)

    # ROI size
    parser.add_argument("--batch_size_per_img", type=int, default=512)

    # Inference parameters

    # not using it in the end
    parser.add_argument("--test_dir", type=pathlib.Path, default="test_dataset")

    # name of the .pth file to use 
    parser.add_argument("--pth_name", type=pathlib.Path, default="model_final.pth")

    args = parser.parse_args()

    # register dataset
    register(args.datadir, args.ratio)

    # get desired model
    model = available_models[args.model]

    # prepare said model
    cfg = model(args)
    
    # execute desired function
    exec(f"{args.command}(cfg, args)")
