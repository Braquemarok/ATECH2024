# lightglue utils
from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import load_image, resize_image, numpy_image_to_torch, rbd
from lightglue import viz2d

# imports
import glob
import os
import matplotlib
from tqdm import tqdm
import torch
import cv2
from pathlib import Path
import numpy as np
import argparse
import math
import pickle
import gc

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# read image and returns four rotated version of it
def read_image_rotate(path: Path, grayscale: bool = False):
    """Read an image from path as RGB or grayscale"""
    # check path
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    
    # mode
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

    imgs = []

    # read
    imgs.append(cv2.imread(str(path), mode))

    # rotate
    for i in range(1, 8, 1):
        #imgs.append(cv2.rotate(imgs[i-1], cv2.ROTATE_90_CLOCKWISE))
        imgs.append(rotate_image(imgs[i-1], 45))

    if imgs[0] is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        for i in range(len(imgs)):
            imgs[i] = imgs[i][..., ::-1]
    return imgs

def load_image_rotate(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    # read
    imgs = read_image_rotate(path)

    # resize
    if resize is not None:
        for i in range(len(imgs)):
            imgs[i], _ = resize_image(imgs[i], resize, **kwargs)

    # to tensor
    for i in range(len(imgs)):
            imgs[i] = numpy_image_to_torch(imgs[i])
    return imgs

def save_features(features, path = "database/"):
    for f in tqdm(features):
        serialized = pickle.dumps(features[f])
        fil = open(path+"feats/"+f.split("/")[1]+".val", "wb")
        fil.write(serialized)
        fil.close()
    
    load_features(features)

def save_feature(f, name, path = "database/"):
    serialized = pickle.dumps(f)
    fil = open(path+"feats/"+name.split("/")[1]+".val", "wb")
    fil.write(serialized)
    fil.close()


def load_features(name, path = "feats"):
    fil = open(path+"/"+name.split("/")[len(name.split("/"))-1]+".val", "rb")
    data = fil.read()
    feature = pickle.loads(data)
    return feature

# loads database
def load_database(path = "database"):

    # feature extractor
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  
    #extractor = SIFT(max_num_keypoints=2048).eval().cuda()  

    # all images .jpg in repo
    img_files2 = glob.glob(os.path.join(path, "imgs",'*.jpg'))
    features = {}

    print("extracting features from database")
    for i in tqdm(img_files2):
        # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]

        if not os.path.exists(path+"/feats/"+i.split("/")[len(i.split("/"))-1]+".val"):
            image1 = load_image(i).cuda()
            # extract local features
            feats1 = extractor.extract(image1)

            feats2 = {}
            # convert to np to keep on cpu, preventing multiple computations of features for one image
            for k in feats1:
                feats2[k] = feats1[k].cpu().data.numpy()
            features[i] = feats2

            #print(i)
            save_feature(features[i], i)
        else:
            features[i] = load_features(i)

    return features          

def match_repository(features, path = "test_dataset"):
    
    TP = 0
    FN = 0

    metrics = {}

    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  
    matcher = LightGlue(features='superpoint').eval().cuda() 
    
    #extractor = DISK(max_num_keypoints=2048).eval().cuda()  
    #matcher = LightGlue(features='disk').eval().cuda()

    #extractor = SIFT(max_num_keypoints=2048).eval().cuda()
    #matcher = LightGlue(features='sift').eval().cuda()

    #matcher.compile(mode='reduce-overhead')
    
    # all .jpg
    img_files = sorted(glob.glob(os.path.join(path,'*.jpg')))
    print("matching")
    for j in range(0, len(img_files), 1):

        metrics[img_files[j]] = {}

        print("img "+str(j)+"/"+str(len(img_files)))
        img_file = img_files[j]

        print(img_files[j])

        if True:

            maxP = 0
            nameP = ""

            # get all rotated version
            imgs = load_image_rotate(img_file)

            #bestPoints = None
            #bestMatches = None
            #bestImg = None

        
            #print("bestMatchs/res"+img_file.split("/")[1]+",bestMatch.jpg")
            for img in imgs:

                # extract features
                feats0 = extractor.extract(img.cuda())

                if(maxP<210):
                    for i in tqdm(features):
                        
                        feats1 = {}
                        # get database features
                        for k in features[i]:
                            feats1[k] = torch.tensor(features[i][k]).cuda()

                        # match the features
                        matches01 = matcher({'image0': feats0, 'image1': feats1})
                        feats0_1, feats1_1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

                        # keypoints
                        kpts0, kpts1, matches = feats0_1["keypoints"], feats1_1["keypoints"], matches01["matches"]

                        # matched keypoints
                        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

                        # get max matched keypoints
                        if(len(m_kpts0)>maxP):
                            maxP=len(m_kpts0)
                            nameP = i
                            #bestPoints = (m_kpts0, m_kpts1, kpts0, kpts1)
                            #bestMatches = matches01
                            #bestImg = img

                        
                        if(i in metrics[img_files[j]]):
                            if(metrics[img_files[j]][i][0]<len(m_kpts0)):
                                metrics[img_files[j]][i] = [len(m_kpts0), len(m_kpts0)/(max(len(kpts0), len(kpts1))), len(m_kpts0)/(len(kpts0) + len(kpts1)), len(m_kpts0)/len(kpts0)]
                        else:
                            metrics[img_files[j]][i] = [len(m_kpts0), len(m_kpts0)/(max(len(kpts0), len(kpts1))), len(m_kpts0)/(len(kpts0) + len(kpts1)), len(m_kpts0)/len(kpts0)]
            
            # get name of the best matching image
            verif = nameP.split("/")[1].replace("R", "")
            verif = verif.split("_")

            rrr = ""
            for veri in range(len(verif)-1):
                rrr += verif[veri]+"_" 

            verif = rrr

            # UNKNOWN MEMORY LEAK ON LINUX

            # load both images
            #image1 = load_image(nameP).cuda() 
            #image0 = bestImg

            # plot of the images with points      
            #axes = viz2d.plot_images([image0, image1])
            #viz2d.plot_matches(bestPoints[0], bestPoints[1], color="lime", lw=0.2)
            # save plot
            #viz2d.save_plot("bestMatchs/res"+img_file.replace("noAnnotResold/", "")+",bestMatch.jpg")
            #matplotlib.pyplot.close()

            # plot of the images with points 
            #kpc0, kpc1 = viz2d.cm_prune(bestMatches["prune0"]), viz2d.cm_prune(bestMatches["prune1"])     
            #axes = viz2d.plot_images([image0, image1])
            #viz2d.plot_keypoints([bestPoints[2], bestPoints[3]], colors=[kpc0, kpc1], ps=6)
            # save plot
            #viz2d.save_plot("bestMatchsStats/res"+img_file.replace("noAnnotResold/", "")+",bestMatch.jpg")

            # free memory
            #matplotlib.pyplot.close()

            # if image matched to corresponding image 
            if(verif in img_file):
                TP+=1
            else:
                FN+=1

            accur = TP/(TP+FN)
            print(nameP)
            print(maxP)
            print("accuracy : "+str(accur)+" at step "+ str(j))

            f = open("resStatsInter.txt", "a")
    
            f.write(img_file + " | "+ nameP + "\n")

            f.close()

            #img_file = img_files[j]
            del verif
            #del image1
            #del image0
            #del bestPoints
            #del bestImg
            
            for i in range(len(imgs)):
                del imgs[0]
            del imgs

            gc.collect()

    accur = TP/(TP+FN)
    print("accuracy : "+str(accur))
    #print(metrics)
    #ecartM = []

    maxM = {}
    for k in metrics:
        maxM[k] = ["" ,[0, 0, 0, 0], [0, 0, 0, 0]]
        for k2 in metrics[k]:
            if(metrics[k][k2][0]>maxM[k][1][0]):
                for i in range(len(metrics[k][k2])):
                    maxM[k][1][i] = metrics[k][k2][i]
                maxM[k][0] = k2
        del metrics[k][maxM[k][0]]
        for k2 in metrics[k]:
            if(metrics[k][k2][0]>maxM[k][2][0]):
                for i in range(len(metrics[k][k2])):
                    maxM[k][2][i] = metrics[k][k2][i]

    print("############### valeurs obtenues ################")

    f = open("final_results.txt", "a")
    
    for k in maxM:
        f.write(k + " : "+ str(maxM[k])+ "\n")
    ecartMoy = [0, 0, 0, 0]
    ecartMax = [0, 0, 0, 0]
    ecartMin = [1000000, 10000000, 1000000, 1000000]

    f.close()

    vMoy = [0, 0, 0, 0]

    for k in maxM:
        for i in range(len(maxM[k][1])):
            vMoy[i]+=maxM[k][1][i]
            ecartMoy[i]+=maxM[k][1][i]-maxM[k][2][i]
            ecartMax[i]=max(maxM[k][1][i]-maxM[k][2][i], ecartMax[i])
            ecartMin[i]=min(maxM[k][1][i]-maxM[k][2][i], ecartMin[i])
    for i in range(len(ecartMoy)):
        ecartMoy[i]/=len(maxM)
        vMoy[i]/=len(maxM)

    ecartType = [0, 0, 0, 0]
    ecartTypeDist = [0, 0, 0, 0]
    for k in maxM:
        for i in range(len(maxM[k][1])):
            ecartType[i]+=(maxM[k][1][i]-vMoy[i])**2
            ecartTypeDist[i]+=(maxM[k][1][i]-maxM[k][2][i]-ecartMoy[i])**2

    for i in range(len(ecartMoy)):
        ecartType[i]/=len(maxM)
        ecartType[i]=math.sqrt(ecartType[i])
        ecartTypeDist[i]/=len(maxM)
        ecartTypeDist[i]=math.sqrt(ecartTypeDist[i])
    
    print("valeurs moyennes : " + str(vMoy))
    print("ecart Type : " + str(ecartType))
    print("ecarts moyens :" + str(ecartMoy))
    print("ecart ecart Type : " + str(ecartTypeDist))
    print("ecarts max :" + str(ecartMax))
    print("ecarts min :" + str(ecartMin))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # different modes
    parser.add_argument("command", choices=["match_repository", "save_features"], default="match_repository")

    # dir of data to be treated
    parser.add_argument("--datadir", type=Path, default='noAnnotResold')

    # dir of database 
    parser.add_argument("--databasedir", type=Path, default='noAnnotResorig')

    args = parser.parse_args()

    # loads database
    features = load_database(args.databasedir)
    
    # matches all images in dir to database
    exec(f"{args.command}(features, args.datadir)")
