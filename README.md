# ICPR2024

## requirements 

Python >3.7

Machine with a GPU

Bash

```
pip install -r requirements.txt
git clone https://github.com/facebookresearch/detectron2.git
pip install -e detectron2
git clone https://github.com/cvg/LightGlue.git
pip install -e LightGlue
```

## Download 

Best segmentation model : https://drive.google.com/file/d/12wZzavrOAZWrStv2qsJO_6ROZP4UWYwN/view?usp=sharing

Matching model : Models are downloaded when first launching the associated program

Matching Test dataset : https://drive.google.com/file/d/1hCKqro5xOqCVYpxVl-TYW7PSZt4j7luW/view?usp=drive_link

Matching database : https://drive.google.com/file/d/1MrXsypMNI9n8DwIKpcDxUi_jn-jfNcYt/view?usp=drive_link

## Instructions

### Expected File tree
```bash
.
├── logs
│   ├── configs
│   ├── yamls
│   └── modelRep
├── database
│   ├── feats
│   └── imgs
└── test_dataset
```
### Segmentation

```
python OurSegTrain.py inference --batch_size 2 --logname SeriousMask --model PointRend --datadir other_dataset --logdir logs --pth_name  validatedmodel.pth
```

With :
- batch_size : size of batch
- logname : model's repository name
- logdir : path to model's repository
- pth_name : name of the model's file
- datadir : path to the images to segment

### Matching 

```
python matching.py --databasedir databasedir --datadir datadir
```

With :
- databasedir : path to the database
- datadir : path to the images to find
