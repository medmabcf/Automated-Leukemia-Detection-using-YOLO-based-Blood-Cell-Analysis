<h2 align="center"> Automated-Leukemia-Detection-using-YOLO-based-Blood-Cell-Analysis<h2>

## Dataset

The [```White-Blood-Cell-Detection-Dataset```](https://github.com/medmabcf/White-Blood-Cell-Detection-Dataset) has been used for automatic identification and counting of white blood cell types. (It has been modified and augmented) . Download the dataset, unzip and put
the ```Training```, ```Testing```, and ```Validation```folders in the working directory.


## Requirements

Follow these steps to set up your environment:

1. **Clone the YOLOv7 model:**
    ```shell
    Remove-Item -Path models\yolov7 -Recurse -Force
    git clone https://github.com/WongKinYiu/yolov7.git models\yolov7
    ```

2. **Clone the Segment Anything model:**
    ```shell
    Remove-Item -Path models\segment-anything -Recurse -Force
    git clone https://github.com/facebookresearch/segment-anything.git  models\segment-anything
    ```

3. **Install the required Python packages:**
    ```shell
    pip install -r requirements.txt
    ```

After you've classified the white blood cells, if you want to extract lymphocyte cells and check if the cell is normal or abnormal (acute leukemia blast), you should download the Segment Anything Model (SAM) weights by running the following command:

```shell
python Download_sam_weights
```
## Getting Started 


1. Run detectcell.py 
```python detectcell.py```



## How to Run the Code  :runner:





## Blood Cell Detection Output


  <p align="center">
  <img src="https://github.com/medmabcf/Automated-Leukemia-Detection-using-YOLO-based-Blood-Cell-Analysis/blob/main/cell_application/output/Im022_1/Im022_1.jpg" width="500">
    
  <img src="https://github.com/medmabcf/Automated-Leukemia-Detection-using-YOLO-based-Blood-Cell-Analysis/blob/main/cell_application/output/Im043_0/Im043_0.jpg" width="500">
  <img src="https://github.com/medmabcf/Automated-Leukemia-Detection-using-YOLO-based-Blood-Cell-Analysis/blob/main/cell_application/output/Im080_0/Im080_0.jpg" width="500">
   <img src="https://github.com/medmabcf/Automated-Leukemia-Detection-using-YOLO-based-Blood-Cell-Analysis/blob/main/cell_application/output/Im079_0/Im079_0.jpg" width="500">
   
   <img src="https://github.com/medmabcf/Automated-Leukemia-Detection-using-YOLO-based-Blood-Cell-Analysis/blob/main/cell_application/output/Im076_0/Im076_0.jpg" width="500">

   <img src="https://github.com/medmabcf/Automated-Leukemia-Detection-using-YOLO-based-Blood-Cell-Analysis/blob/main/cell_application/outred/Im076_0/Im076_0.jpg" width="500">
   
</p>




