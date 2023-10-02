<h2 align="center"># Automated-Leukemia-Detection-using-YOLO-based-Blood-Cell-Analysis<h2>



## Dataset

The [```White-Blood-Cell-Detection-Dataset```](https://github.com/medmabcf/White-Blood-Cell-Detection-Dataset) has been used for automatic identification and counting of white blood cell types. (It has been modified and augmented) . Download the dataset, unzip and put
the ```Training```, ```Testing```, and ```Validation```folders in the working directory.

## Requirements 
First, run ```pip install requirements.txt```. After classifying white blood cells, if you want to extract lymphocyte cells and check if the cell is normal or abnormal (acute leukemia blast), you should download the Segment Anything Model (SAM) weights by running the ```python Download_sam_weights``` command.

## Getting Started 

1. Build the cython extension in place 
```python setup.py build_ext --inplace```
2. Run detect.py 
```python detect.py```



## How to Run the Code  :runner:



## Paper

It will be published soon



## Blood Cell Detection Output

<p align="center">
  <img src="https://github.com/medmabcf/Automated-Leukemia-Detection-using-YOLO-based-Blood-Cell-Analysis/blob/main/cell_application/output/Im022_1/Im022_1.jpg" width="500">
</p>



