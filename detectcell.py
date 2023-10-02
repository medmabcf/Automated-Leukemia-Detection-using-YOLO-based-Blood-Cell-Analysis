from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView,QMessageBox
import subprocess,shutil,os, urllib.request
from PyQt5.QtGui import QPixmap
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import subprocess
import imaplib
from PIL import Image
import copy,cv2
import numpy as np
destination_folder = "cell_application/uploaded_images/"
if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
output_folder="cell_application/output"
if not os.path.exists(output_folder):
        os.makedirs(output_folder)
output_red="cell_application/outred"
if not os.path.exists(output_red):
        os.makedirs(output_red)
output_pl="cell_application/outpl"
if not os.path.exists(output_pl):
        os.makedirs(output_pl)
model_path = "./models/yolov7"  # specify your model path

if not os.path.exists(model_path):
    
    # Create the directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Clone the repository
    subprocess.check_call(['git', 'clone', 'https://github.com/WongKinYiu/yolov7', model_path])

    # Install the requirements
    subprocess.check_call(['pip', 'install', '-r', os.path.join(model_path, 'requirements.txt')])
class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent)
        self._isPanning = False
        self._panStartX = 0
        self._panStartY = 0

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self._isPanning = True
            self._panStartX = event.x()
            self._panStartY = event.y()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
        else:
            super(GraphicsView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._isPanning:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - (event.x() - self._panStartX))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - (event.y() - self._panStartY))
            self._panStartX = event.x()
            self._panStartY = event.y()
            event.accept()
        else:
            super(GraphicsView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self._isPanning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
        else:
            super(GraphicsView, self).mouseReleaseEvent(event)
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setMinimumSize(800, 600)
        Dialog.resize(1147, 840)

        # Create a vertical layout for the main window
        main_layout = QtWidgets.QVBoxLayout(Dialog)

        # Create a horizontal layout for the buttons
        button_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(button_layout)

        # Create the upload button and add it to the button layout
        self.uploadB = QtWidgets.QPushButton("upload")
        self.uploadB.setObjectName("uploadB")
        button_layout.addWidget(self.uploadB)

        # Create a stretchable spacer and add it to the button layout
        button_layout.addStretch()
        
        # Create an input field and add it to the button layout
        self.inputField = QtWidgets.QLineEdit()
        self.inputField.setObjectName("inputField")
        button_layout.addWidget(self.inputField)

       

        # Create labels for the results and add them to the results layout
        # Create a horizontal layout for the results
        self.results_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(self.results_layout)

        # Create labels for the results and add them to the results layout
        self.results = {'Basophil':0, 'Eosinophil':0, 'Erythroblast':0, 'Ig':0, 'Lymphocyte':0, 'Monocyte':0, 'Neutrophil':0,'White Cells':0, 'Red Cells':0, 'platelets':0}
        self.labels = {}  # Dictionary to store the QLabel objects

        i=0
        for d in self.results:
            result_label = QtWidgets.QLabel()
            result_label.setObjectName(f"result{i+1}")
            result_label.setText(f"{d} {self.results[d]}")  # Set some initial text
            self.results_layout.addWidget(result_label)
            self.labels[d] = result_label  # Store the QLabel object in self.labels
            i+=1
           
       # Create the extract_lymphocyte button and add it to the button layout
        self.extract_lymphocyte = QtWidgets.QPushButton("extract_lymphocyte")
        self.extract_lymphocyte.setObjectName("extract_lymphocyte")
        button_layout.addWidget(self.extract_lymphocyte)

        self.applypl = QtWidgets.QPushButton("pl")
        self.applypl.setObjectName("applypl")
        button_layout.addWidget(self.applypl)

        # Create the apply button and add it to the button layout
        self.applyred = QtWidgets.QPushButton("red")
        self.applyred.setObjectName("applyb")
        button_layout.addWidget(self.applyred)

        # Create the apply button and add it to the button layout
        self.applyb = QtWidgets.QPushButton("white")
        self.applyb.setObjectName("applyb")
        button_layout.addWidget(self.applyb)
        
      

       # Create the QGraphicsView and add it to the main layout
        self.graphicsView = GraphicsView()
        main_layout.addWidget(self.graphicsView, stretch=3)

        # Create a QGraphicsScene for the QGraphicsView
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        self.graphicsViewly = GraphicsView()
        main_layout.addWidget(self.graphicsViewly, stretch=1)

        # Create a QGraphicsScene for the second QGraphicsView
        self.scenely = QGraphicsScene()
        self.graphicsViewly.setScene(self.scenely)
        
    

        self.uploadB.clicked.connect(self.handle_upload_button_clicked)
        self.extract_lymphocyte.clicked.connect(self.handle_extract_lymphocyte_button_clicked)
        self.applyb.clicked.connect(self.handle_detect_button_clicked)
        self.applyred.clicked.connect(self.handle_red_button_clicked)
        self.applypl.clicked.connect(self.handle_pl_button_clicked)
    def change_image(self, image_path):
        # Create a QPixmap from the new image path
        pixmap = QtGui.QPixmap(image_path)

        # Clear the QGraphicsScene
        self.scene.clear()

        # Add the QPixmap to the QGraphicsScene
        self.scene.addPixmap(pixmap)

        # Fit the QGraphicsView to the scene's content
        self.graphicsView.fitInView(self.scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        
    def handle_upload_button_clicked(self):
        global take_lymphocyte, yolo_img_resize
        
        # Open a file dialog and get the selected image file path
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Select Image')
        global upload_name

        if file_path:
            upload_name  = os.path.basename(file_path)
            self.change_image(file_path)

            # Load the image into a QPixmap object
            pixmap = QPixmap(file_path)

            # Get the width and height
            width = pixmap.width()
            height = pixmap.height()
            ma = max(width, height)
            yolo_img_resize = max((ma // 640) *640 ,640)
            #yolo_img_resize = max((ma // 384) *384 ,384)
            #yolo_img_resize = (yolo_img_resize // 32) * 32 + (yolo_img_resize % 32 != 0) * 32
            self.inputField.setText(str(yolo_img_resize))

            # Define the destination folders
            destination_folder = "cell_application/uploaded_images/"

            source = file_path
            destination = os.path.join(destination_folder, upload_name)

            if os.path.normpath(source) != os.path.normpath(destination):
                # Copy the file
                shutil.copy(source, destination)
            else:
                # If source and destination are the same, delete and copy
                if os.path.exists(destination):
                    os.remove(destination)
                    shutil.copy(source, destination)
            
            # If the image is not a png or jpg make it jpg then save it
            if not (file_path.lower().endswith(('.png', '.jpg' ))):
               
                im = Image.open(file_path)
                rgb_im = im.convert('RGB')
                rgb_im.save(file_path[:-4] + 'jpg')
            lb = ['Basophil', 'Eosinophil', 'Erythroblast', 'Ig', 'Lymphocyte', 'Monocyte', 'Neutrophil','White Cells', 'Red Cells', 'platelets']
            for l in lb:
                self.labels[l].setText(l+': 0')

            

    
    def handle_detect_button_clicked(self):
        global yolo_img_resize
        yolo_img_resize = int(self.inputField.text())
        global take_lymphocyte
        image_path = "cell_application/uploaded_images/"+upload_name
        print(upload_name)
       
        model_path="models_weights/lastv7_aug_0.978_4.pt"
        command = f"python ./models/yolov7/detect.py --source {image_path} --weights {model_path} --img {yolo_img_resize} --conf 0.5 --project {output_folder} --save-txt"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        
        if os.path.exists("cell_application/output/exp"):
            if os.path.exists("cell_application/output/"+upload_name[0:upload_name.find('.')]):
                 shutil.rmtree("cell_application/output/"+upload_name[0:upload_name.find('.')])
            os.rename("cell_application/output/exp","cell_application/output/"+upload_name[0:upload_name.find('.')])

        image_out_path="cell_application/output/"+upload_name[0:upload_name.find('.')]+"/"+upload_name
        take_lymphocyte=True
       
        txt_file = glob.glob(os.path.join("cell_application/output/"+upload_name[0:upload_name.find('.')]+"/labels/", '*.txt'))[0]
        
  
       
        df = pd.read_csv(txt_file,delimiter=' ', header=None)
        df.columns = ['label', 'x_center', 'y_center', 'width', 'height']
      
        # Count the appearance of each label in the DataFrame
        label_counts = df['label'].value_counts()
       
        # Map the label numbers to their names
        label_names = {0: 'Basophil', 1: 'Eosinophil', 2: 'Erythroblast', 3: 'Ig', 4: 'Lymphocyte', 5: 'Monocyte', 6: 'Neutrophil'}

        # Update the results dictionary with the counts
        w=0
        for label_number, label_name in label_names.items():
            if label_number in label_counts:
                self.labels[label_name].setText(label_name+": "+str(label_counts[label_number]))
                w+=label_counts[label_number]
                
    
        self.labels['White Cells'].setText('White Cells: '+str(w))
       
      
        
   
        
        print(image_out_path)
        self.change_image(image_out_path)
    def handle_red_button_clicked(self):
        img_size = int(self.inputField.text())
        
        image_path = "cell_application/uploaded_images/"+upload_name
        
        model_path="models_weights/bestv7_red.pt"
        command = f"python ./models/yolov7/detect.py --source {image_path} --weights {model_path} --img {img_size} --conf 0.5 --project {output_red} --save-txt"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        if os.path.exists("cell_application/outred/exp"):
            if os.path.exists("cell_application/outred/"+upload_name[0:upload_name.find('.')]):
                 shutil.rmtree("cell_application/outred/"+upload_name[0:upload_name.find('.')])
            os.rename("cell_application/outred/exp","cell_application/outred/"+upload_name[0:upload_name.find('.')])
        image_out_path="cell_application/outred/"+upload_name[0:upload_name.find('.')]+"/"+upload_name
        txt_file = glob.glob(os.path.join("cell_application/outred/"+upload_name[0:upload_name.find('.')]+"/labels/", '*.txt'))[0]
        df = pd.read_csv(txt_file,delimiter=' ', header=None)
        df.columns = ['label', 'x_center', 'y_center', 'width', 'height']
       
        
    
        self.labels['Red Cells'].setText('Red Cells: '+str(len(df)))
       
      
     
        
        self.change_image(image_out_path)
    def handle_pl_button_clicked(self):
        img_size = int(self.inputField.text())
        
        image_path = "cell_application/uploaded_images/"+upload_name
        print(upload_name)
        model_path="models_weights/bestv7_pl_640_3.pt"
        command = f"python ./models/yolov7/detect.py --source {image_path} --weights {model_path} --img {img_size} --conf 0.5 --project {output_pl} --save-txt"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        if os.path.exists("cell_application/outpl/exp"):
            if os.path.exists("cell_application/outpl/"+upload_name[0:upload_name.find('.')]):
                 shutil.rmtree("cell_application/outpl/"+upload_name[0:upload_name.find('.')])
            os.rename("cell_application/outpl/exp","cell_application/outpl/"+upload_name[0:upload_name.find('.')])
        image_out_path="cell_application/outpl/"+upload_name[0:upload_name.find('.')]+"/"+upload_name
        txt_file = glob.glob(os.path.join("cell_application/outpl/"+upload_name[0:upload_name.find('.')]+"/labels/", '*.txt'))[0]
        df = pd.read_csv(txt_file,delimiter=' ', header=None)
        df.columns = ['label', 'x_center', 'y_center', 'width', 'height']
       
        
    
        self.labels['platelets'].setText('platelets: '+str(len(df)))
       
      
        
        
        self.change_image(image_out_path)
    

    def handle_extract_lymphocyte_button_clicked(self):
            global take_lymphocyte
                
                

            def get_most_recently_modified_folder(rootDir):
                # Create a list to store tuples of (folder_path, last_modified_time)
                folders = [(os.path.join(rootDir, dir), os.path.getmtime(os.path.join(rootDir, dir)))
                        for dir in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, dir))]

                # Sort the list by last_modified_time
                folders.sort(key=lambda x: x[1], reverse=True)

                # The first element in the list is the most recently modified folder
                most_recently_modified_folder = folders[0][0] if folders else None
               
                return most_recently_modified_folder

            if not take_lymphocyte:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("You should make a detection for white blood cell first press White button.")
                msg.setWindowTitle("Information")
                msg.exec_()
                
               
                
            
            else :
                main_folder_path = "cell_application/output"
                lastfolder_path=get_most_recently_modified_folder(main_folder_path)
            
                
                txt_file = glob.glob(os.path.join(lastfolder_path+"/labels/", '*.txt'))[0]
                
                df = pd.read_csv(txt_file,delimiter=' ', header=None)
                df.columns = ['label', 'x_center', 'y_center', 'width', 'height']
                df=df[df['label']==4]
                imagename=txt_file[txt_file.rfind('\\')+1:txt_file.find('.')]
                imagepath=os.path.join("cell_application/uploaded_images",imagename)
                
                image = Image.open(imagepath+".jpg")
                npimage=np.array(image)
                # Get the original size of the image
                ma=max(npimage.shape[0],npimage.shape[1])
                yolo_img_resize= (ma*2144) //2592
                yolo_img_resize=(yolo_img_resize//32)*32 +(yolo_img_resize%32!=0)*32
            
                height, width = yolo_img_resize,yolo_img_resize
                height_i, width_i=npimage.shape[:2]
            
                height_scale = height_i/height
                width_scale=width_i/width
                df['x_center'] *= width_i
                df['y_center'] *= height_i
                df['width'] *= width_i
                df['height'] *= height_i
                    
                # Calculate the top-left and bottom-right corners of the bounding boxes
                df1 = pd.DataFrame()
                df1['xmin'] = df['x_center'] - df['width'] / 2
                df1['ymin'] = df['y_center'] - df['height'] / 2
                df1['xmax'] = df['x_center'] + df['width'] / 2
                df1['ymax'] = df['y_center'] + df['height'] / 2
                
                new_box=[]
                crop_dir = lastfolder_path+"/crop_results"
                sam_dir = lastfolder_path+"/sam_results"

                # Check if the directory already exists
                if not os.path.exists(crop_dir):
                    # If not, create the directory
                    os.mkdir(crop_dir)
                if not os.path.exists(sam_dir):
                    # If not, create the directory
                    os.mkdir(sam_dir)
                boxes = [np.array(row) for row in df1.values]
                for i in range(len(boxes)):
                    box =copy.copy(boxes[i])
                    padding = 70
                    box[0] = max(0, box[0] - padding)
                    box[1] = max(0, box[1] - padding)
                    box[2] = min(image.width, box[2] + padding)
                    box[3] = min(image.height, box[3] + padding)

                    # Crop the image using the bounding box coordinates
                    cropped_image1 = np.array(image.crop(boxes[i]))
                    cropped_image = image.crop(box)
                    cropped_image.save(f'{lastfolder_path}/crop_results/{imagename}{i}.jpg')
                    # Save the cropped image
                    # Calculate the new bounding box coordinates
                    new_box.append(np.array([ padding,  padding,  padding+cropped_image1.shape[1],   padding+cropped_image1.shape[0]]))
                box = new_box
                import torch
                CHECKPOINT_PATH="models_weights/sam_vit_h_4b8939.pth"
                DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                MODEL_TYPE = "vit_h"
                import cv2
                    
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
                sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
                mask_predictor = SamPredictor(sam)
                mask_generator = SamAutomaticMaskGenerator(sam)
                from keras.models import load_model
                import cv2
                import supervision as sv
                # Load the model from the .h5 file
                model = load_model('models_weights/EfficientNetB3-leukemia-0.96.h5')
                i=-1
                image_list=[]
                cropimagespath=lastfolder_path+"/crop_results"
                for file in os.listdir(cropimagespath):
                
                    i+=1
                    image_bgr = cv2.imread(os.path.join(cropimagespath,file))
                    

                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                    mask_predictor.set_image(image_rgb)

                    masks, scores, logits = mask_predictor.predict(
                        box=new_box[i],
                        multimask_output=True
                    )
                        

                    img=np.zeros(image_bgr.shape)

                    img[:,:,:]=[0,0,0]
                    img[masks[2,:,:]]=image_bgr[masks[2,:,:]]
                    #imgtostore=img.copy()
                    #cv2.resize(imgtostore,(300,300))
                    cv2.imwrite(f'{sam_dir}/{imagename}{i}.jpg',img)
                
                    annotated_image = img.astype(np.uint8)
                
                
                    
                    """sam_result = mask_generator.generate(img)
                    mask_annotator = sv.MaskAnnotator()
                    
                    detections = sv.Detections.from_sam(sam_result=sam_result)

                    annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
                    print("ok2")"""
                    
                

                        
                    img_size_ly = (300, 300)
                    
                    def evaluate(model ,img):
                        

                        img = cv2.resize(img, img_size_ly)
                        

                        img = np.expand_dims(img, axis=0)
                        
                        # now predict the image
                        pred = model.predict(img,verbose=0)
                        print('the shape of prediction is ', pred.shape)
                        print(pred)
                
                        index = np.argmax(pred[0])

                        return index
                    image_list.append([(evaluate(model,annotated_image)),f'{sam_dir}/{imagename}{i}.jpg'])
            
                print(image_list)
                x_offset = 0 
                label_dict = {0: "all", 1: "hem"}
                for image_info in image_list:
                    label_number, image_path = image_info
                    label = label_dict[label_number]
                    pixmap = QtGui.QPixmap(image_path)

                    # Scale the pixmap without preserving aspect ratio
                    scaled_pixmap = pixmap.scaled(300, 300)

                    item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)
                    item.setPos(x_offset, 0)
                    self.scenely.addItem(item)

                    # Create a QGraphicsTextItem for the label
                    text_item = QtWidgets.QGraphicsTextItem(label)

                    # Set the color of the text to white
                    text_item.setDefaultTextColor(QtGui.QColor('white'))

                    # Calculate the center position of the image for x-coordinate
                    center_x = x_offset + scaled_pixmap.width() / 2 - text_item.boundingRect().width() / 2

                    # Set the position of the text item on the image and centered horizontally
                    text_item.setPos(center_x, scaled_pixmap.height() / 2)
                    
                    self.scenely.addItem(text_item)

                    # Update x offset for next image
                    x_offset += scaled_pixmap.width()

                

                




                 
                 
                 
    
            
      


        



if __name__ == "__main__":
    import sys
    upload_name = ""
    yolo_img_resize=1216
    lastfolder_path=""
    take_lymphocyte=False
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    
    # Add a maximize button to the window
    Dialog.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)

    Dialog.show()
    sys.exit(app.exec_())
