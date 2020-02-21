The goal of this work is to provide examples of how to:

1) Use the **PyTorch Dataset class** in PyTorch for CIFAR10, instead of simply importing it in one line.
   Hopefully this will help someone bridge the gap towards implementing more complex datasets.
   
2) Create a **custom loss** class which can be extended for more advanced loss functions which are not
   included in the standard PyTorch arsenal. 
   
   
Additionally, we have included a **hard-coded VGG16** for anyone that is just beginning and needs that 
level of readability. 

At the bottom the **Data.py**, you will find code which allows you to visualize the images of CIFAR10.
This is the only place we have used **OpenCV**, so if you do not wish to visualize any images, you do
not need to "import cv2".

With the parameters set as is, using a VGG16, the model acheieves **92.96%** test
accuracy. 

The project dependencies can be found in requirements.txt .

