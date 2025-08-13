from jetcam.usb_camera import USBCamera
from utils import preprocess
import torch
import torch.nn.functional as F
import torchvision
import cv2

CATEGORIES = ['thumbs_up', 'thumbs_down']


def live( model, camera ):
    global CATEGORIES
    while 1:
        image = camera.value
        preprocessed = preprocess(image)
        output = model(preprocessed)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        print( 'prediction = ' , CATEGORIES[category_index])
        for i, score in enumerate(list(output)):
            #score_widgets[i].value = score
        
            if i == 0:
                if score >= 0.75:
                    print("Thumbs Up Score = ", score )
                
            else:
                if score >= 0.75:
                    print("Thumbs Down Score = ", score )

         
if __name__ == "__main__":

    camera = USBCamera(width=224, height=224, capture_device=0)
    camera.running = True
    print("camera created")
    cv2.imwrite('./test.jpg', camera.value)
    device = torch.device('cuda')

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, len(CATEGORIES))
    model = model.to(device)
    model.load_state_dict(torch.load('/home/ioscape/nvdli-data/classification/my_model.pth'))
    model.eval()
    
    live(model, camera)
    
    

			 

    
    