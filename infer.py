import argparse
import torch
import os 
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import segmentation_models_pytorch as smp
def load_model(checkpoint_path, device):
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()  
    return model

def preprocess_image(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    test_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    image_tensor = test_transform(image).unsqueeze(0) 
    return image_tensor, image

def postprocess_output(output, original_image):
    to_PIL = transforms.ToPILImage()
    image_shape = original_image.shape
    resize = transforms.Resize(size= (image_shape[0],image_shape[1]))
    output =  F.softmax(output,dim=1)
    output = resize(output)
    predicted_mask = F.one_hot(torch.argmax(output.squeeze(), 0).cpu()).permute(2,0,1).float()
    predicted_mask = to_PIL(predicted_mask)
    return predicted_mask

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    )
    model.to(device)
    # Load the model
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'best_check_point (1).pth')  # Update with your checkpoint path
    state_dict = torch.load(checkpoint_path,map_location=torch.device('cpu'))    
    model.load_state_dict(state_dict)

    image_tensor, original_image = preprocess_image(args.image_path)
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Postprocess the output
    segmented_image = postprocess_output(output, original_image)
    
    # Save the result
    output_path = os.path.splitext(args.image_path)[0] + "_segmented.png"
    segmented_image.save(output_path)
    print(f"Segmented image saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    main(args)