import torch
import numpy as np
import cv2

def grad_cam(model, image_tensor, target_layer="layer4"):
    model.eval()
    gradients = []

    def save_gradient(grad):
        gradients.append(grad)

    for name, module in model.named_modules():
        if name == target_layer:
            module.register_backward_hook(lambda module, grad_input, grad_output: save_gradient(grad_output[0]))

    output = model(image_tensor.unsqueeze(0))
    class_idx = torch.argmax(output)
    model.zero_grad()
    output[0, class_idx].backward()

    grad = gradients[0].cpu().detach().numpy()
    cam = np.mean(grad, axis=1).squeeze()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

    return cam
