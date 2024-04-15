import torch
from PIL import Image
from torchvision import transforms
from ATUIQP import ATUIQP  # ATUIQP


def predict():
    test_img_name = ["5706.jpg", "1418.jpg", "3412.jpg", "4688.jpg"]
    test_img_mos = [1.17, 2.17, 3.33, 4.50]
    device = torch.device((f"cuda:0" if torch.cuda.is_available() else "cpu"))
    model = ATUIQP()
    weight = torch.load("./model/model_epoch_10.pth", map_location="cpu")
    model.load_state_dict(weight)
    del weight
    model.to(device)

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    for i in range(len(test_img_name)):
        img = trans(Image.open(f"./input/{test_img_name[i]}")).unsqueeze(0).to(device)
        score = model(img)
        print(f"Test img: {test_img_name[i]}, predict score: {round(score.item(), 2)}, mos: {test_img_mos[i]}")


def predict2(test_img_path=None):
    device = torch.device((f"cuda:0" if torch.cuda.is_available() else "cpu"))
    model = ATUIQP()
    weight = torch.load("./model/model_epoch_10.pth", map_location="cpu")
    model.load_state_dict(weight)
    del weight
    model.to(device)

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img = trans(Image.open(test_img_path)).unsqueeze(0).to(device)
    score = model(img)
    print(f"Predict score: {round(score.item(), 2)}")


if __name__ == '__main__':
    pass
    # GPU Memory: 1600MB
    # 1
    predict()
    # Test img: 5706.jpg, predict score: 1.41, mos: 1.17
    # Test img: 1418.jpg, predict score: 2.33, mos: 2.17
    # Test img: 3412.jpg, predict score: 3.13, mos: 3.33
    # Test img: 4688.jpg, predict score: 3.81, mos: 4.5
    # 2
    predict2(test_img_path="./input/3412.jpg")
    # Predict score: 3.13
