import torch

def evaluate(net, ds, cls_indexes, device):
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    net = net.to(device)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dl:
            images, labels = data
            outputs, features = net(images.to(device))
            outputs = outputs[:, cls_indexes]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

    return correct / total