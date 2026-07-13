import torch
def wasserstein_distance(pred, target, C=12.8):
    px = pred[:, 0]
    py = pred[:, 1]
    pw = pred[:, 2]
    ph = pred[:, 3]

    gx = target[:, 0]
    gy = target[:, 1]
    gw = target[:, 2]
    gh = target[:, 3]

    center_distance = (px-gx)**2+(py-gy)**2
    shape_distance = ((pw-gw)**2 + (ph-gh)**2) / 4
    
    distance = center_distance + shape_distance
    nwd = torch.exp(-torch.sqrt(distance+1e-7)/C)
    loss = 1-nwd
    return loss

if __name__ == "__main__":
    # test 1: hai box giống nhau ~ 0
    # pred = torch.tensor([
    #     [100, 100, 20, 20]
    # ])
    # target = torch.tensor([
    #     [100, 100, 20, 20]
    # ])

    # test 2: hai box khác nhau > 0
    pred = torch.tensor([
        [110, 105, 22, 18]
    ], dtype=torch.float32)

    target = torch.tensor([
        [100, 100, 20, 20]
    ], dtype=torch.float32)
    print(wasserstein_distance(pred, target))