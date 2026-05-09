import torch
from metrics import focal_eiou_loss   # 从 metrics.py 导入

def test_focal_eiou():
    # 重合框
    box1 = torch.tensor([[0, 0, 10, 10]])
    box2 = torch.tensor([[0, 0, 10, 10]])
    loss = focal_eiou_loss(box1, box2, xywh=False)
    print(f"重合框损失: {loss.item()}")  # 应接近 0

    # 不重叠框
    box1 = torch.tensor([[0, 0, 10, 10]])
    box2 = torch.tensor([[20, 20, 30, 30]])
    loss = focal_eiou_loss(box1, box2, xywh=False)
    print(f"不重叠框损失: {loss.item()}")  # 应接近 1

    print("✅ 测试通过")

if __name__ == "__main__":
    test_focal_eiou()