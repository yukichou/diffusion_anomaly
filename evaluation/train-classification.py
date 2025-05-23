import torch
from torch import optim
from data_loader import MVTec_classification_train,MVTec_classification_test_one_third
from torch.utils.data import DataLoader
import os
from torchvision.models import resnet34
import torch.nn as nn


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def test(args, obj_name, model, anomaly_names):
    model.eval()

    dataset = MVTec_classification_test_one_third(args, obj_name, anomaly_names, num_images=args.num_images)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
    
    total_correct = 0
    total_count = 0

    for i_batch, sample_batched in enumerate(dataloader):
        image, label = sample_batched
        image = image.cuda()
        label = label.cuda()

        y_pred = model(image)
        prediction = torch.argmax(y_pred, 1)

        correct = (prediction == label).sum().float()
        total_correct += correct
        total_count += len(label)
    avg_acc = total_correct / total_count
    print(f"Accuracy (total): {avg_acc:.4f}")
    return avg_acc

def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)


    for obj_name in obj_names:
        print(obj_name)
        run_name = obj_name
        dataset = MVTec_classification_train(args,obj_name)
        class_num=dataset.class_num()
        anomaly_names =dataset.return_anomaly_names()
        model = resnet34(pretrained=True, progress=True)
        model.fc = nn.Linear(model.fc.in_features, class_num)
        model=model.cuda()

        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        criterion = nn.CrossEntropyLoss()
        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=16)
        max_acc=0
        for epoch in range(args.epochs):
            model.train()
            print("Epoch: "+str(epoch),end=' ')
            for i_batch, sample_batched in enumerate(dataloader):
                image,label=sample_batched
                image=image.cuda()
                label=label.cuda()
                y_pred=model(image)
                loss=criterion(y_pred,label)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

            scheduler.step()
            acc = test(args,obj_name, model, anomaly_names)
            if acc> max_acc:
                max_acc=acc
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--anomaly_id',  type=int, default=None)
    parser.add_argument("--sample_name", type=str, nargs='+', default='all', help="List of categories to process")
    parser.add_argument('--mvtec_path', type=str,required=True)
    parser.add_argument('--generated_data_path', type=str, required=True)
    parser.add_argument('--bs', action='store', type=int, default=8)
    parser.add_argument('--lr', action='store', type=float, default=0.0001)
    parser.add_argument('--epochs', action='store', type=int, default=30)
    parser.add_argument('--num_images', type=int, default=3)
    parser.add_argument(
        "--reverse",
        action="store_true", default=False,
    )
    parser.add_argument('--checkpoint_path', default='checkpoints/classification', type=str)

    args = parser.parse_args()

    obj_batch =  [
            'bottle','cable','capsule','carpet','grid','hazelnut','leather',
            'metal_nut','pill','screw','tile','transistor','wood','zipper'
        ]
    if args.reverse:
        obj_batch = reversed(obj_batch)
    if args.sample_name!='all':
        obj_list=args.sample_name
        picked_classes = obj_list
    else:
        picked_classes = obj_batch

    train_on_device(picked_classes, args)
