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

    dataset = MVTec_classification_test_one_third(args,obj_name,anomaly_names,num_images=args.num_images)
    dataloader = DataLoader(dataset, batch_size=100,
                            shuffle=False, num_workers=0)

    total_correct = 0
    total_samples = 0

    for i_batch, sample_batched in enumerate(dataloader):
        image, label = sample_batched
        image = image.cuda()
        label = label.cuda()

        y_pred = model(image)
        prediction = torch.argmax(y_pred, 1)

        correct = (prediction == label).sum().float()
        total_correct += correct.item()
        total_samples += len(label)

    accuracy = total_correct / total_samples
    print(f"Accuracy for {obj_name}: {accuracy:.4f}")
    return accuracy
 
def test_on_device(obj_names, args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    all_accuracies = []

    object_classes  = {'bottle','cable','capsule','hazelnut',
                       'metal_nut','pill','screw','transistor','zipper'}
    texture_classes = {'carpet','grid','leather','tile','wood'}

    object_accs  = []   
    texture_accs = []   


    for obj_name in obj_names:
        dataset = MVTec_classification_train(args, obj_name)
        class_num = dataset.class_num()
        anomaly_names = dataset.return_anomaly_names()

        model = resnet34(pretrained=True, progress=True)
        model.fc = nn.Linear(model.fc.in_features, class_num)
        model = model.cuda()

        model.load_state_dict(torch.load(
            os.path.join(args.checkpoint_path, obj_name + '.pckl'))
        )

        accuracy = test(args, obj_name, model, anomaly_names)
        all_accuracies.append(accuracy)

        if obj_name in object_classes:
            object_accs.append(accuracy)
        elif obj_name in texture_classes:
            texture_accs.append(accuracy)

    avg_accuracy = sum(all_accuracies) / len(all_accuracies)
    print(f"Average Accuracy across all classes: {avg_accuracy:.4f}")

    if object_accs:
        obj_avg = sum(object_accs) / len(object_accs)
        print(f"Average Accuracy for OBJECT classes   : {obj_avg:.4f}")
    if texture_accs:
        tex_avg = sum(texture_accs) / len(texture_accs)
        print(f"Average Accuracy for TEXTURE classes  : {tex_avg:.4f}")

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

    test_on_device(picked_classes, args)


