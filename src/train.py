import argparse
from pathlib import Path
import torch
from data_module import DataModule
from model import CVModel
from trainer import Trainer
from utils.evaluation import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PyTorch model on an image dataset with automatic train/val/test split"
    )
    parser.add_argument("--data-dir",    type=str,   required=True,
                        help="root folder containing class subdirectories")
    parser.add_argument("--model-name", type=str, default = "yolov8n-cls", 
                        help = "training model")
    parser.add_argument("--epochs",      type=int,   default=30,
                        help="number of training epochs")
    parser.add_argument("--batch-size",  type=int,   default=16,
                        help="batch size for train/val/test")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--img-size",    type=int,   nargs=2, default=(224, 224),
                        help="resize images to this width height")
    parser.add_argument("--train-split",   type=float, default=0.7,
                        help="fraction of data for train")
    parser.add_argument("--val-split",   type=float, default=0.15,
                        help="fraction of data for validation")
    parser.add_argument("--test-split",  type=float, default=0.15,
                        help="fraction of data for testing")
    parser.add_argument("--num-workers", type=int,   default=4,
                        help="number of DataLoader workers")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--device",      type=str,   choices=["cpu","cuda"], default="cuda",
                        help="device override (cpu or cuda)")
    parser.add_argument("--patience", type=int, default=5,
                        help="early stop")
    parser.add_argument("--testing",type=int,default=0,
        help="if 1, skip training and just run evaluation on the test set",
    )
    parser.add_argument("--trained-model",type=str,default=None,
        help="path to a .pth checkpoint to load when testing",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device : {device}")

    dm = DataModule(
                    data_dir = args.data_dir,
                    batch_size = args.batch_size,
                    img_size = args.img_size,
                    train_split = args.train_split,
                    val_split=  args.val_split,
                    test_split = args.test_split,
                    num_workers = args.num_workers,
                    seed = args.seed,
    )
    dm.setup()
    train_loader, val_loader, test_loader, dataset_loader = dm.get_loaders()
    print(f"Detected classes: {dm.classes}")


    if args.testing == 1:
        model = CVModel(
            model_name=args.model_name,
            num_classes = len(dm.classes),
            pretrained=False
        )
        model.class_names = dm.classes


        checkpoint = torch.load(args.trained_model, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        evaluate_model(model, dataset_loader, device, dm.classes)
        return 




    model = CVModel(model_name = args.model_name,
                    num_classes = len(dm.classes),
                    pretrained = True)


    model.class_names = dm.classes 

    trainer = Trainer(model, 
                      model_name = args.model_name, 
                      train_loader = train_loader, 
                      val_loader = val_loader, 
                      test_loader = test_loader, 
                      device = device, 
                      lr = args.lr,
                      patience = args.patience)
    trainer.fit(args.epochs)


if __name__ == "__main__":
    try:
        main()
    finally:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()