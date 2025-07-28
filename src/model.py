import inspect
import torch.nn as nn
import torchvision.models as models
from torchvision.models import get_model_weights
from ultralytics import YOLO
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
)


class CVModel(nn.Module):


    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes

        if self.model_name.startswith("yolo"):
            self.model = YOLO(model_name) 
            core = self.model.model

            if self.model_name.endswith("cls"):
                if hasattr(core, "reset_classifier"):
                    core.reset_classifier(num_classes)
                else: 
                    last = getattr(core, "classifier", None)
                    if isinstance(last, nn.Linear):
                        core.classifier = nn.Linear(last.in_features, num_classes)

            for p in core.parameters():
                p.requires_grad = True
            return

        if self.model_name == "fasterrcnn_resnet50_fpn":
            self.model = fasterrcnn_resnet50_fpn(
                weights="DEFAULT" if pretrained else None,
                weights_backbone="DEFAULT" if pretrained else None,
                **kwargs,
            )
            in_feat = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_feat, num_classes
            )
            return


        if hasattr(models, self.model_name):
            fn = getattr(models, self.model_name)
            sig = inspect.signature(fn)
            if "weights" in sig.parameters:
                weights = (
                    get_model_weights(self.model_name).DEFAULT if pretrained else None
                )
                self.model = fn(weights=weights, **kwargs)
            else: 
                self.model = fn(pretrained=pretrained, **kwargs)


            replaced = False
            for attr in ("classifier", "fc", "head", "heads", "last_linear"):
                mod = getattr(self.model, attr, None)

                if isinstance(mod, nn.Linear):
                    setattr(
                        self.model,
                        attr,
                        nn.Linear(mod.in_features, num_classes, bias=mod.bias is not None),
                    )
                    replaced = True
                    break

                if isinstance(mod, nn.Sequential) and mod and isinstance(mod[-1], nn.Linear):
                    mod[-1] = nn.Linear(
                        mod[-1].in_features, num_classes, bias=mod[-1].bias is not None
                    )
                    replaced = True
                    break
            if not replaced:
                raise ValueError("Could not locate a Linear classification head to replace.")
            return


        raise ValueError(f"Unsupported model: {model_name}")


    def forward(self, images, targets=None):

        if self.model_name.startswith("yolo"):
            if self.model_name.endswith("cls"):
                logits = self.model.model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                return logits
            return self.model(images, verbose=False)

        if "faster" in self.model_name:
            return self.model(images, targets)  # type: ignore[arg-type]

        return self.model(images)

    def eval(self):
        super().eval()
        if self.model_name.startswith("yolo"):
            self.model.model.eval()  # set internal core
        return self
