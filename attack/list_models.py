import torchvision

for model_type in [
        torchvision.models.alexnet,
        torchvision.models.convnext,
        torchvision.models.convnext_base,
        torchvision.models.convnext_large,
        torchvision.models.convnext_small,
        torchvision.models.convnext_tiny,
        torchvision.models.densenet,
        torchvision.models.densenet121,
        torchvision.models.densenet161,
        torchvision.models.densenet16
        torchvision.models.densenet201,
        torchvision.models.detection,
        torchvision.models.efficientnet,
        torchvision.models.efficientnet_b0,
        torchvision.models.efficientnet_b1,
        torchvision.models.efficientnet_b2,
        torchvision.models.efficientnet_b3,
        torchvision.models.efficientnet_b4,
        torchvision.models.efficientnet_b5,
        torchvision.models.efficientnet_b6,
        torchvision.models.efficientnet_b7,
        torchvision.models.feature_extraction,
        torchvision.models.googlenet,
        torchvision.models.inception,
        torchvision.models.inception_v3,
        torchvision.models.mnasnet,
        torchvision.models.mnasnet0_5,
        torchvision.models.mnasnet0_75,
        torchvision.models.mnasnet1_0,
        torchvision.models.mnasnet1_3,
        torchvision.models.mobilenet,
        torchvision.models.mobilenet_v2,
        torchvision.models.mobilenet_v3_large,
        torchvision.models.mobilenet_v3_small,
        torchvision.models.mobilenetv2,
        torchvision.models.mobilenetv3, torchvision.models.optical_flow, torchvision.models.quantization, torchvision.models.regnet, torchvision.models.regnet_x_16gf, torchvision.models.regnet_x_1_6gf, torchvision.models.regnet_x_32gf, torchvision.models.regnet_x_3_2gf, torchvision.models.regnet_x_400mf, torchvision.models.regnet_x_800mf, torchvision.models.regnet_x_8gf, torchvision.models.regnet_y_128gf, torchvision.models.regnet_y_16gf, torchvision.models.regnet_y_1_6gf, torchvision.models.regnet_y_32gf, torchvision.models.regnet_y_3_2gf, torchvision.models.regnet_y_400mf, torchvision.models.regnet_y_800mf, torchvision.models.regnet_y_8gf, torchvision.models.resnet, torchvision.models.resnet101, torchvision.models.resnet152, torchvision.models.resnet18, torchvision.models.resnet34, torchvision.models.resnet50, torchvision.models.resnext101_32x8d, torchvision.models.resnext50_32x4d, torchvision.models.segmentation, torchvision.models.shufflenet_v2_x0_5, torchvision.models.shufflenet_v2_x1_0, torchvision.models.shufflenet_v2_x1_5, torchvision.models.shufflenet_v2_x2_0, torchvision.models.shufflenetv2, torchvision.models.squeezenet, torchvision.models.squeezenet1_0, torchvision.models.squeezenet1_1, torchvision.models.vgg, torchvision.models.vgg11, torchvision.models.vgg11_bn, torchvision.models.vgg13, torchvision.models.vgg13_bn, torchvision.models.vgg16, torchvision.models.vgg16_bn, torchvision.models.vgg19, torchvision.models.vgg19_bn, torchvision.models.video, torchvision.models.vision_transformer, torchvision.models.vit_b_16, torchvision.models.vit_b_32, torchvision.models.vit_l_16, torchvision.models.vit_l_32, torchvision.models.wide_resnet101_2, torchvision.models.wide_resnet50_2

        
]:
    net = model_type(pretrained=False)
    print(net.__class__.__name__)
    print("Go on [ENTER]")
    input()
    print(net)
    print()
    print()

