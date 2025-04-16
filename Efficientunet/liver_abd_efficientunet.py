from collections import OrderedDict
from layers import *
from efficientnet import EfficientNet


__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True, multi_class = False):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input
        self.multi_class = multi_class

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x
        #print("input: ", x.shape)

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()

        #print("encoder: ", x.shape)
        x = self.up_conv1(x)
        #print("After up_conv1: \t\t\t\t\t", x.shape)
        
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        #print("After concatenation with block 1: \t\t\t", x.shape)
        
        x = self.double_conv1(x)
        #print("After double_conv1: \t\t\t\t\t", x.shape)

        x = self.up_conv2(x)
        #print("After up_conv2: \t\t\t\t\t", x.shape)

        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        #print("After concatenation with block 2: \t\t\t", x.shape)

        x = self.double_conv2(x)
        #print("After double_conv2: \t\t\t\t\t", x.shape)

        x = self.up_conv3(x)
        #print("After up_conv3: \t\t\t\t\t", x.shape)

        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        #print("After concatenation with block 3: \t\t\t", x.shape)

        x = self.double_conv3(x)
        #print("After double_conv3: \t\t\t\t\t", x.shape)

        x = self.up_conv4(x)
        #print("After up_conv4: \t\t\t\t\t", x.shape)

        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        #print("After concatenation with block 4: \t\t\t", x.shape)

        x = self.double_conv4(x)
        #print("After double_conv4: \t\t\t\t\t", x.shape)

        if self.concat_input:
            x = self.up_conv_input(x)
            #print("After up_conv_input: \t\t\t\t\t", x.shape)
            
            x = torch.cat([x, input_], dim=1)
            #print("After concatenating input: \t\t\t\t", x.shape)
            
            x = self.double_conv_input(x)
            #print("After double_conv_input: \t\t\t\t\t", x.shape)

        x = self.final_conv(x)
        #print("After final_conv: \t\t\t\t\t", x.shape)

        # why do we do sigmoid before interpolating?
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        if self.multi_class:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        
        #print("After interpolation: \t\t\t\t\t", x.shape)

        return x


class EfficientUnetParallel(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        # Decoder for Region 1
        self.up_conv1_region1 = up_conv(self.n_channels, 512)
        self.double_conv1_region1 = double_conv(self.size[0], 512)
        self.up_conv2_region1 = up_conv(512, 256)
        self.double_conv2_region1 = double_conv(self.size[1], 256)
        self.up_conv3_region1 = up_conv(256, 128)
        self.double_conv3_region1 = double_conv(self.size[2], 128)
        self.up_conv4_region1 = up_conv(128, 64)
        self.double_conv4_region1 = double_conv(self.size[3], 64)
        self.final_conv_region1 = nn.Conv2d(64, out_channels, kernel_size=1)

        # Decoder for Region 2
        self.up_conv1_region2 = up_conv(self.n_channels, 512)
        self.double_conv1_region2 = double_conv(self.size[0], 512)
        self.up_conv2_region2 = up_conv(512, 256)
        self.double_conv2_region2 = double_conv(self.size[1], 256)
        self.up_conv3_region2 = up_conv(256, 128)
        self.double_conv3_region2 = double_conv(self.size[2], 128)
        self.up_conv4_region2 = up_conv(128, 64)
        self.double_conv4_region2 = double_conv(self.size[3], 64)
        self.final_conv_region2 = nn.Conv2d(64, out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x
        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, encoder_output = blocks.popitem()
        
        encoder_output_1 = blocks.popitem()[1]
        encoder_output_2 = blocks.popitem()[1]
        encoder_output_3 = blocks.popitem()[1]
        encoder_output_4 = blocks.popitem()[1]

        # Decoder 1
        x1 = self.up_conv1_region1(encoder_output)
        x1 = torch.cat([x1, encoder_output_1], dim=1)
        x1 = self.double_conv1_region1(x1)
        x1 = self.up_conv2_region1(x1)
        x1 = torch.cat([x1, encoder_output_2], dim=1)
        x1 = self.double_conv2_region1(x1)
        x1 = self.up_conv3_region1(x1)
        x1 = torch.cat([x1, encoder_output_3], dim=1)
        x1 = self.double_conv3_region1(x1)
        x1 = self.up_conv4_region1(x1)
        x1 = torch.cat([x1, encoder_output_4], dim=1)
        x1 = self.double_conv4_region1(x1)
        output_region1 = self.final_conv_region1(x1)

        # Decoder 2
        x2 = self.up_conv1_region2(encoder_output)
        x2 = torch.cat([x2, encoder_output_1], dim=1)
        x2 = self.double_conv1_region2(x2)
        x2 = self.up_conv2_region2(x2)
        x2 = torch.cat([x2, encoder_output_2], dim=1)
        x2 = self.double_conv2_region2(x2)
        x2 = self.up_conv3_region2(x2)
        x2 = torch.cat([x2, encoder_output_3], dim=1)
        x2 = self.double_conv3_region2(x2)
        x2 = self.up_conv4_region2(x2)
        x2 = torch.cat([x2, encoder_output_4], dim=1)
        x2 = self.double_conv4_region2(x2)
        output_region2 = self.final_conv_region2(x2)
        
        output_region1 = torch.sigmoid(output_region1)
        output_region2 = torch.sigmoid(output_region2)
        
        output_region1 = F.interpolate(output_region1, size=(1024, 1024), mode='bilinear', align_corners=False)
        output_region2 = F.interpolate(output_region2, size=(1024, 1024), mode='bilinear', align_corners=False)

        return output_region1, output_region2

def get_efficientunet_b0_parallel(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnetParallel(encoder, out_channels=out_channels, concat_input=concat_input)
    return model

class EfficientUnetSharedDecoder(nn.Module):
    def __init__(self, encoder, num_labels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        # Shared Decoder
        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        # Separate final convolution layers for each label
        self.final_conv1 = nn.Conv2d(64, 1, kernel_size=1)
        self.final_conv2 = nn.Conv2d(64, 1, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x
        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, encoder_output = blocks.popitem()

        # Shared decoder
        x = self.up_conv1(encoder_output)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        # Produce separate outputs for each label
        output1 = torch.sigmoid(self.final_conv1(x))
        output2 = torch.sigmoid(self.final_conv2(x))
        
        output1 = F.interpolate(output1, size=(1024, 1024), mode='bilinear', align_corners=False)
        output2 = F.interpolate(output2, size=(1024, 1024), mode='bilinear', align_corners=False)


        return output1, output2

def get_efficientunet_b0_shared_decoder(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnetSharedDecoder(encoder, num_labels=out_channels, concat_input=concat_input)
    return model

def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True, multi_class = False):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input, multi_class=multi_class)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model
