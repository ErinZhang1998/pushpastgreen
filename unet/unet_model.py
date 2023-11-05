""" Full assembly of the parts to form the complete network """
from torch.nn import init
from .unet_parts import *
from torchvision.models import resnet18, ResNet18_Weights

def load_model(model, model_path, weights_prefix_to_load):
    pretrained_dict = torch.load(model_path)
    
    if len(weights_prefix_to_load) > 0:
        filtered_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            load = False 
            for name_prefix in weights_prefix_to_load:
                if k.startswith(name_prefix):
                    load=True 
                    break
            if load:
                print(":: Loading weights from layer: ", k)
                filtered_pretrained_dict[k] = v
        model_dict = model.state_dict()
        model_dict.update(filtered_pretrained_dict) 
        model.load_state_dict(model_dict)
    else:
        print(":: Load all weights")
        model.load_state_dict(pretrained_dict)
    
    return model

def get_action_layers(action_upconv_channels, num_layers):
    action_upconv_layers = []
    for l in range(num_layers):
        in_channels = action_upconv_channels[0] if l==0 else action_upconv_channels[2*l]
        this_layer = UpNoSkipConnection(in_channels, action_upconv_channels[2*l+1], action_upconv_channels[2*l+2], 2, bilinear=False)
        action_upconv_layers.append(this_layer)
    return action_upconv_layers

def weights_init(m, gain=1.0):
    torch.manual_seed(1219)
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if m.weight.data.requires_grad == True:
            init.xavier_normal_(m.weight.data, gain=gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)


class UNet(nn.Module):
    def __init__(self, hp):
        '''
        1. hp['downconv_channels']: 64 64 128 256 512 --> 4 DownMaxpoolSingleConv, if input size is 150, then 75 37 18 9
        2. hp['upconv_channels']: 4 down sampling, so the size of this list is 8, e.g. 512 512 256 256 128 128 64 64
            The ith Up layer will use hp['upconv_channels'][2*i] and hp['upconv_channels'][2*i+1]
            Up(in_channels, out_channels, conv_in_channels, conv_out_channels)
            in_channels = hp['upconv_channels'][2*i-1], but if i=0, then hp['downconv_channels'][-1] + action encoding dimension
            out_channels = hp['upconv_channels'][2*i]
            conv_in_channels = hp['upconv_channels'][2*i] + hp['downconv_channels'][L_layers-1-l], because of skip connections
            conv_out_channels = hp['upconv_channels'][2*i+1]
        '''
        super(UNet, self).__init__()

        n_channels = 3
        if hp['use_height_feature']:
            n_channels += 1
        if hp['no_rgb_feature']:
            assert hp['use_height_feature']
            n_channels = 1
        if hp['action_as_image']:
            if hp['no_image']:
                n_channels = 1
            else:
                n_channels += 1

        # if hp['use_coordinate_feature']:
        #     n_channels += 2
        self.hp = hp
        self.n_channels = n_channels
        self.n_classes = len(hp['classes'])+1
        if hp['use_regression_loss']:
            self.n_classes += 1
        if hp['use_postaction_mask_loss']:
            self.n_classes += 2
        if hp['predict_precondition']:
            self.n_classes = 2
        self.bilinear = False
        self.no_image = hp['no_image']
        self.action_as_image = hp['action_as_image']
        self.no_action = hp['no_action']

        # self.resnet18_inc = resnet18(weights=ResNet18_Weights.DEFAULT)
        if self.action_as_image or (not self.no_image):
            upconv_channels = hp['upconv_channels']
            if hp['use_resnet']:
                resnet_inc = resnet18(weights=ResNet18_Weights.DEFAULT)
                resnet_parts = list(resnet_inc.children())
                downconv_channels = [64,64,128,256,512]
                # nn.Conv2d(self.n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                if hp['resnet_maxpool_after_enc']:
                    enc_parts = [nn.Conv2d(self.n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)] + resnet_parts[1:4]
                    self.inc = nn.Sequential(*enc_parts)
                else:
                    self.inc = SingleConv(self.n_channels, downconv_channels[0])
                other_parts = resnet_parts[4:-2]
                self.downconv = nn.ModuleList(other_parts)
                if hp['freeze_resnet']:
                    for name,param in self.downconv.named_parameters():
                        param.requires_grad = False
            else:
                assert (len(hp['downconv_channels'])-1) * 2 == len(hp['upconv_channels'])
                downconv_channels = hp['downconv_channels']
                self.inc = SingleConv(self.n_channels, downconv_channels[0])
                downconv_layers = []
                for l in range(1, len(downconv_channels)):
                    if hp['unet_downsample_layer_type'] == "DownMaxpoolSingleConv":
                        this_layer = DownMaxpoolSingleConv(downconv_channels[l-1], downconv_channels[l])
                    else:
                        this_layer = DownRelu(downconv_channels[l-1], downconv_channels[l], kernel_size=3, stride=2)
                    downconv_layers.append(this_layer)
                self.downconv = nn.ModuleList(downconv_layers)
            L_layers = len(downconv_channels)-1
            self.L_layers = L_layers
        
        action_upconv_channels = hp['action_upconv_channels']
        if hp['predict_precondition']:
            self.action_inc = nn.Linear(hp['action_input_dim'], action_upconv_channels[0])
            self.outprecond = nn.Linear(hp['precondition_input_dim']+action_upconv_channels[0], self.n_classes)
            return
        
        
        # if hp['action_upconv_sizes'] is not None:
        #     self.action_upconv_sizes = hp['action_upconv_sizes']
        # else:
        #     self.action_upconv_sizes =[2 ** j for j in range(1, len(hp['action_upconv_channels']) // 2 + 1)]
        self.action_upconv_sizes = hp['action_upconv_sizes'] if hp['action_upconv_sizes'] is not None else []
        factor = 2 if self.bilinear else 1
        if (not self.action_as_image) and (not self.no_action):
            self.action_inc = nn.Linear(hp['action_input_dim'], action_upconv_channels[0])
            action_upconv_layers = get_action_layers(action_upconv_channels, len(self.action_upconv_sizes))
            if len(action_upconv_layers) == 0:
                self.has_action_upconv = False 
            else:
                self.has_action_upconv = True
                self.action_upconv = nn.ModuleList(action_upconv_layers)
            action_upconv_channels = action_upconv_channels[:(len(self.action_upconv_sizes)*2+1)]
            if not self.no_image:
                up1_input_ch = downconv_channels[-1]+action_upconv_channels[-1]
        else:
            up1_input_ch = downconv_channels[-1]
        
        if self.action_as_image or (not self.no_image):
            upconv_layers = []
            for l in range(L_layers):
                in_channels = up1_input_ch if l == 0 else upconv_channels[2*l-1]
                conv_in_channels = upconv_channels[2*l] + downconv_channels[L_layers-1-l]
                this_layer = Up(in_channels, upconv_channels[2*l],conv_in_channels, upconv_channels[2*l+1], 2, self.bilinear)
                upconv_layers.append(this_layer)
            if hp['use_resnet'] and hp['resnet_maxpool_after_enc']:
                this_layer = UpNoSkipConnection(upconv_channels[-1], upconv_channels[-1], upconv_channels[-1], 2, bilinear=self.bilinear)
                upconv_layers.append(this_layer)
            self.upconv = nn.ModuleList(upconv_layers)
        
        if self.no_image:
            self.outc = OutConv(action_upconv_channels[-1], self.n_classes)
        else:
            self.outc = OutConv(upconv_channels[-1], self.n_classes)

    def forward(self, x, action_info):
        if self.no_image and self.action_as_image:
            x = x[:,4:,:,:]
        if self.hp['no_rgb_feature']:
            x = x[:,3:,:,:]
        if self.action_as_image or (not self.no_image):
            x_enc = self.inc(x)
            x_downs = [x_enc] # L_layers + 1
            # print(x_enc.shape)
            for down_idx, down_layer in enumerate(self.downconv):
                x_down = down_layer(x_downs[-1])
                # print(x_down.shape)
                x_downs.append(x_down)
            
            height,width = x_downs[-1].shape[-2:]
        else:
            height,width = x.shape[-2:]
        
        if self.hp['predict_precondition']:
            action_info_enc = self.action_inc(action_info)
            outprecond_input = torch.cat([x_downs[-1].view(x_downs[-1].size(0), -1), action_info_enc], dim=1)
            logits = self.outprecond(outprecond_input)
            return logits

        if (not self.action_as_image) and (not self.no_action):
            action_info_enc = self.action_inc(action_info)
            action_info_enc = torch.unsqueeze(torch.unsqueeze(action_info_enc, 2), 3)
            if self.has_action_upconv:
                self.action_upconv_sizes[-1] = height
                action_encs = [action_info_enc]
                for action_idx, action_up_layer in enumerate(self.action_upconv):
                    
                    action_encres = action_up_layer(action_encs[-1], height=self.action_upconv_sizes[action_idx], width=self.action_upconv_sizes[action_idx])
                    action_encs.append(action_encres)
                action_info_enc = action_encs[-1]
                # action_info_enc = self.action_upconv(action_info_enc)
                # padding if necessary
                # diffY = height - action_info_enc.size()[2]
                # diffX = width - action_info_enc.size()[3]
                # action_info_enc = F.pad(action_info_enc, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            else:
                action_info_enc = torch.tile(action_info_enc, (1,1,height,width))
            if self.no_image:
                x_down_last = action_info_enc
            else:
                x_down_last = torch.cat([x_downs[-1], action_info_enc], dim=1)
        else:
            x_down_last = x_downs[-1]
        
        if self.action_as_image or (not self.no_image):
            x_ups = [x_down_last]
            # print(x_down_last.shape)
            for up_idx, up_layer in enumerate(self.upconv[:-1]):
                x_up = up_layer(x_ups[-1], x_downs[self.L_layers-1-up_idx])
                # print(x_up.shape)
                x_ups.append(x_up)
            up_layer = self.upconv[-1]
            if self.hp['use_resnet'] and self.hp['resnet_maxpool_after_enc']:
                x_up = up_layer(x_ups[-1], height=x.shape[2], width=x.shape[3])
                x_ups.append(x_up)
            else:
                x_up = up_layer(x_ups[-1], x_downs[0])
                # print(x_up.shape)
                x_ups.append(x_up)
        if self.action_as_image or (not self.no_image):
            logits = self.outc(x_ups[-1])
        else:
            logits = self.outc(x_down_last)
        return logits

class HeightMapOnly(nn.Module):
    def __init__(self, hp):
        super(HeightMapOnly, self).__init__()
        self.n_channels = 1
        self.hp = hp
        self.n_classes = len(hp['classes'])+1
        
        self.bilinear = False
        self.no_action = hp['no_action']

        downconv_channels = hp['downconv_channels']
        
        action_upconv_channels = hp['action_upconv_channels']
        self.action_upconv_sizes = hp['action_upconv_sizes'] if hp['action_upconv_sizes'] is not None else []
        if not self.no_action:
            # self.enc = nn.Conv2d(self.n_channels, downconv_channels[0], kernel_size=1, bias=False)
            self.enc = nn.Sequential(
                nn.Conv2d(self.n_channels, downconv_channels[0], kernel_size=150, bias=False),
                nn.BatchNorm2d(downconv_channels[0]),
                nn.ReLU(inplace=True)
            )
            # self.enc = SingleConv(self.n_channels, downconv_channels[0])
            
            self.action_inc = nn.Linear(hp['action_input_dim'], action_upconv_channels[0])
            action_upconv_layers = get_action_layers(action_upconv_channels, len(self.action_upconv_sizes))
            if len(action_upconv_layers) == 0:
                self.has_action_upconv = False 
            else:
                self.has_action_upconv = True
                self.action_upconv = nn.ModuleList(action_upconv_layers)
            action_upconv_channels = action_upconv_channels[:(len(self.action_upconv_sizes)*2+1)]
            up1_input_ch = downconv_channels[0]+action_upconv_channels[-1]
            self.outc = OutConv(up1_input_ch, self.n_classes)
        else:
            self.enc = nn.Sequential(
                nn.Conv2d(self.n_channels, self.n_classes * 150 * 150, kernel_size=150, bias=False),
                nn.BatchNorm2d(self.n_classes * 150 * 150),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x, action_info):
        x = x[:,3:4,...]
        height,width = x.shape[-2:]
        
        if not self.no_action:
            x_enc = self.enc(x)
            action_info_enc = self.action_inc(action_info)
            action_info_enc = torch.unsqueeze(torch.unsqueeze(action_info_enc, 2), 3)
            if self.has_action_upconv:
                self.action_upconv_sizes[-1] = height
                action_encs = [action_info_enc]
                for action_idx, action_up_layer in enumerate(self.action_upconv):
                    action_encres = action_up_layer(action_encs[-1], height=self.action_upconv_sizes[action_idx], width=self.action_upconv_sizes[action_idx])
                    action_encs.append(action_encres)
                action_info_enc = action_encs[-1]
            else:
                action_info_enc = torch.tile(action_info_enc, (1,1,height,width))
            x_down_last = torch.cat([x_enc, action_info_enc], dim=1) 
            logits = self.outc(x_down_last)   
        else:

            logits = self.enc(x).view((-1, self.n_classes, height, width)).contiguous()
        
        return logits

class UNetSeparateRgbDepth(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetSeparateRgbDepth, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = SingleConv(3, 64)
        self.down1 = DownRelu(64, 64, kernel_size=3, stride=2, bias=False)
        self.down2 = DownRelu(64, 128, kernel_size=3, stride=2, bias=False)
        self.down3 = DownRelu(128, 256, kernel_size=3, stride=2, bias=False)
        self.down4 = DownRelu(256, 512, kernel_size=3, stride=2, bias=False)

        self.inc_depth = SingleConv(1, 64)
        self.down1_depth = DownRelu(64, 64, kernel_size=3, stride=2, bias=False)
        self.down2_depth = DownRelu(64, 128, kernel_size=3, stride=2, bias=False)
        self.down3_depth = DownRelu(128, 256, kernel_size=3, stride=2, bias=False)
        self.down4_depth = DownRelu(256, 512, kernel_size=3, stride=2, bias=False)
        
        factor = 2 if bilinear else 1
        self.action_inc = nn.Linear(7+2, 64)
        
        self.up1 = Up(512+512+64, 512, 512+256, 512, 2, bilinear)
        self.up2 = Up(512, 256, 256+128, 256, 2, bilinear)
        self.up3 = Up(256, 128, 128+64, 128, 2, bilinear)
        self.up4 = Up(128, 64, 64+64, 64, 2, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, action_info):
        rgb_inputs = x[:,:3,:,:]
        depth_inputs = x[:,3:,:,:]

        depth_enc = self.inc_depth(depth_inputs)
        x1_depth = self.down1_depth(depth_enc)
        x2_depth = self.down2_depth(x1_depth)
        x3_depth = self.down3_depth(x2_depth)
        x4_depth = self.down4_depth(x3_depth)

        x_enc = self.inc(rgb_inputs)
        x1 = self.down1(x_enc)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        _,_,height,width = x4.shape
        
        action_info1 = self.action_inc(action_info)
        action_info1 = torch.unsqueeze(torch.unsqueeze(action_info1, 2), 3)
        action_info1 = torch.tile(action_info1, (1,1,height,width))
        x6 = torch.cat([x4, x4_depth, action_info1], dim=1)
        # x6 = x4
        
        xf = self.up1(x6, x3) # x3: 256
        xf = self.up2(xf, x2) # x2: 128
        xf = self.up3(xf, x1) # x1: 64
        xf = self.up4(xf, x_enc) # x_enc: 64
        
        logits = self.outc(xf)
        return logits

class UNetNoImageInput(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(UNetNoImageInput, self).__init__()
        self.n_channels = 1
        self.n_classes = n_classes
        self.bilinear = bilinear

        
        factor = 2 if bilinear else 1
        self.action_inc = nn.Linear(7+2, 64)
        
        self.up1 = UpNoSkipConnection(64, 128, 128, 128, 2, bilinear)
        self.up2 = UpNoSkipConnection(128, 256, 256, 256, 2, bilinear)
        self.up3 = UpNoSkipConnection(256, 512, 512, 512, 2, bilinear)
        self.up4 = UpNoSkipConnection(512, 256, 256, 256, 2, bilinear)
        self.up5= UpNoSkipConnection(256, 128, 128, 128, 2, bilinear)
        self.up6 = UpNoSkipConnection(128, 64, 64, 64, 2, bilinear)
        self.up7 = UpNoSkipConnection(64, 64, 64, 64, 2, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, action_info):
        height,width = 8,8
        
        action_info1 = self.action_inc(action_info)
        action_info1 = torch.unsqueeze(torch.unsqueeze(action_info1, 2), 3)
        # action_info1 = torch.tile(action_info1, (1,1,height,width))
        x6 = action_info1
        # import pdb; pdb.set_trace()
        xf = self.up1(x6, 2,2) 
        xf = self.up2(xf, 4,4) 
        xf = self.up3(xf, 8,8) 
        xf = self.up4(xf, 17,17) 
        xf = self.up5(xf, 37,37) 
        xf = self.up6(xf, 74,74) 
        xf = self.up7(xf, 150,150) 
        logits = self.outc(xf)
        return logits

class UNetNoImageInputActionImage(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(UNetNoImageInputActionImage, self).__init__()
        self.n_channels = 1
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = SingleConv(1, 64)
        self.down1 = DownMaxpoolSingleConv(64, 64, kernel_size=2)
        self.down2 = DownMaxpoolSingleConv(64, 128, kernel_size=2)
        self.down3 = DownMaxpoolSingleConv(128, 128, kernel_size=2)
        self.down4 = DownMaxpoolSingleConv(128, 256, kernel_size=2)
        self.down5 = DownMaxpoolSingleConv(256, 256, kernel_size=2)
        self.down6 = DownMaxpoolSingleConv(256, 512, kernel_size=2)
        self.down7 = DownMaxpoolSingleConv(512, 512, kernel_size=2)

        # self.up1 = Up(512, 512, 512+256, 512, 2, bilinear)
        # self.up2 = Up(512, 256, 256+128, 256, 2, bilinear)
        # self.up3 = Up(256, 128, 128+64, 128, 2, bilinear)
        # self.up4 = Up(128, 64, 64+64, 64, 2, bilinear)

        self.up1 = Up(512, 128, 128+512, 128, 2, bilinear)
        self.up2 = Up(128, 256, 256+256, 256, 2, bilinear)
        self.up3 = Up(256, 512, 512+256, 512, 2, bilinear)
        self.up4 = Up(512, 256, 256+128, 256, 2, bilinear)
        self.up5= Up(256, 128, 128+128, 128, 2, bilinear)
        self.up6 = Up(128, 64, 64+64, 64, 2, bilinear)
        self.up7 = Up(64, 64, 64+64, 64, 2, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x, action_info):
        x_enc = self.inc(x[:,4:,:,:])
        x1 = self.down1(x_enc)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
          
        xf = self.up1(x7, x6) 
        xf = self.up2(xf, x5)
        xf = self.up3(xf, x4) 
        xf = self.up4(xf, x3) 
        xf = self.up5(xf, x2) 
        xf = self.up6(xf, x1) 
        xf = self.up7(xf, x_enc) 
        
        logits = self.outc(xf)
        return logits