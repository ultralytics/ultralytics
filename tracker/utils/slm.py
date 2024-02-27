import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler

from tracker.utils.models.osnet import osnet_x1_0
from tracker.utils.models.resnet import ResNet, resnet18

"""
Helper Functions
##############################################################################
"""


def get_scheduler(optimizer, args):
    """
    Return a learning rate scheduler.

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':

        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):

    def forward(self, x):
        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=1, output_sigmoid=False)

    elif args.net_G == 'osnet':
        # TODO: OSNET IMPORT

        net = osnet_x1_0(num_classes=1000, pretrained=True, loss='softmax')
    elif args.net_G == 'patch_trans':
        net = DLASeg()

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

        self.con1x1 = nn.Conv2d(planes, planes * 2, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes * 2)

    def forward(self, x, residual=None):
        #         if residual is None:
        #             residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #         out += residual
        out = self.relu(out)

        # out = self.con1x1(out)

        return out


class patchLinearAttention(nn.Module):

    def __init__(self,
                 chan=128,
                 chan_out=None,
                 kernel_size=1,
                 padding=0,
                 stride=1,
                 key_dim=32,
                 value_dim=32,
                 heads=4,
                 norm_queries=True):
        super().__init__()
        self.chan = chan
        chan_out = chan if chan_out is None else chan_out

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads

        self.norm_queries = norm_queries

        conv_kwargs = {'padding': padding, 'stride': stride}
        self.to_q = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_k = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_v = nn.Conv2d(chan, value_dim * heads, kernel_size, **conv_kwargs)

        out_conv_kwargs = {'padding': padding}
        self.to_out = nn.Conv2d(value_dim * heads, chan_out, kernel_size, **out_conv_kwargs)

    def forward(self, x, y, context=None):
        b, c, h, w, k_dim, heads = *x.shape, self.key_dim, self.heads

        q, k, v = (self.to_q(x), self.to_k(y), self.to_v(y))

        q, k, v = map(lambda t: t.reshape(b, heads, -1, h * w), (q, k, v))

        q, k = map(lambda x: x * (self.key_dim ** -0.25), (q, k))

        if context is not None:
            context = context.reshape(b, c, 1, -1)
            ck, cv = self.to_k(context), self.to_v(context)
            ck, cv = map(lambda t: t.reshape(b, heads, k_dim, -1), (ck, cv))
            k = torch.cat((k, ck), dim=3)
            v = torch.cat((v, cv), dim=3)

        k = k.softmax(dim=-1)

        if self.norm_queries:
            q = q.softmax(dim=-2)

        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhdn,bhde->bhen', q, context)
        out = out.reshape(b, -1, h, w)
        out = self.to_out(out)
        return out


class DLASeg(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = BasicBlock(3, 64)

        self.patch_attention = patchLinearAttention(chan=32)

        #         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        #         self.oneone = nn.Conv2d(512, 128, kernel_size=1)
        #         self.SELayer = SELayer(channel = 128)
        self.fc = self._construct_fc_layer(128, 128, dropout_p=None)
        self.resnet = resnet18(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.resnet_stages_num = 4
        expand = 1

        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError

        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def inference_forward_fast(self, x1):

        x1 = x1.unsqueeze(-1)

        x1 = x1.permute(3, 2, 0, 1)

        x1 = F.interpolate(x1, (224, 80), mode='bilinear')

        x1 = self.forward_single(x1)  # shape: (B,32,28,10)

        width = x1.shape[-1]
        height = x1.shape[-2]
        width = int(width)
        height = int(height)

        # 整張
        temp_all = x1
        # 左上
        temp_lup = x1[:, :, 0:(height // 2), 0:(width // 2)]
        # 右上
        temp_rup = x1[:, :, 0:(height // 2), (width // 2):width]
        # 左下
        temp_ldown = x1[:, :, (height // 2):height, 0:(width // 2)]
        # 右下
        temp_rdown = x1[:, :, (height // 2):height, (width // 2):width]

        # round1
        A = self.patch_attention(temp_lup, temp_lup)
        B = self.patch_attention(temp_lup, temp_rup)
        C = self.patch_attention(temp_lup, temp_ldown)
        D = self.patch_attention(temp_lup, temp_rdown)
        final1 = A + B + C + D

        # round2
        A = self.patch_attention(temp_rup, temp_rup)
        B = self.patch_attention(temp_rup, temp_lup)
        C = self.patch_attention(temp_rup, temp_ldown)
        D = self.patch_attention(temp_rup, temp_rdown)
        final2 = A + B + C + D

        # round3
        A = self.patch_attention(temp_ldown, temp_ldown)
        B = self.patch_attention(temp_ldown, temp_rup)
        C = self.patch_attention(temp_ldown, temp_lup)
        D = self.patch_attention(temp_ldown, temp_rdown)
        final3 = A + B + C + D

        # round4
        A = self.patch_attention(temp_rdown, temp_rdown)
        B = self.patch_attention(temp_rdown, temp_lup)
        C = self.patch_attention(temp_rdown, temp_rup)
        D = self.patch_attention(temp_rdown, temp_ldown)
        final4 = A + B + C + D

        v1 = torch.cat((final1, final2, final3, final4), 1)

        v1 = self.maxpool(v1)

        v1 = v1.squeeze(-1)
        v1 = v1.squeeze(-1)

        v1 = self.fc(v1)

        return v1

    def forward(self, x1, x2):

        x1 = x1.permute(0, 3, 1, 2)  # shape: (B,64,224,80)
        x2 = x2.permute(0, 3, 1, 2)
        x1 = x1.float()
        x2 = x2.float()

        # x1 = self.conv1(x1)  # shape: (B,64,56,20)
        # x2 = self.conv1(x2)
        x1 = self.forward_single(x1)  # shape: (B,32,28,10)
        x2 = self.forward_single(x2)

        width = x1.shape[-1]
        height = x1.shape[-2]
        width = int(width)
        height = int(height)

        # 整張
        temp_all = x1
        # 左上
        temp_lup = x1[:, :, 0:(height // 2), 0:(width // 2)]
        # 右上
        temp_rup = x1[:, :, 0:(height // 2), (width // 2):width]
        # 左下
        temp_ldown = x1[:, :, (height // 2):height, 0:(width // 2)]
        # 右下
        temp_rdown = x1[:, :, (height // 2):height, (width // 2):width]

        # round1
        A = self.patch_attention(temp_lup, temp_lup)
        B = self.patch_attention(temp_lup, temp_rup)
        C = self.patch_attention(temp_lup, temp_ldown)
        D = self.patch_attention(temp_lup, temp_rdown)
        final1 = A + B + C + D

        # round2
        A = self.patch_attention(temp_rup, temp_rup)
        B = self.patch_attention(temp_rup, temp_lup)
        C = self.patch_attention(temp_rup, temp_ldown)
        D = self.patch_attention(temp_rup, temp_rdown)
        final2 = A + B + C + D

        # round3
        A = self.patch_attention(temp_ldown, temp_ldown)
        B = self.patch_attention(temp_ldown, temp_rup)
        C = self.patch_attention(temp_ldown, temp_lup)
        D = self.patch_attention(temp_ldown, temp_rdown)
        final3 = A + B + C + D

        # round4
        A = self.patch_attention(temp_rdown, temp_rdown)
        B = self.patch_attention(temp_rdown, temp_lup)
        C = self.patch_attention(temp_rdown, temp_rup)
        D = self.patch_attention(temp_rdown, temp_ldown)
        final4 = A + B + C + D

        v1 = torch.cat((final1, final2, final3, final4), 1)

        v1 = self.maxpool(v1)

        v1 = v1.squeeze(-1)
        v1 = v1.squeeze(-1)

        v1 = self.fc(v1)

        # 整張
        temp_all = x2
        # 左上
        temp_lup = x2[:, :, 0:(height // 2), 0:(width // 2)]
        # 右上
        temp_rup = x2[:, :, 0:(height // 2), (width // 2):width]
        # 左下
        temp_ldown = x2[:, :, (height // 2):height, 0:(width // 2)]
        # 右下
        temp_rdown = x2[:, :, (height // 2):height, (width // 2):width]

        # round1
        A = self.patch_attention(temp_lup, temp_lup)
        B = self.patch_attention(temp_lup, temp_rup)
        C = self.patch_attention(temp_lup, temp_ldown)
        D = self.patch_attention(temp_lup, temp_rdown)
        final1 = A + B + C + D

        # round2
        A = self.patch_attention(temp_rup, temp_rup)
        B = self.patch_attention(temp_rup, temp_lup)
        C = self.patch_attention(temp_rup, temp_ldown)
        D = self.patch_attention(temp_rup, temp_rdown)
        final2 = A + B + C + D

        # round3
        A = self.patch_attention(temp_ldown, temp_ldown)
        B = self.patch_attention(temp_ldown, temp_rup)
        C = self.patch_attention(temp_ldown, temp_lup)
        D = self.patch_attention(temp_ldown, temp_rdown)
        final3 = A + B + C + D

        # round4
        A = self.patch_attention(temp_rdown, temp_rdown)
        B = self.patch_attention(temp_rdown, temp_lup)
        C = self.patch_attention(temp_rdown, temp_rup)
        D = self.patch_attention(temp_rdown, temp_ldown)
        final4 = A + B + C + D

        v2 = torch.cat((final1, final2, final3, final4), 1)

        v2 = self.maxpool(v2)

        v2 = v2.squeeze(-1)
        v2 = v2.squeeze(-1)

        v2 = self.fc(v2)

        sim = self.cos(v1, v2)

        return sim

    def forward_single(self, x):

        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


def load_model(model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    model = DLASeg()
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch_id']))
    state_dict_ = checkpoint['model_G_state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model
