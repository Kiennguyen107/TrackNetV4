import torch
import torch.nn as nn
import torch.nn.functional as F

# The utility functions and MotionPromptLayer, FusionLayerTypeA, FusionLayerTypeB, _conv_block classes are assumed to be the same as in your prompt.
# I'll include them here for completeness but they don't need modification.

def rearrange_tensor(input_tensor, order):
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    return input_tensor.permute(*[order.index(dim) for dim in "BTCHW"])

def reverse_rearrange_tensor(input_tensor, order):
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    return input_tensor.permute(*["BTCHW".index(dim) for dim in order])

def power_normalization(input, a, b):
    return 1 / (1 + torch.exp(-(5 / (0.45 * torch.abs(torch.tanh(a)) + 1e-1)) * (torch.abs(input) - 0.6 * torch.tanh(b))))

class MotionPromptLayer(nn.Module):
    def __init__(self, penalty_weight=0.0):
        super(MotionPromptLayer, self).__init__()
        self.input_permutation = "BTCHW"
        self.input_color_order = "RGB"
        self.color_map = {'R': 0, 'G': 1, 'B': 2}
        self.gray_scale = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)

        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))
        
        self.lambda1 = penalty_weight

    def forward(self, video_seq):
        loss = torch.tensor(0.0, device=video_seq.device)

        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        norm_seq = video_seq * 0.225 + 0.45

        idx_list = [self.color_map[idx] for idx in self.input_color_order]
        gray_scale_tensor = self.gray_scale.to(video_seq.device)[idx_list]
        weights = gray_scale_tensor.to(dtype=norm_seq.dtype)
        grayscale_video_seq = torch.einsum("btcwh,c->btwh", norm_seq, weights)

        B, T, H, W = grayscale_video_seq.shape
        frame_diff = grayscale_video_seq[:, 1:] - grayscale_video_seq[:, :-1]

        attention_map = power_normalization(frame_diff, self.a, self.b)
        norm_attention = attention_map.unsqueeze(2)

        if self.training:
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = torch.sum(torch.square(temp_diff)) / (H * W * (T - 2) * B)
            loss = self.lambda1 * temporal_loss

        return attention_map, loss

class FusionLayerTypeA(nn.Module):
    def forward(self, inputs):
        feature_map, attention_map = inputs
        output_1 = feature_map[:, 0, :, :]
        output_2 = feature_map[:, 1, :, :] * attention_map[:, 0, :, :]
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 1, :, :]
        return torch.stack([output_1, output_2, output_3], dim=1)

class FusionLayerTypeB(nn.Module):
    def forward(self, inputs):
        feature_map, attention_map = inputs
        output_1 = feature_map[:, 0, :, :] * attention_map[:, 0, :, :]
        output_2 = feature_map[:, 1, :, :] * ((attention_map[:, 0, :, :] + attention_map[:, 1, :, :])/2)
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 1, :, :]
        return torch.stack([output_1, output_2, output_3], dim=1)

def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )

# New Fusion Layer for Player Detection
class FusionLayerPlayer(nn.Module):
    def forward(self, inputs):
        feature_map, attention_map = inputs
        # The player detection is less dependent on fine-grained motion
        # so we can use a simpler fusion strategy.
        # We'll use the average attention map to apply to all features.
        avg_attention = (attention_map[:, 0, :, :] + attention_map[:, 1, :, :]) / 2
        fused_output_1 = feature_map[:, 0, :, :] * avg_attention
        fused_output_2 = feature_map[:, 1, :, :] * avg_attention
        fused_output_3 = feature_map[:, 2, :, :] * avg_attention
        return torch.stack([fused_output_1, fused_output_2, fused_output_3], dim=1)

class TrackNetV5(nn.Module):
    def __init__(self, input_height, input_width, fusion_layer_type="TypeA"):
        super(TrackNetV5, self).__init__()
        self.input_height = input_height
        self.input_width = input_width

        # Ball Detection Fusion Layer
        if fusion_layer_type == "TypeA":
            self.fusion_layer_ball = FusionLayerTypeA()
        elif fusion_layer_type == "TypeB":
            self.fusion_layer_ball = FusionLayerTypeB()
        else:
            raise ValueError("Unknown Motion Fusion Type")

        # Player Detection Fusion Layer
        self.fusion_layer_player = FusionLayerPlayer()

        self.motion_prompt = MotionPromptLayer()

        self.conv1 = _conv_block(9, 64)
        self.conv2 = _conv_block(64, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = _conv_block(64, 128)
        self.conv4 = _conv_block(128, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv5 = _conv_block(128, 256)
        self.conv6 = _conv_block(256, 256)
        self.conv7 = _conv_block(256, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv8 = _conv_block(256, 512)
        self.conv9 = _conv_block(512, 512)
        self.conv10 = _conv_block(512, 512)
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv11 = _conv_block(768, 256)
        self.conv12 = _conv_block(256, 256)
        self.conv13 = _conv_block(256, 256)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv14 = _conv_block(384, 128)
        self.conv15 = _conv_block(128, 128)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv16 = _conv_block(192, 64)
        self.conv17 = _conv_block(64, 64)
        
        # Ball detection head
        self.conv18_ball = nn.Sequential(
            nn.Conv2d(64, 3, 1, padding=0),
            nn.Sigmoid()
        )
        
        # Player detection head
        self.conv19_player = nn.Sequential(
            nn.Conv2d(64, 3, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        motion_input = x.view(-1, 3, 3, self.input_height, self.input_width)
        residual_maps, motion_loss = self.motion_prompt(motion_input)

        x1 = self.conv2(self.conv1(x))
        p1 = self.pool1(x1)
        x2 = self.conv4(self.conv3(p1))
        p2 = self.pool2(x2)
        x3 = self.conv7(self.conv6(self.conv5(p2)))
        p3 = self.pool3(x3)
        x4 = self.conv10(self.conv9(self.conv8(p3)))

        u1 = self.up1(x4)
        c1 = torch.cat([u1, x3], dim=1)
        x5 = self.conv13(self.conv12(self.conv11(c1)))

        u2 = self.up2(x5)
        c2 = torch.cat([u2, x2], dim=1)
        x6 = self.conv15(self.conv14(c2))

        u3 = self.up3(x6)
        c3 = torch.cat([u3, x1], dim=1)
        x7 = self.conv17(self.conv16(c3))

        # Ball Detection Branch
        x_ball_pre = self.conv18_ball[0](x7)
        # Apply fusion layer for ball detection
        x_ball_fused = self.fusion_layer_ball((x_ball_pre, residual_maps))
        out_ball = self.conv18_ball[1](x_ball_fused)

        # Player Detection Branch
        x_player_pre = self.conv19_player[0](x7)
        # Apply a new fusion layer for player detection
        x_player_fused = self.fusion_layer_player((x_player_pre, residual_maps))
        out_player = self.conv19_player[1](x_player_fused)
        
        return out_ball, out_player, motion_loss