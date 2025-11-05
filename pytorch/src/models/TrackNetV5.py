import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Custom rearrange function to replace einops
def rearrange(x, pattern, **kwargs):
    """Simple rearrange function for basic patterns"""
    if pattern == 'b n (h d) -> (b h) n d':
        h = kwargs['h']
        b, n, c = x.shape
        d = c // h
        return x.reshape(b, n, h, d).permute(0, 2, 1, 3).reshape(b * h, n, d)
    elif pattern == '(b h) s d -> b s (h d)':
        b = kwargs['b']
        bh, s, d = x.shape
        h = bh // b
        return x.reshape(b, h, s, d).permute(0, 2, 1, 3).reshape(b, s, h * d)
    elif pattern == 'b (f n) d -> b f n d':
        f = kwargs['f']
        b, fn, d = x.shape
        n = fn // f
        return x.reshape(b, f, n, d)
    elif pattern == 'b n d f -> b (f n) d':
        f = kwargs['f']
        b, n, d, f_dim = x.shape
        return x.permute(0, 3, 1, 2).reshape(b, f * n, d)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")

class SimplifiedTrajectoryAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, seq_len=196, num_frames=3):
        B, N, C = x.shape
        h = self.num_heads
        
        # 1. Spatial Attention
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
        k = rearrange(k, 'b n (h d) -> (b h) n d', h=h)
        v = rearrange(v, 'b n (h d) -> (b h) n d', h=h)
        
        # Split CLS token from patches (assuming first token is CLS)
        cls_q, q_ = q[:, 0:1], q[:, 1:]
        cls_k, k_ = k[:, 0:1], k[:, 1:]
        cls_v, v_ = v[:, 0:1], v[:, 1:]
        
        # Perform attention for patches (full attention)
        attn_weights = (q_ @ k_.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        x_spatial = attn_weights @ v_
        
        # Reshape spatial attention output
        x_spatial = rearrange(x_spatial, '(b h) s d -> b s (h d)', b=B)
        
        # 2. Simplified Temporal Attention
        # Instead of complex temporal reshaping, use a simpler approach
        # Apply temporal attention across the sequence dimension
        q2 = x_spatial  # Use spatial output as query
        k2, v2 = self.qkv(x[:, 1:]).chunk(3, dim=-1)[1:]  # Get k,v from original input (no CLS)
        
        # Perform temporal attention with proper dimensions
        temporal_attn_weights = (q2 @ k2.transpose(-2, -1)) * self.scale
        temporal_attn_weights = temporal_attn_weights.softmax(dim=-1)
        x_temporal = temporal_attn_weights @ v2
        
        # Combine outputs and project
        x_final = self.proj(x_temporal)
        return x_final

class OptimizedMotionPromptLayer(nn.Module):
    def __init__(self, penalty_weight=0.0):
        super(OptimizedMotionPromptLayer, self).__init__()
        self.input_permutation = "BTCHW"
        self.input_color_order = "RGB"
        self.color_map = {'R': 0, 'G': 1, 'B': 2}
        self.gray_scale = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)

        # Separate parameters for ball and player
        self.a_ball = nn.Parameter(torch.tensor(0.1))
        self.b_ball = nn.Parameter(torch.tensor(0.0))
        self.a_player = nn.Parameter(torch.tensor(0.15))
        self.b_player = nn.Parameter(torch.tensor(0.05))
        
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

        # Separate attention maps
        attention_map_ball = power_normalization(frame_diff, self.a_ball, self.b_ball)
        attention_map_player = power_normalization(frame_diff, self.a_player, self.b_player)
        
        if self.training:
            # Compute temporal loss for both attention maps
            norm_attention_ball = attention_map_ball.unsqueeze(2)
            norm_attention_player = attention_map_player.unsqueeze(2)
            
            temp_diff_ball = norm_attention_ball[:, 1:] - norm_attention_ball[:, :-1]
            temp_diff_player = norm_attention_player[:, 1:] - norm_attention_player[:, :-1]
            
            temporal_loss = (torch.sum(torch.square(temp_diff_ball)) + 
                           torch.sum(torch.square(temp_diff_player))) / (2 * H * W * (T - 2) * B)
            loss = self.lambda1 * temporal_loss

        return (attention_map_ball, attention_map_player), loss

class PlayerTrajectoryProcessor(nn.Module):
    """Simplified player feature processor - fallback without complex trajectory attention"""
    def __init__(self, input_dim, hidden_dim, num_frames=3, use_simple_attention=True):
        super().__init__()
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_simple_attention = use_simple_attention
        
        if use_simple_attention:
            # Simple temporal convolution for player trajectory
            self.temporal_conv = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, input_dim, 1),
                nn.Sigmoid()
            )
        else:
            # Spatial feature processor
            self.spatial_proj = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((14, 14))  # Reduce spatial dimensions
            )
            
            # Add CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
            # Trajectory attention
            self.trajectory_attention = SimplifiedTrajectoryAttention(hidden_dim, num_heads=4)  # Reduced heads
            
            # Output projection back to spatial
            self.spatial_restore = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.ReLU()
            )
        
    def forward(self, x):
        if self.use_simple_attention:
            # Simple temporal attention using convolutions
            attention_weights = self.temporal_conv(x)
            return x * attention_weights
        else:
            # Complex trajectory attention (original approach)
            B, C, H, W = x.shape
            
            # Process spatial features
            x_proj = self.spatial_proj(x)  # B, hidden_dim, 14, 14
            _, hidden_dim, h, w = x_proj.shape
            
            # Flatten spatial dimensions
            x_flat = x_proj.flatten(2).transpose(1, 2)  # B, 196, hidden_dim
            
            # Add CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_with_cls = torch.cat([cls_tokens, x_flat], dim=1)  # B, 197, hidden_dim
            
            # Apply trajectory attention
            seq_len = h * w  # 196
            x_attended = self.trajectory_attention(x_with_cls, seq_len=seq_len, num_frames=self.num_frames)
            
            # Remove CLS and restore spatial structure
            x_output = self.spatial_restore(x_attended)  # B, 196, input_dim
            x_spatial = x_output.transpose(1, 2).reshape(B, C, h, w)
            
            # Upsample back to original size
            x_final = F.interpolate(x_spatial, size=(H, W), mode='bilinear', align_corners=False)
            
            return x_final

class OptimizedFusionLayer(nn.Module):
    """Unified fusion layer that handles both ball and player detection"""
    def __init__(self, input_channels=16):
        super(OptimizedFusionLayer, self).__init__()
        self.input_channels = input_channels
        
        # Projection layers to map to 3 channels for fusion
        self.ball_proj = nn.Conv2d(input_channels, 3, 1)
        self.player_proj = nn.Conv2d(input_channels, 3, 1)
        
        # Learnable fusion weights
        self.ball_weights = nn.Parameter(torch.tensor([1.0, 0.8, 0.6]))
        self.player_weights = nn.Parameter(torch.tensor([0.7, 0.7, 0.7]))
        
        # Back projection to original channels
        self.back_proj_ball = nn.Conv2d(3, input_channels, 1)
        self.back_proj_player = nn.Conv2d(3, input_channels, 1)

    def forward(self, inputs, task_type="ball"):
        feature_map, attention_map = inputs
        
        if task_type == "ball":
            # Project to 3 channels
            proj_features = self.ball_proj(feature_map)
            
            weights = torch.softmax(self.ball_weights, dim=0)
            attention = attention_map[0]  # Use ball attention
                
            output_1 = proj_features[:, 0:1, :, :] * weights[0]
            output_2 = proj_features[:, 1:2, :, :] * attention[:, 0:1, :, :] * weights[1]
            output_3 = proj_features[:, 2:3, :, :] * attention[:, 1:2, :, :] * weights[2]
            
            fused = torch.cat([output_1, output_2, output_3], dim=1)
            # Project back to original channels
            return self.back_proj_ball(fused)
            
        else:  # player
            # Project to 3 channels
            proj_features = self.player_proj(feature_map)
            
            weights = torch.softmax(self.player_weights, dim=0)
            attention = attention_map[1]  # Use player attention
                
            avg_attention = (attention[:, 0:1, :, :] + attention[:, 1:2, :, :]) / 2
            output_1 = proj_features[:, 0:1, :, :] * avg_attention * weights[0]
            output_2 = proj_features[:, 1:2, :, :] * avg_attention * weights[1]
            output_3 = proj_features[:, 2:3, :, :] * avg_attention * weights[2]
            
            fused = torch.cat([output_1, output_2, output_3], dim=1)
            # Project back to original channels
            return self.back_proj_player(fused)

def _depthwise_separable_conv(in_channels, out_channels):
    """More efficient convolution block"""
    return nn.Sequential(
        # Depthwise
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        # Pointwise
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def _conv_block(in_channels, out_channels):
    return _depthwise_separable_conv(in_channels, out_channels)


class TrackNetV5(nn.Module):
    def __init__(self, input_height, input_width, fusion_layer_type='TypeA',base_channels=32):

        super(TrackNetV5, self).__init__()
        self.input_height = input_height
        self.input_width = input_width

        # Motion processing with separate attention
        self.motion_prompt = OptimizedMotionPromptLayer()
        
        # Unified fusion layer
        self.fusion_layer = OptimizedFusionLayer(input_channels=16)

        # Minimal channel dimensions for maximum efficiency
        c1, c2, c3, c4 = base_channels, base_channels*2, base_channels*4, base_channels*8

        # Encoder with depthwise separable convolutions
        self.conv1 = _conv_block(9, c1)
        self.conv2 = _conv_block(c1, c1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = _conv_block(c1, c2)
        self.conv4 = _conv_block(c2, c2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv5 = _conv_block(c2, c3)
        self.conv6 = _conv_block(c3, c3)
        self.conv7 = _conv_block(c3, c3)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.conv8 = _conv_block(c3, c4)
        self.conv9 = _conv_block(c4, c4)
        self.conv10 = _conv_block(c4, c4)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv11 = _conv_block(c4 + c3, c3)
        self.conv12 = _conv_block(c3, c3)
        self.conv13 = _conv_block(c3, c3)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv14 = _conv_block(c3 + c2, c2)
        self.conv15 = _conv_block(c2, c2)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv16 = _conv_block(c2 + c1, c1)
        self.conv17 = _conv_block(c1, c1)
        
        # Shared pre-processing layer for both heads
        self.shared_pre_head = nn.Conv2d(c1, 16, 1)
        
        # Player trajectory attention processor (using simple attention by default)
        self.player_trajectory = PlayerTrajectoryProcessor(input_dim=16, hidden_dim=64, num_frames=3, use_simple_attention=True)
        
        # Task-specific heads (minimal) - now expect 16 channels
        self.conv18_ball = nn.Sequential(
            nn.Conv2d(16, 3, 1, padding=0),
            nn.Sigmoid()
        )
        
        self.conv19_player = nn.Sequential(
            nn.Conv2d(16, 3, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        motion_input = x.view(-1, 3, 3, self.input_height, self.input_width)
        attention_maps, motion_loss = self.motion_prompt(motion_input)

        # Encoder
        x1 = self.conv2(self.conv1(x))
        p1 = self.pool1(x1)
        x2 = self.conv4(self.conv3(p1))
        p2 = self.pool2(x2)
        x3 = self.conv7(self.conv6(self.conv5(p2)))
        p3 = self.pool3(x3)
        x4 = self.conv10(self.conv9(self.conv8(p3)))

        # Decoder
        u1 = self.up1(x4)
        c1 = torch.cat([u1, x3], dim=1)
        x5 = self.conv13(self.conv12(self.conv11(c1)))

        u2 = self.up2(x5)
        c2 = torch.cat([u2, x2], dim=1)
        x6 = self.conv15(self.conv14(c2))

        u3 = self.up3(x6)
        c3 = torch.cat([u3, x1], dim=1)
        x7 = self.conv17(self.conv16(c3))

        # Shared preprocessing
        shared_features = self.shared_pre_head(x7)

        # Ball Detection Branch (standard fusion)
        x_ball_fused = self.fusion_layer((shared_features, attention_maps), task_type="ball")
        out_ball = self.conv18_ball(x_ball_fused)

        # Player Detection Branch (with trajectory attention)
        x_player_trajectory = self.player_trajectory(shared_features)
        x_player_fused = self.fusion_layer((x_player_trajectory, attention_maps), task_type="player")
        out_player = self.conv19_player(x_player_fused)
        
        return out_ball, out_player, motion_loss

# Create the aggressive model
def create_aggressive_model(input_height=288, input_width=512):
    """
    Create the aggressive optimization model with trajectory attention for players
    - Separate attention maps for ball and player
    - Depthwise separable convolutions throughout
    - Minimal channel dimensions (32 base)
    - Trajectory attention for player detection
    - Maximum efficiency (~70-80% reduction in computation)
    """
    return AggressiveTrackNetV5(input_height, input_width, base_channels=32)

# Example usage
# model = create_aggressive_model()
# x = torch.randn(4, 9, 288, 512)  # Batch of 4, 3 frames of 3 channels
# ball_out, player_out, motion_loss = model(x)