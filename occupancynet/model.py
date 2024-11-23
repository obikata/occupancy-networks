import torch
import torch.nn as nn
import torchvision.models as models


import torch
import torch.nn as nn
import torchvision.models as models


class OccupancyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = MultiFeatureExtractor(backbone='resnet18', num_images=2)
        self.attn_module = AttentionModule(embed_dim=512, num_heads=8, mlp_dim=512)
        self.deconvnet = DeconvNet()

    def forward(self, left_image, right_image):
        # PATCHING AND EMBEDDING
        left_features, right_features = self.feature_extractor([left_image, right_image])
        left_features_flat = left_features.view(left_features.size(0), left_features.size(1), -1).permute(0, 2, 1)
        right_features_flat = right_features.view(right_features.size(0), right_features.size(1), -1).permute(0, 2, 1)

        # ENCODING
        fused_features = self.attn_module(left_features_flat, right_features_flat)

        # 3D DECONVOLUTIONS
        fused_features = fused_features.view(-1, 512, 7, 7, 7)  # reshaping to 3D
        output = self.deconvnet(fused_features)

        return output
        

class MultiFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18', weights=models.ResNet18_Weights, num_images=2):
        super().__init__()
        self.model = getattr(models, backbone)(weights=models.ResNet18_Weights)
        self.feature_extractors = nn.ModuleList([nn.Sequential(*list(self.model.children())[:-2]) for _ in range(num_images)])

    def forward(self, images):
        return [feature_extractor(x) for x, feature_extractor in zip(images, self.feature_extractors)]
    

class AttentionModule(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_dim=512, max_seq_length=1024, grid_size=7):
        super().__init__()

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(grid_size**3, embed_dim))
        self.max_seq_length = max_seq_length  # Maximum expected sequence length for padding

        # Class token
        self.class_tokens = nn.Parameter(torch.randn(1, embed_dim))  # 1 for one class token

        # 2D Random Positional Embedding
        # Create a parameter directory with the correct shape
        self.left_pos_embedding = nn.Parameter(torch.randn(max_seq_length, embed_dim))
        self.right_pos_embedding = nn.Parameter(torch.randn(max_seq_length, embed_dim))

        # 3D Random Positional Embedding
        # Create a parameter directly with the correct shape
        self.pos_embedding_3d = nn.Parameter(torch.randn(grid_size**3, embed_dim))

        # Transformer encoder
        self.transformer = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)

    def forward(self, left_features, right_features):

        batch_size, left_seq_length, _ = left_features.size()
        _, right_seq_length, _ = right_features.size()

        # Add positional embedding after class token
        left_features = left_features + self.left_pos_embedding[:left_seq_length].unsqueeze(0).expand(batch_size, -1, -1)
        right_features = right_features + self.right_pos_embedding[:right_seq_length].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate features along the sequence axis
        concat_features = torch.cat((left_features, right_features), dim=1)
        seq_length = concat_features.size(1)

        # Fixed Queries
        fixed_query_length = self.queries.size(0)

        # Expand queries to match the batch size of concat_features
        pos_embedding_3d = self.pos_embedding_3d.unsqueeze(0).expand(batch_size, -1, -1)
        fixed_queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        fixed_queries = fixed_queries + pos_embedding_3d

        # Pad concat_features to match or exceed queries' length
        if concat_features.size(1) < fixed_query_length:
            pad_length = fixed_query_length - concat_features.size(1)
            concat_features = torch.nn.functional.pad(concat_features, (0, 0, 0, pad_length))

        # Create mask for padded positions
        # mask shape should be[batch_size, seq_length] where True indicates positions to be ignored
        mask = torch.zeros(concat_features.size(0), concat_features.size(1), dtype=torch.bool, device=concat_features.device)
        if concat_features.size(1) > seq_length:
            mask[:, seq_length:] = True

        # Transform features using fixed queries
        fused_features = self.transformer(concat_features, fixed_queries, mask)

        return fused_features


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_dim=512):
        super().__init__()

        # Transformer layers
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=mlp_dim)
    
    def forward(self, x, fixed_queries=None, mask=None):

        # Normalization steps before attention for stability in training
        x1 = self.norm_1(x)

        if fixed_queries != None:
            q = fixed_queries
        else:
            q = x1
        k = x1
        v = x1
            
        # Attention computation
        # Note: We use batch_first=True, so we don't need to permute keys and values
        attn_output, _ = self.mha(q, k, v, key_padding_mask=mask, need_weights=False)

        # Residual connection
        # We only need to pad concat_features if it's shorter than attn_output which shouldn't happen if we've already padded it
        x = attn_output + x

        # Apply layer norm after the residual connection
        x2 = self.norm_2(x)

        # Apply MLP to further transform features
        mlp_output = self.mlp(x2)

        # Residual connection
        x = mlp_output + x

        return x


class MLP(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        
        # Start with 512 channels, reduce to 256, then to 128, then to 64, and finally to 1 channel
        # Each deconvolution will double the spatial dimensions
        self.deconv1 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=2, stride=2)
        
        # ReLU activation for intermediate layers
        self.relu = nn.ReLU()
        
        # Sigmoid to get binary output (0 or 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, 512, 7, 7, 7]
        x = self.relu(self.deconv1(x))
        # Shape is now [batch_size, 256, 14, 14, 14]
        x = self.relu(self.deconv2(x))
        # Shape is now [batch_size, 128, 28, 28, 28]
        x = self.relu(self.deconv3(x))
        # Shape is now [batch_size, 64, 56, 56, 56]
        x = self.sigmoid(self.deconv4(x))
        # Shape is now [batch_size, 1, 112, 112, 112]
        return x