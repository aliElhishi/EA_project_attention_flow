
import torch
import networkx as nx
import numpy as np
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

# code inspired from https://github.com/jacobgil/vit-explain.git

def flow(attentions, fusion='mean'):
    """
    Given a list of attention tensors (with shape [1, num_heads, num_tokens, num_tokens])
    and a fusion method ('mean', 'min', or 'max'), this function constructs a layered graph
    to compute the attention flow (via a maximum flow algorithm) from the CLS token in the final layer
    to the input tokens. It returns a 2D heatmap (mask) showing the maximum attention flow for each image patch.

    In:
      attentions (list[torch.Tensor]): List of attention tensors.
      fusion (str): Method to fuse attention heads. Options: 'mean', 'min', or 'max'.

    Out:
      heatmap (np.ndarray): 2D numpy array (e.g. 14x14) normalized to [0, 1] indicating the attention flow.
    """

    def adjust_attention(att, fusion):
        if fusion == 'mean':
            att_fused = att.mean(dim=1)
        elif fusion == 'max':
            att_fused = att.max(dim=1)[0]
        elif fusion == 'min':
            att_fused = att.min(dim=1)[0]
        else:
            raise ValueError("Unsupported fusion type: choose 'mean', 'min', or 'max'.")

        A = att_fused[0]
        n = A.shape[0]
        I = torch.eye(n, device=A.device)
        A_adjusted = (A + I) / 2
        A_adjusted = A_adjusted / A_adjusted.sum(dim=-1, keepdim=True)
        return A

    layers = [adjust_attention(att, fusion) for att in attentions]
    L = len(layers)
    n = layers[0].shape[0]

    print("Step 1")

    G = nx.DiGraph()
    for l in range(L + 1):
        for i in range(n):
            G.add_node((l, i))

    print("Step 2")

    for l in range(1, L + 1):
        A = layers[l - 1]
        for j in range(n):
            for i in range(n):
                capacity = float(A[j, i].item())
                if capacity > 0:
                    G.add_edge((l, j), (l - 1, i), capacity=capacity)

    print("Step 3")

    flow_dict = {}

    for i in range(n) :
        # print(i)
        sink = (0,i)
        source = (L, 0)

        flow_value,_ = nx.maximum_flow(G, source, sink)
        flow_dict[i] = flow_value
        # print(flow_value)

    print("Step 4")

    input_flows = np.zeros(n)
    for i in range(n):
        input_flows[i] = flow_dict.get(i)

    input_flows = np.max(input_flows) - input_flows

    print("Step 5")

    patch_flows = input_flows[1:]
    num_patches = patch_flows.shape[0]
    side = int(np.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError("Number of image patches is not a perfect square")

    heatmap = patch_flows.reshape(side, side)
    heatmap = heatmap / heatmap.max()
    print("Heatmap shape:", heatmap.shape)
    print("Heatmap (normalized):")
    print(heatmap)
    return heatmap

class VITAttentionFlow:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean"):
        self.model = model
        self.head_fusion = head_fusion
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            _ = self.model(input_tensor)
        return flow(self.attentions, self.head_fusion)
