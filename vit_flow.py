import torch
import networkx as nx
import numpy as np

def flow(attentions, fusion='mean'):
    """
    Given a list of attention tensors (e.g., self.attentions with shape [1, num_heads, num_tokens, num_tokens])
    and a fusion method ('mean', 'min', or 'max'), this function constructs a layered graph 
    to compute the attention flow (via a maximum flow algorithm) from the CLS token in the final layer 
    to the input tokens. It returns a 2D heatmap (mask) showing the relative attention flow for the image patches.
    
    Assumptions:
      - Each attention tensor corresponds to one transformer layer.
      - The tokens consist of 1 [CLS] token plus the image patches (e.g. 197 tokens = 1 CLS + 196 patches).
      - The number of image patches (here n-1) is a perfect square (e.g. 196 → 14×14).
    
    Parameters:
      attentions (list[torch.Tensor]): List of attention tensors.
      fusion (str): Method to fuse attention heads. Options: 'mean', 'min', or 'max'.
    
    Returns:
      heatmap (np.ndarray): 2D numpy array (e.g. 14x14) normalized to [0, 1] indicating the attention flow.
    """
    
    def adjust_attention(att, fusion):
        # att: [1, num_heads, num_tokens, num_tokens]
        if fusion == 'mean':
            att_fused = att.mean(dim=1)  # [1, num_tokens, num_tokens]
        elif fusion == 'max':
            att_fused = att.max(dim=1)[0]
        elif fusion == 'min':
            att_fused = att.min(dim=1)[0]
        else:
            raise ValueError("Unsupported fusion type: choose 'mean', 'min', or 'max'.")
        
        # Remove the batch dimension -> [num_tokens, num_tokens]
        A = att_fused[0]
        n = A.shape[0]
        # Incorporate residual connection: add identity matrix and average
        I = torch.eye(n, device=A.device)
        A_adjusted = (A + I) / 2
        # Normalize each row so they sum to 1
        A_adjusted = A_adjusted / A_adjusted.sum(dim=-1, keepdim=True)
        return A  # [num_tokens, num_tokens]
    
    # Process each layer's attention matrix using the chosen fusion method.
    layers = [adjust_attention(att, fusion) for att in attentions]
    L = len(layers)  # number of layers
    n = layers[0].shape[0]  # total tokens per layer (e.g., 197)
    
    print("Step 1")

    # Build the layered graph: nodes are (layer, token_index)
    G = nx.DiGraph()
    for l in range(L + 1):
        for i in range(n):
            G.add_node((l, i))

    print("Step 2")
    
    # For each layer l (1 to L), add directed edges from nodes in layer l to nodes in layer l-1.
    # layers[l-1] represents attention from tokens in layer l to tokens in layer l-1.
    for l in range(1, L + 1):
        A = layers[l - 1]  # shape: [n, n] where row j corresponds to token j in layer l
        for j in range(n):
            for i in range(n):
                capacity = float(A[j, i].item())
                if capacity > 0:
                    G.add_edge((l, j), (l - 1, i), capacity=capacity)
    
    print("Step 3")

    flow_dict = {}

    for i in range(n) :
        print(i)
        sink = (0,i)
        
        # Define the source as the CLS token in the final layer (assumed to be token index 0).
        source = (L, 0)
        
        # Compute maximum flow from the source to the sink.
        flow_value,_ = nx.maximum_flow(G, source, sink)

        flow_dict[i] = flow_value
        print(flow_value)

    print("Step 4")
    
    # Extract flow reaching each input token in layer 0.
    input_flows = np.zeros(n)
    for i in range(n):
        input_flows[i] = flow_dict.get(i)
    
    input_flows = np.max(input_flows) - input_flows
    
    print("Step 5")
    
    # Typically, token 0 in the input layer is the [CLS] token; discard it.
    patch_flows = input_flows[1:]  # remaining tokens are image patches
    num_patches = patch_flows.shape[0]
    side = int(np.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError("Number of image patches is not a perfect square")
    
    # Reshape the patch flows into a 2D heatmap and normalize.
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
        # Register a hook for each attention drop module
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
