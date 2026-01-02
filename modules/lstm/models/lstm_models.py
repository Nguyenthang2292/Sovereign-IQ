from typing import Literal
import torch
import torch.nn as nn

from modules.common.ui.logging import log_warn, log_model
from modules.lstm.core import (MultiHeadAttention, 
                                FeedForward, 
                                PositionalEncoding, 
                                CNN1DExtractor)
from config.lstm import (
    LSTM_HIDDEN_DIM_L2,
    LSTM_HIDDEN_DIM_L3,
    LSTM_ATTENTION_DIM,
    DROPOUT_FINAL_LAYER,
    CLASSIFIER_HIDDEN_DIM,
)

class LSTMModel(nn.Module):
    """
    Enhanced PyTorch LSTM model for time series classification (e.g., cryptocurrency price prediction).

    This model applies a stack of LSTM layers with regularization, configurable hidden sizes, and dropout.
    By default, the model predicts the probability distribution over three classes (SELL, NEUTRAL, BUY).
    
    Args:
        input_size (int): Number of input features.
        hidden_size (int): Hidden size for the first LSTM layer. Default is 64.
        num_layers (int): Number of stacked LSTM layers. Default is 3.
        num_classes (int): Number of output classes. Default is 3.
        dropout (float): Dropout probability (applied to all LSTM layers). Default is 0.3.
    """

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 3, 
        num_classes: int = 3, 
        dropout: float = 0.3,
        output_mode: str = 'classification'
    ) -> None:
        super().__init__()
        
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {hidden_size}")
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be an integer >= 1, got {num_layers}")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be a number between 0.0 and 1.0, got {dropout}")
        if output_mode not in ['classification', 'regression']:
            raise ValueError(f"output_mode must be 'classification' or 'regression', got {output_mode}")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_mode = output_mode

        # Support arbitrary depth LSTM stack by generalizing hidden dimensions
        self.lstm_layers = nn.ModuleList()
        # Build a list of hidden_sizes for each LSTM layer
        # First layer uses `hidden_size`, next two use config constants, rest use `hidden_size`
        hidden_sizes = [hidden_size]
        if num_layers > 1:
            hidden_sizes.append(LSTM_HIDDEN_DIM_L2)
        if num_layers > 2:
            hidden_sizes.append(LSTM_HIDDEN_DIM_L3)
        # For depths > 3, use `hidden_size` for the rest
        if num_layers > 3:
            hidden_sizes += [hidden_size] * (num_layers - 3)
        input_sizes = [input_size] + hidden_sizes[:-1]
        for in_size, out_size in zip(input_sizes, hidden_sizes):
            self.lstm_layers.append(nn.LSTM(in_size, out_size, num_layers=1, batch_first=True))

        # Dropout per LSTM layer (last one uses DROPOUT_FINAL_LAYER)
        dropout_probs = [dropout] * (num_layers - 1) + [DROPOUT_FINAL_LAYER]
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_probs[:num_layers]])

        # Final FC layers adapted to variable LSTM stack depth
        fc_in = hidden_sizes[-1]  # Use the last LSTM layer's output dimension        
        self.fc1 = nn.Linear(fc_in, CLASSIFIER_HIDDEN_DIM)
        self.fc2 = nn.Linear(CLASSIFIER_HIDDEN_DIM, num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Shape (batch_size, num_classes) (probabilities).
        """
        out = x
        # Process input through LSTM layers
        for i, (lstm_layer, dropout_layer) in enumerate(zip(self.lstm_layers, self.dropouts)):
            out, _ = lstm_layer(out)
            out = dropout_layer(out)
        # Use last time-step output for classification
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.softmax(out)
        return out
    

class LSTMAttentionModel(nn.Module):
    """
    PyTorch LSTM model with Multi-Head Attention for cryptocurrency price prediction.
    
    Architecture: Data → LSTM → Attention → Classification
    This model enhances LSTM with attention mechanism for better sequence modeling.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden dimension of LSTM layers (default: 64)
        num_layers: Number of LSTM layers (default: 3)
        num_classes: Number of output classes (default: 3)
        dropout: Dropout probability (default: 0.3)
        num_heads: Number of attention heads (default: 8)
        use_positional_encoding: Whether to use positional encoding (default: True)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 3, 
        num_classes: int = 3,
        dropout: float = 0.3, 
        num_heads: int = 8, 
        use_positional_encoding: bool = True,
        output_mode: str = 'classification'
    ) -> None:
        super().__init__()
        
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {hidden_size}")
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be an integer >= 1, got {num_layers}")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}")
        if not isinstance(dropout, float) or not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be a float between 0.0 and 1.0, got {dropout}")
        if output_mode not in ['classification', 'regression']:
            raise ValueError(f"output_mode must be 'classification' or 'regression', got {output_mode}")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding
        self.output_mode = output_mode
        
        # Dynamically construct LSTM layers according to num_layers.
        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTM(in_dim, hidden_size, num_layers=1, batch_first=True))
        
        # Dynamically construct dropout layers for each LSTM layer (except last, which uses DROPOUT_FINAL_LAYER if available)
        self.dropout_layers = nn.ModuleList()
        for i in range(self.num_layers):
            d = DROPOUT_FINAL_LAYER if (i == self.num_layers - 1) else dropout
            self.dropout_layers.append(nn.Dropout(d))
        
        self.attention_dim = LSTM_ATTENTION_DIM
        
        if self.attention_dim % num_heads != 0:
            # Find largest divisor of attention_dim that is <= num_heads
            for h in range(min(num_heads, self.attention_dim), 0, -1):
                if self.attention_dim % h == 0:
                    num_heads = h
                    break
            log_warn(f"Adjusted num_heads to {num_heads} to be compatible with attention_dim {self.attention_dim}")
        
        # Projection layer từ LSTM output (hidden_size) sang attention dimension
        # Giải quyết dimension mismatch: LSTM output có hidden_size dim, nhưng attention cần attention_dim dim
        self.lstm_to_attention_proj = nn.Linear(self.hidden_size, self.attention_dim)
        
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(self.attention_dim)
        
        self.multihead_attention = MultiHeadAttention(
            d_model=self.attention_dim, 
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.feed_forward = FeedForward(
            d_model=self.attention_dim,
            d_ff=self.attention_dim * 2,
            dropout=dropout
        )
        
        self.layer_norm1 = nn.LayerNorm(self.attention_dim)
        self.layer_norm2 = nn.LayerNorm(self.attention_dim)
        
        self.attention_pooling = nn.Sequential(
            nn.Linear(self.attention_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_dim, CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_FINAL_LAYER),
            nn.Linear(CLASSIFIER_HIDDEN_DIM, num_classes),
            nn.Softmax(dim=1)
        )
        
        log_model(f"LSTM-Attention model initialized with {num_heads} heads and {self.attention_dim}D attention")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM-Attention model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output probabilities of shape (batch_size, num_classes)
        """
        # Process through dynamic LSTM layers
        out = x
        for lstm_layer, dropout_layer in zip(self.lstm_layers, self.dropout_layers):
            out, _ = lstm_layer(out)
            out = dropout_layer(out)
        
        # Project LSTM output (hidden_size) to attention dimension
        # Shape: (batch, seq_len, hidden_size) -> (batch, seq_len, attention_dim)
        out = self.lstm_to_attention_proj(out)
        
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            out = self.pos_encoding(out)
        
        attn_input = out
        attn_output = self.multihead_attention(attn_input, attn_input, attn_input)
        attn_output = self.layer_norm1(attn_output + attn_input)
        
        ff_output = self.feed_forward(attn_output)
        ff_output = self.layer_norm2(ff_output + attn_output)
        
        attention_weights = self.attention_pooling(ff_output)
        pooled_output = torch.sum(ff_output * attention_weights, dim=1)
        
        output = self.classifier(pooled_output)
        
        return output


class CNNLSTMAttentionModel(nn.Module):
    """
    Enhanced model: CNN feature extraction → LSTM sequence modeling → Attention → Classification/Regression.
    
    Pipeline:
    1. CNN 1D feature extraction
    2. LSTM processing  
    3. Multi-Head Attention
    4. Final Classification/Regression
    
    Args:
        input_size: Number of input features
        look_back: Sliding window size (default: 60)
        cnn_features: Number of CNN features (default: 64)
        lstm_hidden: LSTM hidden dimension (default: 32)
        num_layers: Number of LSTM layers (default: 2)
        num_classes: Number of output classes (default: 3)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.3)
        use_attention: Whether to use attention mechanism (default: True)
        use_positional_encoding: Whether to use positional encoding (default: True)
        output_mode: Output mode - 'classification' or 'regression' (default: 'classification')
    """
    
    def __init__(
        self, 
        input_size: int, 
        look_back: int = 60, 
        cnn_features: int = 64,
        lstm_hidden: int = 32, 
        num_layers: int = 2, 
        num_classes: int = 3,
        num_heads: int = 4, 
        dropout: float = 0.3,
        use_attention: bool = True, 
        use_positional_encoding: bool = True,
        output_mode: Literal['classification', 'regression'] = 'classification'
    ) -> None:
        super().__init__()
        
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}")
        if not isinstance(look_back, int) or look_back <= 0:
            raise ValueError(f"look_back must be a positive integer, got {look_back}")
        if not isinstance(cnn_features, int) or cnn_features <= 0:
            raise ValueError(f"cnn_features must be a positive integer, got {cnn_features}")
        if not isinstance(lstm_hidden, int) or lstm_hidden <= 0:
            raise ValueError(f"lstm_hidden must be a positive integer, got {lstm_hidden}")
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be an integer >= 1, got {num_layers}")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}")
        if not isinstance(dropout, float) or not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be a float between 0.0 and 1.0, got {dropout}")
        if output_mode not in ['classification', 'regression']:
            raise ValueError(f"output_mode must be 'classification' or 'regression', got {output_mode}")
        
        self.input_size = input_size
        self.look_back = look_back
        self.cnn_features = cnn_features
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding
        self.output_mode = output_mode
        
        self.cnn_extractor = CNN1DExtractor(
            input_channels=input_size,
            cnn_features=cnn_features,
            dropout=dropout
        )
        
        # Dynamically construct LSTM layers based on num_layers
        # The first layer maps cnn_features -> lstm_hidden
        # The next (num_layers-2) layers keep same hidden size
        # The last layer maps lstm_hidden -> lstm_hidden // 2
        self.lstm_layers = nn.ModuleList()
        if self.num_layers == 1:
            self.lstm_layers.append(nn.LSTM(cnn_features, lstm_hidden // 2, batch_first=True))
        else:
            self.lstm_layers.append(nn.LSTM(cnn_features, lstm_hidden, num_layers=1, batch_first=True))
            for i in range(self.num_layers - 2):
                self.lstm_layers.append(nn.LSTM(lstm_hidden, lstm_hidden, num_layers=1, batch_first=True))
            self.lstm_layers.append(nn.LSTM(lstm_hidden, lstm_hidden // 2, num_layers=1, batch_first=True))
        
        # Create dropout layers for each LSTM layer
        self.dropout_layers = nn.ModuleList()
        for i in range(self.num_layers):
            d = DROPOUT_FINAL_LAYER if (i == self.num_layers - 1) else dropout
            self.dropout_layers.append(nn.Dropout(d))
        
        if use_attention:
            self.attention_dim = lstm_hidden // 2
            
            if self.attention_dim % num_heads != 0:
                num_heads = min(num_heads, self.attention_dim)
                log_warn(f"Adjusted num_heads to {num_heads} for compatibility")
            
            if use_positional_encoding:
                self.pos_encoding = PositionalEncoding(self.attention_dim, max_seq_length=look_back)
            
            self.multihead_attention = MultiHeadAttention(
                d_model=self.attention_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            
            self.feed_forward = FeedForward(
                d_model=self.attention_dim,
                d_ff=self.attention_dim * 2,
                dropout=dropout
            )
            
            self.layer_norm1 = nn.LayerNorm(self.attention_dim)
            self.layer_norm2 = nn.LayerNorm(self.attention_dim)
            
            self.attention_pooling = nn.Sequential(
                nn.Linear(self.attention_dim, 1),
                nn.Softmax(dim=1)
            )
            
            final_features = self.attention_dim
        else:
            final_features = lstm_hidden // 2
        
        if output_mode == 'classification':
            self.classifier = nn.Sequential(
                nn.Linear(final_features, final_features//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(final_features//2, num_classes),
                nn.Softmax(dim=1)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(final_features, final_features//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(final_features//2, 1),
                nn.Tanh()
            )
        
        log_model("CNN-LSTM-Attention model initialized:")
        log_model(f"  - Look back: {look_back}, CNN features: {cnn_features}")
        log_model(f"  - LSTM hidden: {lstm_hidden}, Attention: {use_attention}")
        log_model(f"  - Output mode: {output_mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN-LSTM-Attention model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            For classification: probabilities of shape (batch_size, num_classes)
            For regression: predictions of shape (batch_size, 1)
        """
        cnn_features = self.cnn_extractor(x)
        
        # Process through dynamic LSTM layers
        out = cnn_features
        for lstm_layer, dropout_layer in zip(self.lstm_layers, self.dropout_layers):
            out, _ = lstm_layer(out)
            out = dropout_layer(out)
        
        if self.use_attention:
            if self.use_positional_encoding and hasattr(self, 'pos_encoding'):
                out = self.pos_encoding(out)
            
            attn_input = out
            attn_output = self.multihead_attention(attn_input, attn_input, attn_input)
            attn_output = self.layer_norm1(attn_output + attn_input)
            
            ff_output = self.feed_forward(attn_output)
            ff_output = self.layer_norm2(ff_output + attn_output)
            
            attention_weights = self.attention_pooling(ff_output)
            pooled_output = torch.sum(ff_output * attention_weights, dim=1)
        else:
            pooled_output = out[:, -1, :]
        
        if self.output_mode == 'classification':
            output = self.classifier(pooled_output)
        else:
            output = self.regressor(pooled_output)
        
        return output

