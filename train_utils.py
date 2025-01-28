import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        else:
            raise NotImplementedError("Stateful mode not implemented")

        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(isinstance(elem, tuple) for elem in kernel_size))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class MTSAD(nn.Module):
    def __init__(self, feats, hidden_dim, seq_len):
        super(MTSAD, self).__init__()
        self.name = 'MTSAD'
        self.n_feats = feats
        self.n_window = seq_len
        self.hidden_dim = hidden_dim
        self.conv_lstm = ConvLSTM(1, hidden_dim, (3, 3), 3, batch_first=True, bias=True)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 1, (3, 3), 1, 1),
            nn.Sigmoid()
        )

    def forward(self, g):
        """
        Expects input tensor `g` of shape (batch_size, feats, seq_len).
        """
        z = g.view(g.size(0), 1, 1, self.n_feats, self.n_window)
        _, last_state_list = self.conv_lstm(z)
        h, _ = last_state_list[-1]
        x = self.decoder(h)
        return x.view(-1, self.n_window, self.n_feats)

    
class MTSAD_NoConvLSTM(nn.Module):
    def __init__(self, feats, hidden_dim, seq_len, num_layers=3):
        super(MTSAD_NoConvLSTM, self).__init__()
        self.name = 'MTSAD_NoConvLSTM'
        self.n_feats = feats
        self.n_window = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Encoder: Replace ConvLSTM with standard LSTM
        self.lstm = nn.LSTM(
            input_size=feats,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # Decoder: Use Linear layers instead of ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, feats * seq_len),
            nn.Sigmoid()
        )

    def forward(self, g):
        """
        Expects input tensor `g` of shape (batch_size, feats, seq_len).
        """
        
        # Reshape input for LSTM: (batch_size, seq_len, feats)
#         g = g.permute(0, 2, 1)  # (batch_size, seq_len, feats)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(g)  # h_n: (num_layers, batch, hidden_dim)

        # Use the last hidden state from the top LSTM layer
        last_hidden = h_n[-1]  # (batch, hidden_dim)

        # Decode the hidden state
        x = self.decoder(last_hidden)  # (batch, feats * seq_len)

        # Reshape to (batch, seq_len, feats)
        x = x.view(-1, self.n_window, self.n_feats)

        return x

class MTSAD_NoTransposedConv_Dynamic(nn.Module):
    def __init__(self, feats, hidden_dim, seq_len):
        super(MTSAD_NoTransposedConv_Dynamic, self).__init__()
        self.name = 'MTSAD_NoTransposedConv_Dynamic'
        self.n_feats = feats
        self.n_window = seq_len
        self.hidden_dim = hidden_dim

        # ConvLSTM Encoder
        self.conv_lstm = ConvLSTM(1, hidden_dim, (3, 3), num_layers=3, batch_first=True, bias=True)

        # Temporary dummy input to infer the flattened size
        dummy_input = torch.zeros(1, 1, 1, feats, seq_len)
        _, last_state = self.conv_lstm(dummy_input)
        h, _ = last_state[-1]
        flattened_size = hidden_dim * self.n_feats * self.n_window

        # Decoder
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.n_feats * self.n_window),
            nn.Sigmoid()
        )

    def forward(self, g):
        """
        Expects input tensor `g` of shape (batch_size, feats, seq_len).
        """
        z = g.view(g.size(0), 1, 1, self.n_feats, self.n_window)  # (B, C, H, W, T)
        _, last_state_list = self.conv_lstm(z)
        h, _ = last_state_list[-1]  # h shape: (batch, hidden_dim, H, W)

        # Flatten h and pass through the linear decoder
        x = self.decoder(h)
        x = x.view(-1, self.n_window, self.n_feats)  # (batch, seq_len, feats)
        return x
    
