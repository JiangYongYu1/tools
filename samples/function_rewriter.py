import torch.nn.functional as F
import torch
import types

def forward(self, x):
    """
    x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        the mel spectrogram of the audio
    """
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
    x = (x + self.positional_embedding).to(x.dtype)

    x = self.fc_sub_mean(x)
    for block in self.blocks:
        x = block(x)

    x = self.ln_post(x)
    return x

def warp_whispermodel(whispermodel):
    dim = whispermodel.encoder.conv2.out_channels
    whispermodel.encoder.fc_sub_mean = torch.nn.Linear(dim, dim, bias=False).to(whispermodel.encoder.conv2.weight.data)
    whispermodel.encoder.fc_sub_mean.weight.data = (torch.diag(torch.ones(dim)) - torch.ones(dim, dim) / dim).to(whispermodel.encoder.conv2.weight.data)
    whispermodel.encoder.forward = types.MethodType(forward, whispermodel.encoder)

def load_model(ckpt_dir, device):
    snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    whispermodel = whisper.load_model("small").to(device)
    warp_whispermodel(whispermodel)
    text_tokenizer = Tokenizer(ckpt_dir)
    fabric = L.Fabric(devices=1, strategy="auto")
    config = Config.from_file(ckpt_dir + "/model_config.yaml")
    config.post_adapter = False

    with fabric.init_module(empty_init=False):
        model = GPT(config)

    model = fabric.setup(model)
    state_dict = lazy_load(ckpt_dir + "/lit_model.pth")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    return fabric, model, text_tokenizer, snacmodel, whispermodel
