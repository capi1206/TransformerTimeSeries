import torch.nn as nn


from .units import ClassificationHead, Encoder, PositionalEncoding
#from .encoders import PositionalEncoding, Encoder

class Transformer(nn.Module):

    def __init__(self,device, d_model=100, n_head=4, max_len=500, seq_len=30,
                 ffn_hidden=128, n_layers=4, drop_prob=0.1, details =False):
        super().__init__()
        self.device = device
        self.details = details
        self.encoder_input_layer = nn.Linear(
            in_features=73,
            out_features=d_model
            )

        #self.pos_emb = PositionalEncoding( max_seq_len=max_len,batch_first=False, d_model=d_model, dropout=0.1) #try different values of drupout?
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               details=details,
                               device=device)
        self.class_head = ClassificationHead(seq_len=seq_len,d_model=d_model,details=details,n_classes=1)
    
    def reset_pe(self):
        self.pos_emb.reset_parameters()
        
    def forward(self, src ):
        if self.details: print('before input layer: '+ str(src.size()) )
        src= self.encoder_input_layer(src)
        if self.details: print('after input layer: '+ str(src.size()) )
        # src= self.pos_emb(src)
        # if self.details: print('after pos_emb: '+ str(src.size()) )
        enc_src = self.encoder(src)
        cls_res = self.class_head(enc_src)
        if self.details: print('after cls_res: '+ str(cls_res.size()) )
        return cls_res