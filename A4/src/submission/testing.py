import torch
comboed = []

#e_t = dec_hidden * enc_hidden_proj
#where:
#dec_hidden.shape = (b,h) = (32,256)
#enc_hidden_proj.shape = (b, sent_len, h) = (32,46,256)
#target = e_t.shape = (b,sent_len) = 32,46

hid_enc = torch.randn((2,3,6))
z = hid_enc.view()

#dec_hidden = torch.randn((32,256))
#enc_hidden_proj = torch.randn((32,46,256))
#
#dec_hidden = torch.unsqueeze(dec_hidden,2)
#
#e_t = torch.bmm(enc_hidden_proj, dec_hidden)
#
#print(e_t.shape())

#for i in range(5):
#    o_t = torch.rand((5,3))
#    comboed.append(o_t)
    