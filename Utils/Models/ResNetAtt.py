import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True,
                 padding_layer=torch.nn.ReflectionPad1d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        
        if dilation > 1:
            ka = ka * dilation
            kb = kb * dilation
        
        self.net = torch.nn.Sequential(padding_layer((ka,kb)),
                                       torch.nn.Conv1d(in_channels, out_channels, kernel_size, 
                                                       dilation=dilation, bias=bias)
                                       )

    def forward(self, x):
        return self.net(x)

class DilatedResUnit(nn.Module):       
    def __init__(self, c_in, c_out, k=3, dilation=1, dp_rate=0.1):
        super(DilatedResUnit,self).__init__()
        
        self.convdilatedlayers = nn.Sequential(Conv1dSame(in_channels=c_in, out_channels=c_out, 
                                                          kernel_size=k, dilation=dilation),
                                               nn.ReLU(),
                                               nn.BatchNorm1d(c_out),
                                               nn.Dropout(dp_rate)
                                               ) 
    def forward(self,x):
        new_x = self.convdilatedlayers(x)
        return x + new_x

class ResUnit(nn.Module):
    def __init__(self, c_in, c_out, k=3, dp_rate=0.1):
        super(ResUnit,self).__init__()
        self.convlayers = nn.Sequential(Conv1dSame(in_channels=c_in, out_channels=c_out, kernel_size=k),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(c_out),
                                        nn.Dropout(dp_rate)
                                        ) 
    def forward(self,x):
        new_x = self.convlayers(x)
        return x + new_x
    
class DilatedBlock(nn.Module):  
    def __init__(self, c_in=24, c_out=24, kernel_size=3, dilation_list=[1, 2, 4, 8], dp_rate=0.1):
        super(DilatedBlock,self).__init__()
 
        layers = []
        
        for i, dilation in enumerate(dilation_list):
            layers.append(DilatedResUnit(c_in[i], c_out, kernel_size, dilation, dp_rate))
            
        self.network = torch.nn.Sequential(*layers)
            
    def forward(self,x):
        x = self.network(x)
        return x
    
class AttentionBlock(nn.Module):  
    def __init__(self, c_in=24, c_out=24, kernel_size=3, dp_rate=0.1):
        super(AttentionBlock,self).__init__()
        
        self.ResUnit1 = ResUnit(c_in, c_out)
        self.ResUnit2 = ResUnit(c_in, c_out)
        self.ResUnit3 = ResUnit(c_in, c_out)
        self.ResUnit4 = ResUnit(c_in, c_out)
        
        self.max_pool1 = nn.MaxPool1d(kernel_size, return_indices=True)
        self.max_pool2 = nn.MaxPool1d(kernel_size, return_indices=True)
        
        self.max_unpool1 = nn.MaxUnpool1d(kernel_size)
        self.max_unpool2 = nn.MaxUnpool1d(kernel_size)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_input):
        x = self.ResUnit1(x_input)
        S1 = x.size()
        new_x, indices1 = self.max_pool1(x)
        x = self.ResUnit2(new_x)
        S2 = x.size()
        x, indices2 = self.max_pool2(x)
        x = self.ResUnit3(x)
        x = self.max_unpool1(x, indices2, output_size=S2)
        x = self.ResUnit4(x + new_x)
        x = self.max_unpool2(x, indices1, output_size=S1)
        x = self.sigmoid(x + x_input)

        x = torch.mul(x_input, x)
        
        return x_input + x
    
class ResNetAtt(nn.Module):
    """
    Model from "Residential Appliance Detection Using Attentionbased Deep Convolutional Neural Network" paper.
    """
    def __init__(self, c_in=1, n_dilated_block=6, n_attention_block=2, in_model_channel=24, 
                 kernel_size=8, dilation_list=[1, 2, 4, 8], d_ff=256, dp_rate=0.2, nb_class=2):
        super(ResNetAtt, self).__init__()
        
        layers = []
        
        for i in range(n_dilated_block):
            in_channel = []
            if i==0:
                in_channel.append(c_in)
                in_channel = in_channel + [in_model_channel for i in range(len(dilation_list)-1)]
            else:
                in_channel = [in_model_channel for i in range(len(dilation_list))]
            
            layers.append(DilatedBlock(in_channel, in_model_channel, kernel_size, dilation_list, dp_rate))
            
        for i in range(n_attention_block):
            layers.append(AttentionBlock(in_model_channel, in_model_channel, kernel_size, dp_rate))
            
        self.blocks_network = torch.nn.Sequential(*layers)
        
        self.linear1 = nn.LazyLinear(d_ff)
        self.linear2 = nn.Linear(d_ff, nb_class)

    def forward(self, x) -> torch.Tensor:
        x = self.blocks_network(x)
        x = F.relu(self.linear1(torch.flatten(x, 1)))

        return self.linear2(x)