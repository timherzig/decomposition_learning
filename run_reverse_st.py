import torch
from models.up_scaling.reverse_st.upsampling import SwinTransformer3D_up

def main(): 

    x = torch.rand(10,768,8,8,8)
    model = SwinTransformer3D_up()
    x = model.forward(x)

if __name__ == '__main__':
    main()