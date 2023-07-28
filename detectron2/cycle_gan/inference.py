import torch
import os
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

from maskrcnn_benchmark.cycle_gan.model import FPN_RESNET_v2

def inference_cycle(
        model,
        nucleus_cycle_dataloader,
        gland_cycle_dataloader,
        output_dir,
        cycle_weight
):
    G_AB = FPN_RESNET_v2(type='FPN50', BN=True, norm_eval=False)
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    G_AB.load_state_dict(torch.load(cycle_weight), strict=True)
    if cuda:
        G_AB.cuda()
        model.cuda()
        
    os.makedirs(os.path.join(output_dir, 'nucleus'), exist_ok=True)
    
    if gland_cycle_dataloader is not None:
        os.makedirs(os.path.join(output_dir, 'gland'), exist_ok=True)
        for batches_done, (item, _, _, _) in enumerate(gland_cycle_dataloader):
            item=item.tensors
            real_A = Variable(item[:,:3,:,:].cuda().type(Tensor))
            real_B = Variable(item[:,3:,:,:].cuda().type(Tensor))
            G_AB.eval()
            model.eval()
            fake_B = G_AB(real_A)
            fake_A = model(real_B, None, None, 'cycle').type(Tensor)
            # Arange images along x-axis
            real_A = make_grid(real_A, nrow=4, normalize=True)
            real_B = make_grid(real_B, nrow=4, normalize=True)
            fake_A = make_grid(fake_A, nrow=4, normalize=True)
            fake_B = make_grid(fake_B, nrow=4, normalize=True)
            # Arange images along y-axis
            image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
            save_image(image_grid, os.path.join(output_dir, 'gland', str(batches_done)+'_val.png'), normalize=False)
    
    for batches_done, (item, _, _, _) in enumerate(nucleus_cycle_dataloader):
        item=item.tensors
        real_A = Variable(item[:,:3,:,:].cuda().type(Tensor))
        real_B = Variable(item[:,3:,:,:].cuda().type(Tensor))
        G_AB.eval()
        model.eval()
        fake_B = G_AB(real_A)
        fake_A = model(real_B, None, None, 'cycle').type(Tensor)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=4, normalize=True)
        real_B = make_grid(real_B, nrow=4, normalize=True)
        fake_A = make_grid(fake_A, nrow=4, normalize=True)
        fake_B = make_grid(fake_B, nrow=4, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, os.path.join(output_dir, 'nucleus', str(batches_done)+'_val.png'), normalize=False)
    