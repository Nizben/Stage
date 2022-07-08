import os
from tqdm import trange
import torch
from torch._vmap_internals import vmap
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    compute_iou, make_3d_grid, add_key,
)
from src.utils import visualize as vis
from src.training import BaseTrainer
from numpy import sqrt


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    
    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        
        batch_size = points.size(0)

        kwargs = {}
        
        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs, 
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        
        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        c = self.model.encode_inputs(inputs)

        kwargs = {}
        # General points
        logits = self.model.decode(p, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss_i.sum(-1).mean()

        return loss
 
 
class TrainerS(Trainer):
     ''' Computes the score matching paper loss

        '''
     def jacobian_func(self, p, c):
        
        kwargs = {}
        x = p[...,0].unsqueeze(-1); x.requires_grad_(True)
        y = p[...,1].unsqueeze(-1); y.requires_grad_(True)
        z = p[...,2].unsqueeze(-1); z.requires_grad_(True)
        p = torch.cat([x,y,z], -1)
        logits = self.model.decode(p, c, **kwargs).logits

        jacobian_row_0 = torch.cat([torch.autograd.grad(logits[..., 0].sum(), i, retain_graph=True)[0] for i in [x,y,z]], dim = -1)
        jacobian_row_1 = torch.cat([torch.autograd.grad(logits[..., 1].sum(), i, retain_graph=True)[0] for i in [x,y,z]], dim = -1)
        jacobian_row_2 = torch.cat([torch.autograd.grad(logits[..., 2].sum(), i, retain_graph=True)[0] for i in [x,y,z]], dim = -1)
        #print(jacobian_row_0.size())
        jacobian = torch.stack((jacobian_row_0, jacobian_row_1, jacobian_row_2), dim=-1)

        return jacobian, logits

     def metropolis_step(self, points, score, z_t, alpha=10e-4):
        points_k1 = points + (alpha/2)*score + sqrt(alpha)*z_t
        return points_k1

    


     def compute_loss(self, data):
        
        device = self.device
        p = data.get('inputs').to(device)
        closest_points = data.get('points.closest_points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        points = data.get('points').to(device)
        distances = data.get('points.dist').to(device)

        c = self.model.encode_inputs(inputs)
        
        #print(jacobian.size())
        #pdx = torch.autograd.grad(logits[..., 0].sum(), x, create_graph=True)[0]
        #print(pdx.size())
        jacobian_inputs, logits = self.jacobian_func(p, c)
        #print(logits.shape)

        loss_gradient = jacobian_inputs[:, :, 0, 0] + jacobian_inputs[: ,: ,1 ,1] + jacobian_inputs[: ,: ,2 ,2]
        loss_norm = 0.5*torch.norm(logits, dim=-1)**2
        #inputs_curl = torch.stack((jacobian_inputs[:,:,2,1] - jacobian_inputs[:,:,1,2], jacobian_inputs[:,:,0,2]-jacobian_inputs[:,:,2,0], jacobian_inputs[:,:,1,0] - jacobian_inputs[:,:,0,1]), dim=-1)

        jacobian_points, score = self.jacobian_func(points, c)


        points_curl = torch.stack((jacobian_points[:,:,2,1] - jacobian_points[:,:,1,2], jacobian_points[:,:,0,2]-jacobian_points[:,:,2,0], jacobian_points[:,:,1,0] - jacobian_points[:,:,0,1]), dim=-1)
        difference_vector = closest_points - points
        #batched_dot = (difference_vector*logits).sum(dim=-1)
        #max_dot_product = torch.relu(-batched_dot)
        #print(batched_dot.shape) 


        # metropolis loss
        sigma = 10
        z_t = torch.randn_like(score)
        alpha = 1e-3
        points_k1 = self.metropolis_step(points, score, z_t, alpha)


        d_k = distances**2
        d_k1 = torch.cdist(points_k1, inputs).topk(1).values.pow(2).squeeze(-1)
        q_0 = -torch.norm(alpha*score + sqrt(alpha)*z_t, dim=-1)/(2*alpha)
        q_1 = -torch.norm(sqrt(alpha)*z_t, dim=-1)/(2*alpha)
        #print(d_k.shape)
        #print(d_k1.shape)
        #print(q_0.shape)
        #print(q_1.shape)
        terme = torch.exp((d_k-d_k1)/sigma +q_0-q_1)
        ratio = torch.relu(1-terme)
       
        #upper = torch.exp((-1/2*alpha)*torch.norm(alpha*))
        #d_k1 = torch.cat([data['inputs.kdtree'][i].query(x)[0] for i, x in enumerate(points_k1)])

        loss = loss_norm + loss_gradient #+ torch.norm(inputs_curl, dim=-1)
        loss = loss.mean() + ratio.mean()  #+ 0.1*torch.norm(points_curl, dim=-1).mean() + max_dot_product.mean()

        return loss, loss_norm.mean(), loss_gradient.mean(), ratio.mean() #, torch.norm(inputs_curl, dim=-1).mean()
    

     def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss[0].backward()
        self.optimizer.step()

        return loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item()
           