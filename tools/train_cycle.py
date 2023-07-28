#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer, build_cycle_optimizer
from detectron2.utils.events import EventStorage

from detectron2.data.detection_utils import get_coco_target
from detectron2.data.dataset_mapper import CycleDatasetMapper

from detectron2.cycle_gan.model import DLinkNet50Generator, FPN_RESNET, FPN_RESNET_v2, FPN_RESNET_Rethinking, init_weight

logger = logging.getLogger("detectron2")

def get_nucleus_dicts(cfg):
    root = os.path.join(cfg.CYCLE.DATASET_ROOT, cfg.CYCLE.DATASET_A_NAME, cfg.CYCLE.DATASET_A_SCALE)
    mode = 'train'
    random.seed(10)
    img_A_path = os.path.join(root, "%s/A" % mode)
    files_A = sorted(glob.glob(img_A_path + "/*.*"))
    dataset_dicts = []
    for idx, filename in enumerate(files_A):
        record = {}
        image_A = Image.open(filename)
        width, height = image_A.size
        
        dir_name = dirname(dirname(dirname(filename)))
        npy_filename = os.path.join(dir_name.replace('40x','cell_mask'), filename.split('/')[-1].split('_')[0]+'.npy')
        
        anno_result = get_coco_target(file_path=npy_filename, img_index=idx, h_index=int(filename[:-4].split('/')[-1].split('_')[2]), w_index=int(filename[:-4].split('/')[-1].split('_')[1]), size=image_A.size)
        
        if anno_result:
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            record["annotations"] = anno_result
            record["annotations"] = anno_result
            dataset_dicts.append(record)
    return dataset_dicts


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    print('train annotation file path:', data_loader.dataset.ann_file)
    print('test annotation file path:', data_loader_val.dataset.ann_file)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'eval_result'),exist_ok=True)
    ########################for cycle#####################
    def r1_reg(d_out, x_in):
        # zero-centered gradient penalty for real images
        batch_size = x_in.size(0)
        
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg
    
    def sample_images(batches_done, G_AB, G_BA, D_A, D_B, val_dataloader, test_dataloader):
        """Saves a generated sample from the test set"""
        (item,_,_,idx) = next(iter(val_dataloader))
        item=item.tensors
        real_A = Variable(item[:,:3,:,:].cuda().type(Tensor))
        real_B = Variable(item[:,3:,:,:].cuda().type(Tensor))
        G_AB.eval()
        G_BA.eval()
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B, None, None, 'cycle').type(Tensor)
    
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=4, normalize=True)
        real_B = make_grid(real_B, nrow=4, normalize=True)
        fake_A = make_grid(fake_A, nrow=4, normalize=True)
        fake_B = make_grid(fake_B, nrow=4, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, os.path.join(image_save_dir, str(batches_done)+'_val.png'), normalize=False)
        
        G_AB.eval()
        G_BA.eval()
        (item,_,_,idx) = next(iter(test_dataloader))
        item=item.tensors
        real_A = Variable(item[:,:3,:,:].cuda().type(Tensor))
        real_B = Variable(item[:,3:,:,:].cuda().type(Tensor))
        fake_A = G_BA(real_B, None, None, 'cycle').type(Tensor)
        fake_B = G_AB(real_A)
        real_A = make_grid(real_A, nrow=4, normalize=True)
        real_B = make_grid(real_B, nrow=4, normalize=True)
        fake_A = make_grid(fake_A, nrow=4, normalize=True)
        fake_B = make_grid(fake_B, nrow=4, normalize=True)
        
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, os.path.join(image_save_dir, str(batches_done)+'_test.png'), normalize=False)

    class GANLoss(nn.Module):
        def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
            super(GANLoss, self).__init__()
            self.register_buffer('real_label', torch.tensor(target_real_label))
            self.register_buffer('fake_label', torch.tensor(target_fake_label))
            if use_lsgan:
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.BCELoss()
    
        def get_target_tensor(self, input, target_is_real):
            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            return target_tensor.expand_as(input)
    
        def __call__(self, input, target_is_real):
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)

    critic=cfg.CYCLE.CRITIC_DISCRIMINATOR
    lambda_A=cfg.CYCLE.LABMDA_A
    lambda_B=cfg.CYCLE.LABMDA_B
    lambda_D_A=cfg.CYCLE.LABMDA_D_A
    lambda_D_B=cfg.CYCLE.LABMDA_D_B
    lambda_gp=cfg.CYCLE.LABMDA_GP
    lambda_cyc=cfg.CYCLE.LABMDA_CYCLE
    identity=cfg.CYCLE.IDENTITY
    lambda_id=cfg.CYCLE.LABMDA_IDENTITY
    lambda_mrcnn_nucleus=cfg.CYCLE.LABMDA_MRCNN_NUCLEUS
    lambda_mrcnn_gland=cfg.CYCLE.LABMDA_MRCNN_GLAND
    sample_interval=cfg.CYCLE.SAMPLING_INTERVAL

    dis_critic=cfg.CYCLE.CRITIC_DISCRIMINATOR
    cycle_critic=cfg.CYCLE.CRITIC_CYCLE
    cycle_weight=cfg.CYCLE.CRITIC_WEIGHT
    checkpoint_interval=cfg.CYCLE.CHECKPOINT_INTERVAL
    
    image_save_dir = os.path.join(cfg.OUTPUT_DIR, 'cycle_result/imgs')
    model_save_dir = os.path.join(cfg.OUTPUT_DIR, 'cycle_result/models')
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    criterion_GAN = GANLoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    
    cuda = torch.cuda.is_available()
    num_gpu=torch.cuda.device_count()

    input_shape = (3, cfg.CYCLE.INPUT_SIZE, cfg.CYCLE.INPUT_SIZE)
    G_AB = FPN_RESNET_v2(type=cfg.CYCLE.FPNTYPE, BN=True, norm_eval=False)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)
    
    if cycle_weight is not None:
        G_AB.load_state_dict(torch.load(cycle_weight), strict=True)
        model.load_state_dict(torch.load(cycle_weight.replace('G_AB','model')), strict=True)
        D_A.load_state_dict(torch.load(cycle_weight.replace('G_AB','D_A')), strict=True)
        D_B.load_state_dict(torch.load(cycle_weight.replace('G_AB','D_B')), strict=True)
    else:
        print('cycle_weight is None!!!')
    if cuda:
        G_AB = G_AB.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        model = model.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
    num_gpu=torch.cuda.device_count()
    if num_gpu > 1:
        G_AB = nn.DataParallel(G_AB)
        model = nn.DataParallel(model)
        D_A = nn.DataParallel(D_A)
        D_B = nn.DataParallel(D_B)
    
    optimizer, optimizer_D_A, optimizer_D_B = build_cycle_optimizer(cfg, model, G_AB, D_A, D_B)
    scheduler = build_lr_scheduler(cfg, optimizer)
    scheduler_D_A = build_lr_scheduler(cfg, optimizer_D_A)
    scheduler_D_B = build_lr_scheduler(cfg, optimizer_D_B)
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    cycle_train_dataset=ImageDataset(os.path.join(cfg.CYCLE.DATASET_ROOT, cfg.CYCLE.DATASET_A_NAME, cfg.CYCLE.DATASET_B_SCALE),
                root_B=os.path.join(cfg.CYCLE.DATASET_ROOT, cfg.CYCLE.DATASET_B_NAME, cfg.CYCLE.DATASET_B_SCALE),
                unaligned=False, multiscale=False, target='nucleus'),
    DatasetCatalog.register("nucleus_train", cycle_train_dataset)
    MetadataCatalog.get("nucleus_train").set(thing_classes=["nucleus"])
    CycleDatasetMapper=CycleDatasetMapper(cfg, True)

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg, mapper=CycleDatasetMapper)
    logger.info("Starting training from iteration {}".format(start_iter))
    
    target_type='nucleus'
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            real_A = Variable(model.preprocess_image(data).tensor.cuda().type(Tensor))
            real_B = Variable(model.preprocess_image(data, mode='B').tensor.cuda().type(Tensor))

            G_AB.train()
            model.train()
            
            if (iteration+1) % dis_critic == 0:
                optimizer.zero_grad()
        
                # GAN loss
                fake_B = G_AB(real_A)
                loss_GAN_AB = criterion_GAN(D_B(fake_B), True)
                fake_A = model(real_B, None, None, 'cycle')
                loss_GAN_BA = criterion_GAN(D_A(fake_A), True)
        
                loss_GAN = (loss_GAN_AB * lambda_D_B + loss_GAN_BA * lambda_D_A) / 2
        
                # Cycle loss
                if target_type == 'nucleus':
                    recov_A, loss_dict = model(data, input_images=fake_B, mode='both', item='nucleus')
                if target_type == 'gland':
                    recov_A, loss_dict = model(data, input_images=fake_B, targets, mode='both', item='gland')
                
                losses_mrcnn = sum(loss_dict.values())
                assert torch.isfinite(losses_mrcnn).all(), loss_dict
    
                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                loss_cycle_A = criterion_cycle(recov_A, real_A)*lambda_A
                recov_B = G_AB(fake_A)
                loss_cycle_B = criterion_cycle(recov_B, real_B)*lambda_B
        
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        
                # Total loss
                if identity is not None:
                    # Identity loss
                    #loss_id_A = criterion_identity(model(real_A, None, 'cycle'), real_A)
                    loss_id_B = criterion_identity(G_AB(real_B), real_B)
                    #loss_identity = (loss_id_A + loss_id_B) / 2
                    loss_identity = loss_id_B
                    loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity + lambda_mrcnn_nucleus * losses_mrcnn
                else:
                    loss_G = loss_GAN + lambda_cyc * loss_cycle  + lambda_mrcnn_nucleus * losses_mrcnn

                loss_G.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()
                
            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()
            # Real loss
            real_A.requires_grad_()
            D_A_out = D_A(real_A)
            loss_real = criterion_GAN(D_A_out, True)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), False)
            loss_fake.backward()
            # Total loss
            
            if cfg.CYCLE.GP_A:
                loss_reg = r1_reg(D_A_out.view(D_A_out.size(0), -1), real_A)
                
                loss_D_A = (loss_real + loss_reg*lambda_gp) / 2 * lambda_D_A
                loss_D_A.backward(retain_graph=True)
            else:
                loss_D_A = (loss_real) / 2 * lambda_D_A
                loss_D_A.backward(retain_graph=True)

            optimizer_D_A.step()
            
            # -----------------------
            #  Train Discriminator B
            # -----------------------
    
            # Real loss
            real_B.requires_grad_()
            D_B_out = D_B(real_B)
            loss_real = criterion_GAN(D_B_out, True)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), False)
            loss_fake.backward()
            # Total loss
            if cfg.CYCLE.GP_B:
                loss_reg = r1_reg(D_B_out.view(D_B_out.size(0), -1), real_B)
                loss_D_B = (loss_real + loss_reg*lambda_gp) / 2 * lambda_D_B
                loss_D_B.backward(retain_graph=True)
            else:
                loss_D_B = (loss_real) / 2 * lambda_D_B
                loss_D_B.backward(retain_graph=True)

            optimizer_D_B.step()
    
            loss_D = (loss_D_A + loss_D_B / 2)
            
            if iteration % critic == 0:
                prev_time = time.time()
                if identity is not None:
                    sys.stdout.write(
                        "\r[Batch %d/%d] [D loss: %.4f] [G loss: %.4f, adv: %.4f, cycle: %.4f, cycle_A: %.4f, cycle_B: %.4f, identity: %.4f, mrcnn: %.4f]"
                        % (
                            iteration,
                            cfg.SOLVER.MAX_ITER,
                            loss_D.item(),
                            loss_G.item(),
                            loss_GAN.item(),
                            loss_cycle.item(),
                            loss_cycle_A.item(),
                            loss_cycle_B.item(),
                            loss_identity.item(),
                            losses_mrcnn.item(),
                        )
                    )
                else:
                    sys.stdout.write(
                        "\r[Batch %d/%d] [D loss: %.4f] [G loss: %.4f, adv: %.4f, cycle: %.4f, cycle_A: %.4f, cycle_B: %.4f, mrcnn: %.4f]"
                        % (
                            iteration,
                            cfg.SOLVER.MAX_ITER,
                            loss_D.item(),
                            loss_G.item(),
                            loss_GAN.item(),
                            loss_cycle.item(),
                            loss_cycle_A.item(),
                            loss_cycle_B.item(),
                            losses_mrcnn.item(),
                        )
                    )
            if iteration % checkpoint_interval == 0:
                if num_gpu>1:
                # Save model checkpoints
                    torch.save(G_AB.module.state_dict(), os.path.join(model_save_dir, "G_AB_%d.pth"% (iteration)))
                    torch.save(model.module.state_dict(), os.path.join(model_save_dir, "model_%d.pth"% (iteration)))
                    torch.save(D_A.module.state_dict(), os.path.join(model_save_dir, "D_A_%d.pth"% (iteration)))
                    torch.save(D_B.module.state_dict(), os.path.join(model_save_dir, "D_B_%d.pth"% (iteration)))
                else:
                    # Save model checkpoints
                    torch.save(G_AB.state_dict(), os.path.join(model_save_dir, "G_AB_%d.pth"% (iteration)))
                    torch.save(model.state_dict(), os.path.join(model_save_dir, "model_%d.pth"% (iteration)))
                    torch.save(D_A.state_dict(), os.path.join(model_save_dir, "D_A_%d.pth"% (iteration)))
                    torch.save(D_B.state_dict(), os.path.join(model_save_dir, "D_B_%d.pth"% (iteration)))
            
            if (iteration) % sample_interval == 0:
                sample_images(iteration, G_AB, model, D_A, D_B, val_cycle_dataloader, test_cycle_dataloader)

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
