import torch
from torch.utils import data
import numpy as np
import argparse
import os

from model_config import DOSconfig, BSconfig, TISEconfig
from ssl_mode import SimCLR, BYOL
from models import Net, Pred
from load_data import get_TISE_datasets, get_PhC_datasets, get_unlabeled_datasets
from loss_func import LogLoss, FracLoss, DOSLoss
from training_modules import *

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def main(args):
    # Define device and reset seed
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    reset_seeds(args.seed)
    reset_seeds(args.ftseed)

    # Create path for pretrained models
    if not os.path.exists('./pretrained_models'):
        os.makedirs('./pretrained_models')

    ## Load default model config
    if args.predict == 'DOS':
        cfg = DOSconfig()
    elif 'band' in args.predict:
        cfg = BSconfig()
    elif 'eig' in args.predict:
        cfg = TISEconfig()

    # Define strings for logging and saving model
    if args.train == 'sl':
        savestring = args.iden+'FS'
        FSstring = 'FS_bs{}'.format(args.batchsize)
    elif args.train == 'tl':
        savestring = args.iden+'tl'
        ptstring = 'ptbs{}_ptlr{}'.format(args.batchsize,args.learning_rate)
    elif args.train == 'sibcl':
        savestring = args.iden+'sibcl_with_{}'.format(args.ssl_mode)
        ptstring = 'ptbs{}_bscl{}_temp{}_ptlr{}'.format(args.batchsize,
                    args.batchsize_cl,args.temperature,args.learning_rate)
    elif args.train == 'ssl':
        savestring = args.iden+'ssl'
        ptstring = 'ssl_bscl{}_temp{}_ptlr{}'.format(args.batchsize_cl,
                    args.temperature,args.learning_rate)

    # Retrieve Data
    print("Retrieving data.. ")
    if 'eig' in args.predict:
        src_Ds, tgt_train_Ds, tgt_test_Ds = get_TISE_datasets(args)
    else:
        src_Ds, tgt_train_Ds, tgt_test_Ds = get_PhC_datasets(args)

    tgt_train_Dl = data.DataLoader(tgt_train_Ds, batch_size = args.batchsize,
                    shuffle = True)
    tgt_test_Dl = data.DataLoader(tgt_test_Ds, batch_size=200)
    print("Data retrieved!")

    ## Define Loss criterion
    if cfg.ft_lossfn == 'MSE':
        ft_criterion = torch.nn.MSELoss()
    elif cfg.ft_lossfn == 'L1':
        ft_criterion = torch.nn.L1Loss()
    if cfg.pt_lossfn == 'MSE':
        pt_criterion = torch.nn.MSELoss()
    elif cfg.pt_lossfn == 'log':
        pt_criterion = LogLoss()

    ## Training modules
    if args.train == 'sl':
        # No pre-training needed
        reset_seeds(args.ftseed)
        model = Net(Lv=cfg.Lv,ks=cfg.ks,Lvp=cfg.Lvp,nbands=args.nbands,
                ndos=400,neval=args.neval,predict=args.predict,
                dim = args.ndim).to(device)
        finetune_model(args,model,args.learning_rate_ft, ft_criterion,
                    tgt_train_Dl, tgt_test_Dl, device, savestring, FSstring, 0)

    elif args.train == 'tl':
        eplist = [40, 100, 200] # epochs to save model and finetune from

        if not args.no_pretrain: # tl pre-training
            train_sDl = data.DataLoader(src_Ds, batch_size = args.batchsize)
            model = Net(Lv=cfg.Lv,ks=cfg.ks,Lvp=cfg.Lvp,nbands=args.nbands,
                    ndos=400,neval=args.neval,predict=args.predict,
                    dim = args.ndim).to(device)
            tl_pretrain(args, model, args.learning_rate, pt_criterion,
                    train_sDl, device, savestring, ptstring, eplist)

        if not args.no_finetune: # Fine-tuning
            reset_seeds(args.ftseed)
            for PTEP in eplist:
                model = Net(Lv=cfg.Lv,ks=cfg.ks,Lvp=cfg.Lvp,nbands=args.nbands,
                        ndos=400,neval=args.neval,predict=args.predict,
                        dim = args.ndim).to(device)
                finetune_model(args,model,args.learning_rate_ft, ft_criterion,
                    tgt_train_Dl, tgt_test_Dl, device, savestring, ptstring, PTEP)

    elif args.train == 'sibcl' or args.train == 'ssl': # sibcl, ssl
        eplist = [40, 100, 200, 300, 400] # epochs to save model and finetune from
        # eplist = [1, 2] # for debugging

        if not args.no_pretrain:
        # Load new unlabeled datasets for sibcl model training
            ntarget = 20000 if args.train == 'sibcl' else 20480 # ssl we use ~20k
            print("Retrieving unlabelled target data set..")
            ul_tgt_Ds = get_unlabeled_datasets(args)

            train_Ptxt_tDl = data.DataLoader(ul_tgt_Ds,
                                batch_size=args.batchsize_cl,drop_last = True)

            if args.train == 'sibcl':
                train_Ptxt_sDl = data.DataLoader(src_Ds,
                                        batch_size = int(args.batchsize_cl/2),
                                        drop_last = True)
                train_sDl = data.DataLoader(src_Ds, batch_size = args.batchsize) # this is predictor training set
                Prednet = Pred(predict=args.predict,latent_dim=cfg.latent_dim,
                                Lvp=cfg.Lvp,nbands=args.nbands,
                                neval=args.neval).to(device)
            else:
                train_Ptxt_sDl = None
                train_sDl = None
                Prednet = None

            print("Unlabeled data retrieved!")
            print("Check; if SSL should print none:", train_Ptxt_sDl)

            batchsizeptxt = int(args.batchsize_cl/2*3) if train_Ptxt_sDl \
                                is not None else args.batchsize_cl

            if args.ssl_mode == 'simclr':
                Encnet = SimCLR(cfg.Lv,cfg.Lvpj,cfg.ks,args.ndim,device,
                batchsizeptxt, args.temperature,use_projector=cfg.use_projector,
                bnorm = cfg.proj_use_bnorm, depth = cfg.proj_depth).to(device)
            elif args.ssl_mode == 'byol':
                Encnet = BYOL(cfg.Lv,cfg.Lvpj,cfg.ks,args.ndim,
                use_projector=cfg.use_projector,bnorm = cfg.proj_use_bnorm,
                depth = cfg.proj_depth).to(device)

            sibcl_pretrain(args, Encnet, args.learning_rate,
                            pt_criterion, train_Ptxt_tDl, device, savestring,
                            ptstring, eplist, Prednet, train_Ptxt_sDl, train_sDl)

        if not args.no_finetune:
            reset_seeds(args.ftseed)
            for PTEP in eplist:
                # we can load this since we kept the same state dict keys
                # by introducing new model classes for Enc & Pred
                model = Net(Lv=cfg.Lv,ks=cfg.ks,Lvp=cfg.Lvp,nbands=args.nbands,
                        ndos=400,neval=args.neval,predict=args.predict,
                        dim = args.ndim).to(device)

                finetune_model(args,model,args.learning_rate_ft,ft_criterion,
                tgt_train_Dl, tgt_test_Dl, device, savestring, ptstring, PTEP)

    else:
        raise ValueError("Invalid train mode. Either sl, tl, sibcl, ssl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Data and task parameters
    parser.add_argument('--path_to_h5', type = str, help="Directory with h5 data files", default = '/home/gridsan/cloh/h5data')
    parser.add_argument('--device',type=int, help="GPU device number. Default = 0 ", default = 0)
    parser.add_argument('--iden',type=str, required=True, help="REQUIRED: Model and logging identifier for current run")
    parser.add_argument('--predict', type = str, required=True, help = 'REQUIRED: DOS, bandstructures, oneband, eigval or eigvec')
    parser.add_argument('--train', type = str, help = 'tl, sl, sibcl or ssl, default = sibcl', default = 'sibcl')
    parser.add_argument('--log_to_tensorboard',action='store_true')

    # PhC data parameters
    parser.add_argument('--nbands',type=int, help='if predict bands, specify num of bands, must be <= 10. default = 6', default=6)
    parser.add_argument('--srcband',type=int, help='if predict oneband, specify source band index, default = 0', default=0)
    parser.add_argument('--tgtband',type=int, help='if predict oneband, specify target band index, default = 0', default=0)

    # TISE data parameters
    parser.add_argument('--neval',type=int, help='if predict eigval or eigvec, specify num of eigvals/eigvecs counting from ground state, default = 1', default=1)
    parser.add_argument('--ndim',type=int, help='if predict eigval or eigvec, specify dimensions: 2 or 3, default = 2', default=2)
    parser.add_argument('--tisesource',type=str, help='specify source dataset, lr (lowres) or qho, default = lr', default='lr')

    # Training parameters
    parser.add_argument('--nsam', type=int, help='no. of labelled target training samples, default = 100',default=100)
    parser.add_argument('--nsource', type=int, help='no. of source training samples',default=10000)
    parser.add_argument('--batchsize', type=int, help='batchsize for predictor (both pretraining and finetuning). Default = 64',default=64)
    parser.add_argument('--learning_rate',type=float, help='pretraining learning rate', default=1e-4)
    parser.add_argument('--learning_rate_ft',type=float, help='finetuning learning rate', default=1e-3)
    parser.add_argument('--finetune_epochs',type=int, help='total no. of epochs to finetune, default = 100', default=100)
    parser.add_argument('--seed',type=int, help='Random seed', default=1)
    parser.add_argument('--ftseed',type=int, help='Random ft seed', default=1)
    parser.add_argument('--pt_scheduler', type=str,help = 'to add scheduler for pt. cosine or cosine-all or none', default='none')
    parser.add_argument('--weight_decay',type=float, help='weight decay', default=1e-5)
    parser.add_argument('--no_scheduler',action='store_true',help="Set this flag to not use schedule")
    parser.add_argument('--no_pretrain',action='store_true',help="Set this flag to not pretrain")
    parser.add_argument('--no_finetune',action='store_true',help="Set this flag to not finetune")
    parser.add_argument('--freeze_enc_ft',action='store_true',help="For sibcl, tl or ssl, set this flag to freeze encoder during finetuning")

    # CL specific parameters
    parser.add_argument('--ssl_mode', type = str, help = 'if sibcl or ssl, specify if simclr or byol, default = simclr', default = 'simclr')
    parser.add_argument('--batchsize_cl', type=int, help='specify batchsize for CL if sibcl or ssl, default = 512', default=512)
    parser.add_argument('--temperature',type=float, help='specify temperature for CL loss function', default=0.1)
    parser.add_argument('--train_ratio',type=int, help='specify no. of epoch for sibcl training before we train predictor once, default = 1', default=1)

    # Augmentation parameters
    parser.add_argument('--translate_pbc', action='store_true', help = 'to randomly translate input image or not (to take care of PBC). uses rolling translation')
    parser.add_argument('--pg_uniform', action='store_true', help = 'to uniformly sample point group operations (rotations and flips) ')
    parser.add_argument('--flip', action='store_true', help = 'to randomly apply mirror flip horizontally or vertically')
    parser.add_argument('--rotate', action='store_true', help = 'to randomly rotate image, one among 4-fold.')
    parser.add_argument('--scale', action='store_true', help = 'to randomly scale input')
    parser.add_argument('--stoch_aug_p', type=float, help = 'stochastic sampling parameter, default = 0.5', default =0.5)

    # Visualize model performance
    parser.add_argument('--minsaveloss',type=float, help='specify min fractional loss in which we save ft model', default=0.05)
    parser.add_argument('--saveplots',action='store_true',help="Set this flag to save prediction plots")

    args = parser.parse_args()

    main(args)
