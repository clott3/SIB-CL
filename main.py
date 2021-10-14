import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import json
import sys

from model_config import DOSconfig, BSconfig, TISEconfig
from ssl_mode import SimCLR, BYOL
from models import Net, Pred
from load_data import get_TISE_datasets, get_PhC_datasets, get_unlabeled_datasets
from loss_func import LogLoss, FracLoss, DOSLoss
from symop_utils import get_aug
from lr_utils import Cosine_Scheduler

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def evaluate_model(args, dataloader, model, device, predmodel = None,
                    return_tensors = False):

    calL1loss = torch.nn.L1Loss()
    startpt = 100  # ignore first 100 points
    calDOSloss = DOSLoss(startpt)
    calfracloss = FracLoss()

    for step, task_set in enumerate(dataloader): # one step
        in_testdata = task_set[0]

        in_testdata = in_testdata.to(device)
        out_testdata = task_set[1].to(device)
        if args.predict == 'DOS': # predict ELdiff
            y_testdata = task_set[2].to(device)
            EL_testdata = task_set[3].to(device)

        if predmodel is None:
            output = model(in_testdata)
        else:
            output = predmodel(model.encoder(in_testdata))

        if args.predict == 'DOS':
            l1loss = calL1loss(output,out_testdata).item()
            ploss = calDOSloss(output+EL_testdata,y_testdata).item() # predict ELdiff
        else:
            l1loss = calL1loss(output,out_testdata).item()
            ploss = calfracloss(output,out_testdata).item()

    if not return_tensors:
        return ploss, l1loss
    else:
        if args.predict == 'DOS':
            return ploss, l1loss, output, out_testdata, EL_testdata
        else:
            return ploss, l1loss, output, out_testdata

def tl_pretrain(args, model, lr, criterion, source_dataloader, device, savestring, ptstring, eplist):
    ''' We pre-train model using the labeled source dataset and save model at the specified epochs in eplist.
        This module is used for tl. '''

    opt = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=args.weight_decay)

    if args.log_to_tensorboard:
        tb_dir = 'runs_'+savestring +'/' + ptstring
        writer = SummaryWriter(tb_dir)
    else:
        if not os.path.exists('./tlogs/'+savestring):
            os.makedirs('./tlogs/'+savestring)
        if os.path.exists('./tlogs/'+savestring+'/' + ptstring+".json"):
            with open('./tlogs/'+savestring+'/' + ptstring+".json") as f:
                tlog = json.load(f)
        else:
            tlog = {}
            tlog['loss'] = {}

    for epoch in range(1,max(eplist)+1):

        model.train()
        epochloss = 0.

        for step, task_set in enumerate(source_dataloader):

            in_data = task_set[0]
            in_data = get_aug(in_data,translate=args.translate_pbc,\
            rotate=args.rotate ,flip=args.flip ,scale=args.scale, \
            p=args.stoch_aug_p, pg_uniform=args.pg_uniform)
            in_data = in_data.to(device)
            out_data = task_set[1].to(device)

            opt.zero_grad()
            loss = criterion(model(in_data), out_data)
            loss.backward()
            epochloss += loss.item()
            opt.step()

        if args.log_to_tensorboard:
            writer.add_scalar("TL source loss vs epoch", epochloss/(step+1), epoch)
        else:
            tlog['loss'][str(epoch)] = epochloss/(step+1)

        if epoch in eplist:
            if not os.path.exists('./pretrained_models/'+savestring):
                os.makedirs('./pretrained_models/'+savestring)

            savemodeldir = './pretrained_models/'+savestring+'/'+'EP{}'.format(epoch)+ptstring
            torch.save({'model_state_dict': model.state_dict(),
                        'ptepoch': epoch},
                        savemodeldir)
    if args.log_to_tensorboard:
        writer.close()
    else:
        logfile = json.dumps(tlog)
        with open('./tlogs/'+savestring+'/' + ptstring+".json", "w") as f:
            f.write(logfile)

def sibcl_pretrain(args, Encmodel, lr, criterion, train_Ptxt_tgtdataloader,
                device, savestring, ptstring, eplist, Predmodel = None,
                train_Ptxt_srcdataloader=None, train_dataloader = None):
    ''' This module is used for both "sibcl" and "ssl".
        We pre-train model using pretext tasks, which is by default contrastive learning (args.pttask).
        If labeled source dataset is available ("sibcl"), we need to input Predmodel, train_Ptxt_srcdataloader, train_dataloader & test_dataloader,
        and the Predictor will be trained every args.train_ratio epochs of training the pretext model. Encoder and Predictor will be saved at specified epochs in eplist.
        Otherwise, this module will execute training of the unlabeled target set and save the Encoder only at specified epochs in eplist. ("ssl")
        The pretext task have separately defined batchsizes hence the need for different dataloaders
        '''
    print("We are pre-training {} using SSL mode {}".format(args.train,args.ssl_mode))

    optED = torch.optim.Adam(Encmodel.parameters(),lr=lr) # We use same learning rate
    if 'cosine' in args.pt_scheduler:
        schedulerED = Cosine_Scheduler(optED, num_epochs = max(eplist), base_lr = lr, \
                                        iter_per_epoch = len(train_Ptxt_tgtdataloader))

    if Predmodel is not None:
        combinedparamEP = list(Encmodel.parameters()) + list(Predmodel.parameters())
        optEP = torch.optim.Adam(combinedparamEP,lr=lr) # We use same learning rate
        if args.pt_scheduler == 'cosine-all':
            schedulerEP = Cosine_Scheduler(optEP, num_epochs = max(eplist), base_lr = lr, \
                                            iter_per_epoch = len(train_dataloader))
    if args.log_to_tensorboard:
        tb_dir = 'runs_'+savestring +'/' + ptstring
        writer = SummaryWriter(tb_dir)
    else:
        if not os.path.exists('./tlogs/'+savestring):
            os.makedirs('./tlogs/'+savestring)
        if os.path.exists("./tlogs/"+savestring+'/' + ptstring+".json"):
            with open("./tlogs/"+savestring+'/' + ptstring+".json") as f:
                tlog = json.load(f)
        else:
            tlog = {}
            tlog['loss'] = {}
            tlog['tl_loss'] = {}

    for epoch in range(1,max(eplist)+1):

        Encmodel.train()
        epochloss = 0.0

        if train_Ptxt_srcdataloader is not None:
            ptxt_dataloader = zip(train_Ptxt_srcdataloader,train_Ptxt_tgtdataloader)
        else:
            ptxt_dataloader = train_Ptxt_tgtdataloader

        for step, load_set in enumerate(ptxt_dataloader): # batchsize of batchsize

            if train_Ptxt_srcdataloader is not None:
                input_set = torch.cat((load_set[0][0],load_set[1][0]),dim=0)
                input_set = input_set[torch.randperm(len(input_set))]
            else:
                input_set = load_set[0]

            input1 = get_aug(input_set,translate=args.translate_pbc,\
            rotate=args.rotate ,flip=args.flip ,scale=args.scale, \
            p=args.stoch_aug_p, pg_uniform=args.pg_uniform).to(device)
            input2 = get_aug(input_set,translate=args.translate_pbc,\
            rotate=args.rotate ,flip=args.flip ,scale=args.scale, \
            p=args.stoch_aug_p, pg_uniform=args.pg_uniform).to(device)

            optED.zero_grad()
            loss = Encmodel(input1,input2)
            loss.backward()
            epochloss += loss.item()

            optED.step() # Update Enc and Dec
            if 'cosine' in args.pt_scheduler:
                schedulerED.step()
        if args.log_to_tensorboard:
            writer.add_scalar('Total Ptxt Loss vs ptepoch', epochloss/(step+1), epoch)
        else:
            tlog['loss'][str(epoch)] = epochloss/(step+1)

        if epoch % 10 == 0:
            print(f"Ptxt epoch {epoch}: {epochloss/(step+1)}")

        if Predmodel is not None and epoch % args.train_ratio == 0: # train predictor model every train_ratio epochs

            Encmodel.train()
            Predmodel.train()
            running_loss = 0.

            for step, task_set in enumerate(train_dataloader):

                in_data = task_set[0]
                in_data = in_data.to(device)
                out_data = task_set[1].to(device)

                optEP.zero_grad()
                loss = criterion(Predmodel(Encmodel.encoder(in_data)), out_data)
                loss.backward()
                running_loss += loss.item()
                optEP.step()

            if args.pt_scheduler == 'cosine-all':
                schedulerEP.step()
            if args.log_to_tensorboard:
                writer.add_scalar('PT: Predictor Loss vs epoch', running_loss/(step+1), epoch)
            else:
                tlog['tl_loss'][str(epoch)] = running_loss/(step+1)

        if epoch in eplist:
            if not os.path.exists('./pretrained_models/'+savestring):
                os.makedirs('./pretrained_models/'+savestring)

            mergedict = {} # only save the encoder. ignore projector, target_encoder, etc
            mergedict.update(Encmodel.encoder.state_dict())
            if Predmodel is not None: # also add state_dict of predictor if not ssl
                mergedict.update(Predmodel.state_dict()) # we can do these since the keys do not overlap
            torch.save({'model_state_dict': mergedict,
                        'ptepoch': epoch},
                        './pretrained_models/'+savestring+'/'+'EP{}'.format(epoch)+ptstring)
    if args.log_to_tensorboard:
        writer.close()
    else:
        logfile = json.dumps(tlog)
        with open("./tlogs/"+savestring+'/' + ptstring+".json", "w") as f:
            f.write(logfile)

def finetune_model(args, model, ftlr, criterion, train_dataloader, test_dataloader, device, savestring, ftstring, PTEP):
    ''' We finetune on the small target dataset as per normal training.
        If we are doing pre-training (tl, sibcl or ssl), the pre-trained model state dicts will be loaded.
        Otherwise, if we are doing 'fromscratch', we leave the model randomly initialized using the default initialization '''
    if not os.path.exists('./dicts'):
        os.makedirs('./dicts')

    if os.path.exists("./dicts/"+args.iden+".json"):
        with open("./dicts/"+args.iden+".json") as f:
            eval_dict = json.load(f)
            print("loaded evaluation dict")
    else:
        eval_dict = {}
    dnsam = str(args.nsam); dPTEP = str(PTEP); dftseed = str(args.ftseed)

    if dnsam not in eval_dict:
        eval_dict[dnsam] = {}
    if dPTEP not in eval_dict[dnsam]:
        eval_dict[dnsam][dPTEP] = {}
    if dftseed not in eval_dict[dnsam][dPTEP]:
        eval_dict[dnsam][dPTEP][dftseed]=[99.0,[0,0,'']]

    minacrossftepoch = eval_dict[dnsam][dPTEP][dftseed][0]
    bestftparams = eval_dict[dnsam][dPTEP][dftseed][1]

    opt = torch.optim.Adam(model.parameters(),lr=ftlr, weight_decay=args.weight_decay)
    if not args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, \
        patience=5, verbose=False, threshold=0.01, factor = 0.1)

    model_dir = './pretrained_models/'+savestring+'/'+"EP{}".format(PTEP)+ftstring
    if args.log_to_tensorboard:
        tb_dir = f'FE{args.freeze_enc_ft}_{savestring}\
        /{ftstring}_EP{PTEP}_ftlr{ftlr}_ftseed{args.ftseed}_nsam{args.nsam}'

        writer = SummaryWriter(tb_dir)
    else:
        if not os.path.exists('./tlogs/'+savestring+'/nsam{}'.format(args.nsam)):
            os.makedirs('./tlogs/'+savestring+'/nsam{}'.format(args.nsam))
        ftstring2 = args.iden + "_bs{}_ftlr{}_ftseed{}".format(args.batchsize,ftlr,args.ftseed)

        if os.path.exists("./tlogs/"+savestring+'/nsam{}'.format(args.nsam)+'/' + ftstring2 +".json"):
            with open("./tlogs/"+savestring+'/nsam{}'.format(args.nsam)+'/' + ftstring2 +".json") as f:
                tlog = json.load(f)
        else:
            tlog = {}
            if 'loss' not in tlog:
                tlog['loss'] = {}
                tlog['testloss'] = {}
        if str(PTEP) not in tlog['loss']:
            tlog['loss'][str(PTEP)] = {}
            tlog['testloss'][str(PTEP)] = {}

    if args.train == 'tl' or args.train == 'sibcl':
        checkpoint = torch.load(model_dir,map_location = torch.device(device)) # we kept the same dict keys
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded {} state dict".format(args.train))

    elif args.train == 'ssl':
        model_dict = model.state_dict()
        checkpoint = torch.load(model_dir,map_location = torch.device(device))
        pretrained_dict = checkpoint['model_state_dict']
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Loaded SSL state dict")

    if args.freeze_enc_ft: ## freeze whole enc
        for name, param in model.named_parameters():
            if ("enc" in name):
                param.requires_grad = False

    for epoch in range(1,args.finetune_epochs+1):
        with torch.no_grad():
            model.eval() # evaluate model at every epoch
            if args.saveplots:
                if args.predict == 'DOS': # ELdiff
                    ploss, loss, output, actual, EL = evaluate_model(args, test_dataloader, model, device,
                    return_tensors = args.saveplots)
                else:
                    ploss, loss, output, actual = evaluate_model(args, test_dataloader, model, device,
                    return_tensors = args.saveplots)
            else:
                ploss, loss = evaluate_model(args, test_dataloader, model, device,
                return_tensors = args.saveplots)

            if args.log_to_tensorboard:
                writer.add_scalar("FT: Test L1 loss vs epoch", loss, epoch)
                writer.add_scalar("FT: Test percent loss vs epoch", ploss, epoch)
                writer.add_scalar("FT: Test percent loss vs nsam", ploss, args.nsam)
            else:
                tlog['testloss'][str(PTEP)][str(epoch)] = ploss

            if ploss < args.minsaveloss and args.saveplots:
                runstring = ftstring+f"_EP{PTEP}_ftlr{ftlr}_ftseed{args.ftseed}_nsam{args.nsam}"
                if args.predict == 'DOS':
                    save_plots(args, output, actual, savestring, runstring, num_to_save=5, startpt=startpt, emptylattice=EL)
                else:
                    save_plots(args, output, actual, savestring, runstring, num_to_save=5, startpt=startpt)

            minerror = min(minerror,ploss)

            if ploss <= minacrossftepoch:
                minacrossftepoch = ploss
                bestftparams = [ftlr,ftstring]

        model.train()
        epochloss = 0.

        for step, task_set in enumerate(train_dataloader):

            in_data = task_set[0]
            in_data = in_data.to(device)
            out_data = task_set[1].to(device)

            opt.zero_grad()
            loss = criterion(model(in_data), out_data)
            loss.backward()
            epochloss += loss.item()
            opt.step()
        if not args.no_scheduler:
            scheduler.step(epochloss)
        if args.log_to_tensorboard:
            writer.add_scalar("Finetuning loss vs epoch", epochloss/(step+1), epoch)
        else:
            tlog['loss'][str(PTEP)][str(epoch)] = epochloss/(step+1)

    if args.log_to_tensorboard:
        writer.close()
    else:
        logfile = json.dumps(tlog)
        with open("./tlogs/"+savestring+'/nsam{}'.format(args.nsam)+'/' + ftstring2 +".json","w") as f:
            f.write(logfile)

    eval_dict[dnsam][dPTEP][dftseed] = [minacrossftepoch, bestftparams]
    dictjson = json.dumps(eval_dict)
    with open("./dicts/EXP_"+args.iden+".json", "w") as f:
        f.write(dictjson)
        f.close()

def main(args):
    # Define device and reset seed
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    reset_seeds(args.seed)
    reset_seeds(args.ftseed)

    # Create path for pretrained models
    if not os.path.exists('./pretrained_models'):
        os.makedirs('./pretrained_models')

    ## Load model config
    if args.predict == 'DOS':
        cfg = DOSconfig()
    elif 'band' in args.predict:
        cfg = BSconfig()
    elif 'eig' in args.predict:
        cfg = TISEconfig()

    # Define strings for logging and saving model
    if args.train == 'fromscratch':
        savestring = args.iden+'FS'
        FSstring = 'FS_bs{}'.format(args.batchsize)
    elif args.train == 'tl':
        savestring = args.iden+'tl'
        ptstring = 'ptbs{}_ptlr{}'.format(args.batchsize,args.learning_rate)
    elif args.train == 'sibcl':
        savestring = args.iden+'sibcl_with_{}'.format(args.ssl_mode)
        ptstring = 'ptbs{}_bscl{}_temp{}_ptlr{}'.format(args.batchsize,args.batchsize_cl,args.temperature,args.learning_rate)
    elif args.train == 'ssl':
        savestring = args.iden+'ssl'
        ptstring = 'ssl_bscl{}_temp{}_ptlr{}'.format(args.batchsize_cl,args.temperature,args.learning_rate)

    # Retrieve Data
    print("Retrieving data.. ")
    if 'eig' in args.predict:
        src_Ds, tgt_train_Ds, tgt_test_Ds = get_TISE_datasets(args)
    else:
        src_Ds, tgt_train_Ds, tgt_test_Ds = get_PhC_datasets(args)

    tgt_train_Dl = data.DataLoader(tgt_train_Ds, batch_size = args.batchsize, shuffle = True)
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
    if args.train == 'fromscratch':
        # No pre-training needed
        reset_seeds(args.ftseed)
        model = Net(cfg.Lv,cfg.ks,cfg.Lvp,nbands=args.nbands,
                    ndos=400,neval=args.neval,predict=args.predict,
                    dim = args.ndim).to(device)
        finetune_model(args,model,args.learning_rate_ft, ft_criterion,
                    tgt_train_Dl, tgt_test_Dl, device, savestring, FSstring, 0)

    elif args.train == 'tl':
        eplist = [40, 100, 200] # epochs to save model and finetune from

        if not args.no_pretrain: # tl pre-training
            train_sDl = data.DataLoader(src_Ds, batch_size = args.batchsize)
            model = Net(cfg.Lv,cfg.ks,cfg.Lvp,nbands=args.nbands,
                        ndos=400,neval=args.neval,predict=args.predict,
                        dim = args.ndim).to(device)
            tl_pretrain(args, model, args.learning_rate, pt_criterion,
                    train_sDl, device, savestring, ptstring, eplist)

        if not args.no_finetune: # Fine-tuning
            reset_seeds(args.ftseed)
            for PTEP in eplist:
                model = Net(cfg.Lv,cfg.ks,cfg.Lvp,nbands=args.nbands,
                            ndos=400,neval=args.neval,predict=args.predict,
                            dim = args.ndim).to(device)
                finetune_model(args,model,args.learning_rate_ft, ft_criterion,
                    tgt_train_Dl, tgt_test_Dl, device, savestring, ptstring, PTEP)

    elif args.train == 'sibcl' or args.train == 'ssl': # sibcl, ssl
        eplist = [40, 100, 200, 300, 400] # epochs to save model and finetune from

        if not args.no_pretrain:
        # Load new unlabeled datasets for sibcl model training
            ntarget = 20000 if args.train == 'sibcl' else 20480 # ssl we use ~20k

            print("Retrieving unlabelled target data set..")
            ul_tgt_Ds = get_unlabeled_datasets(args)

            train_Ptxt_tDl = data.DataLoader(ul_tgt_Ds,batch_size = args.batchsize_cl,drop_last = True)
            train_Ptxt_sDl = None # to overwrite later for train == 'sibcl'
            train_sDl = None # to overwrite later for train == 'sibcl'
            Prednet = None # to overwrite later for train == 'sibcl'

            if args.train == 'sibcl':
                train_Ptxt_sDl = data.DataLoader(src_Ds,batch_size = int(args.batchsize_cl/2),drop_last = True)
                train_sDl = data.DataLoader(src_Ds, batch_size = args.batchsize) # this is predictor training set
                Prednet = Pred(cfg.latent_dim,cfg.Lvp,args.nbands,ndos,args.neval,args.predict,npredict,do=args.dropout).to(device)

            print("Unlabeled data retrieved!")
            print("Check: is source dataloader none?", train_Ptxt_sDl)

            batchsizeptxt = int(args.batchsize_cl/2*3) if train_Ptxt_sDl is not None else args.batchsize_cl

            if args.ssl_mode == 'simclr':
                Encnet = SimCLR(cfg.Lv,cfg.Lvpj,cfg.ks,args.ndim,device,batchsizeptxt,
                args.temperature,use_projector=cfg.use_projector,
                bnorm = cfg.proj_use_bnorm, depth = cfg.proj_depth).to(device)
            elif args.ssl_mode == 'byol':
                Encnet = BYOL(cfg.Lv,cfg.Lvpj,cfg.ks,args.ndim,use_projector=cfg.use_projector,
                    bnorm = cfg.proj_use_bnorm, depth = cfg.proj_depth).to(device)

            sibcl_pretrain(args, Encnet, args.learning_rate,
                            pt_criterion, train_Ptxt_tDl, device, savestring, ptstring, eplist,
                            Prednet, train_Ptxt_sDl, train_sDl)

        if not args.no_finetune:
            reset_seeds(args.ftseed)
            for PTEP in eplist:
                # we can load this since we kept the same state dict keys by introducing new model classes for Enc & Pred
                model = Net(cfg.Lv,cfg.ks,cfg.Lvp,nbands=args.nbands,
                        ndos=400,neval=args.neval,predict=args.predict,
                        dim = args.ndim).to(device)

                finetune_model(args,model,args.learning_rate_ft, ft_criterion ,tgt_train_Dl,
                                tgt_test_Dl, device, savestring, ptstring, PTEP)

    else:
        raise ValueError("Invalid train mode. Either fromscratch, tl, sibcl, ssl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Data and task parameters
    parser.add_argument('--path_to_h5', type = str, help="Directory with h5 data files", default = '/home/gridsan/cloh/h5data')
    parser.add_argument('--device',type=int, help="GPU device number. Default = 0 ", default = 0)
    parser.add_argument('--iden',type=str, required=True, help="REQUIRED: Model and logging identifier for current run")
    parser.add_argument('--predict', type = str, required=True, help = 'REQUIRED: DOS, bandstructures, oneband, eigval or eigvec')
    parser.add_argument('--train', type = str, help = 'tl, fromscratch, sibcl or ssl, default = sibcl', default = 'sibcl')
    parser.add_argument('--log_to_tensorboard',action='store_true')

    # PhC data parameters
    parser.add_argument('--nbands',type=int, help='if predict bands, specify num of bands, must be <= 10. default = 6', default=6)
    parser.add_argument('--srcband',type=int, help='if predict oneband, specify source band index, default = 0', default=0)
    parser.add_argument('--tgtband',type=int, help='if predict oneband, specify target band index, default = 0', default=0)

    # TISE data parameters
    parser.add_argument('--neval',type=int, help='if predict eigval or eigvec, specify num of eigvals/eigvecs counting from ground state, default = 1', default=1)
    parser.add_argument('--ndim',type=int, help='if predict eigval or eigvec, specify dimensions: 2 or 3, default = 2', default=2)
    parser.add_argument('--tisesource',type=str, help='specify source dataset, lr (lowres) or sho, default = lr', default='lr') ## TODO: modify sourceversion for SE

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
    parser.add_argument('--freeze_enc_ft',action='store_true',help="For pretrain_with_AE/CL, Set this flag to freeze whole enc in predictor finetuning")

    # CL specific parameters
    parser.add_argument('--ssl_mode', type = str, help = 'if sibcl, specify if simclr or byol, default = simclr', default = 'simclr')
    parser.add_argument('--batchsize_cl', type=int, help='specify batchsize for CL if sibcl or ssl, default = 512', default=512)
    parser.add_argument('--temperature',type=float, help='specify temperature for CL loss function', default=0.1)
    parser.add_argument('--train_ratio',type=int, help='specify no. of epoch for sibcl training before we train predictor once', default=1)

    # Augmentation parameters
    parser.add_argument('--translate_pbc', action='store_true', help = 'to randomly translate input image or not (to take care of PBC)')
    parser.add_argument('--pg_uniform', action='store_true', help = 'to uniformly sample point group operations (rotations and flips) ')
    parser.add_argument('--flip', action='store_true', help = 'to randomly flip image ')
    parser.add_argument('--rotate', action='store_true', help = 'to randomly rotate image')
    parser.add_argument('--scale', action='store_true', help = 'to randomly scale input')
    parser.add_argument('--stoch_aug_p', type=float, help = 'stochastic sampling parameter' default =1.0)

    # Visualize model performance
    parser.add_argument('--minsaveloss',type=float, help='if saveftmodel, specify min fractional loss in which we save ft model', default=0.05)
    parser.add_argument('--saveplots',action='store_true',help="Set this flag to save prediction plots")

    args = parser.parse_args()

    main(args)
