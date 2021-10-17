import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from symop_utils import get_aug
from loss_func import LogLoss, FracLoss, DOSLoss
from lr_utils import Cosine_Scheduler

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

    paramsum = 0
    for name, parameter in Encmodel.named_parameters():
        if not parameter.requires_grad: continue
        paramsum+=parameter.numel()
    print("Pre-training no. of trainable params: ", paramsum)

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
        Otherwise, if we are doing 'sl', we leave the model randomly initialized using the default initialization '''
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
    minerror = 99.0
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
