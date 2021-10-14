from datasets_PhC_SE import PhC2D, TISEdata

def get_TISE_datasets(args):
    if args.train == 'ssl' or args.train == 'fromscratch':
        src_ds = None
    else:
        src_ds = TISEdata(args.path_to_h5, args.nsource , validsize = 0, testsize = 0,
                            predict=args.predict,ndim = args.ndim, neval = args.neval,
                            domain = 'source', split = 'train',
                            lowres = 5, downres = 32,
                            tisesource = args.tisesource)
    pick_random_ftset = np.random.choice(np.arange(1,10000),args.nsam,replace=False)

    totalnum = 14999 if args.ndim == 2 else 11000
    tgt_train_ds = TISEdata(args.path_to_h5, args.nsam , validsize = totalnum-2000-args.nsam,
                            testsize = 2000,predict=args.predict,ndim = args.ndim,
                            neval = args.neval,domain = 'target', split = 'train',
                            lowres = 5, downres = 32,
                            ftsetind = pick_random_ftset)

    tgt_test_ds = TISEdata(args.path_to_h5, args.nsam , validsize = totalnum-2000-args.nsam,
                            testsize = 2000, predict=args.predict,ndim = args.ndim,
                            neval = args.neval, domain = 'target', split = 'test',
                            lowres = 5, downres = 32,
                            ftsetind = pick_random_ftset)
    return src_ds, tgt_train_ds, tgt_test_ds

def get_PhC_datasets(args):
    if args.train == 'ssl' or args.train == 'fromscratch':
        src_ds = None
    else:
        srcdomain = 'target' if args.predict == 'oneband' else 'source'
        mode = 'ELdiff' if args.predict == 'DOS' else 'raw'
        src_ds = PhC2D(args.path_to_h5, args.nsource , validsize = 500, testsize = 0,
                        predict=args.predict,mode=mode,
                        domain = srcdomain, split = 'train', band = args.srcband)

    pick_random_ftset = np.random.choice(np.arange(90,18000),args.nsam,replace=False)
    tgt_train_ds = PhC2D(args.path_to_h5, args.nsam , validsize = 0, testsize = 2000,
                            predict=args.predict, mode=mode, domain = 'target',
                            split = 'train', band = args.tgtband, ftsetind = pick_random_ftset)

    tgt_test_ds = PhC2D(args.path_to_h5, args.nsam , validsize = 5000-args.nsam, testsize = 2000,
                            predict=args.predict,mode=mode, domain = 'target',
                            split = 'test', band = args.tgtband, ftsetind = pick_random_ftset)
    return src_ds, tgt_train_ds, tgt_test_ds

def get_unlabeled_datasets(args):
    ntarget = args.nsource*2
    if 'eig' in args.predict:
        ul_ds = TISEdata(args.path_to_h5, ntarget , validsize = 0, testsize = 0,
                                predict=args.predict,ndim = args.ndim, neval = args.neval,
                                domain = 'nolabel', split = 'train',
                                lowres = 5, tisesource=args.tisesource,
                                downres = 32)
    else:
        ul_ds = PhC2D(args.path_to_h5, ntarget, validsize = 0, testsize =0,
                            predict=args.predict,mode=args.mode,domain = 'target',
                            targetmode = 'unlabeled', split = 'train')
    return ul_ds
