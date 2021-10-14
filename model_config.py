
class DOSconfig():
    Lvpj = [1024,256]
    use_projector = True
    # mode = 'ELdiff'
    ndim = 2
    ks = 7
    latent_dim = 1024
    Lv = [64,256,256,1024,1024]
    Lvp = [1024,1024,512]
    ft_lossfn = 'L1'
    pt_lossfn = 'log'
    use_projector = True
    proj_use_bnorm = True
    proj_depth = 4

class BSconfig():
    # mode = 'raw'
    ndim = 2
    ks = 9
    latent_dim = 256
    Lv = [64,256,256,256,1024]
    Lvp = [256,512,512] # we dont add too many nodes here since there's a branch for each band
    ft_lossfn = 'MSE'
    pt_lossfn = 'MSE'
    use_projector = True
    proj_use_bnorm = False
    proj_depth = 2


class TISEconfig():
    # mode = 'raw'
    ks = 7
    latent_dim = 256
    Lv = [64,256,256,256,256]
    Lvp = [256,256,32]
    ft_lossfn = 'MSE'
    pt_lossfn = 'MSE'
    use_projector = True
    proj_use_bnorm = False
    proj_depth = 2
