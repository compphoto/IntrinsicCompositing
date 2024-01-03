from datetime import datetime

def normalize(x):
    return (x - 0.5) / 0.5

def create_exp_name(args, prefix='RealismNet'):

    components = []
    components.append(f"{prefix}")
    components.append(f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}")
    components.append(f"{args.expdscp}")
    components.append(f"lrd_{args.lr_d}")

    name = "_".join(components)
    return name

def create_exp_name_editnet(args, prefix='EditingNet'):

    components = []
    components.append(f"{prefix}")
    components.append(f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}")
    components.append(f"{args.expdscp}")
    components.append(f"lredit_{args.lr_editnet}")
    components.append(f"rlubias_{args.loss_relu_bias}")
    components.append(f"edtloss_{args.edit_loss}")
    components.append(f"fkgnlwdv_{args.fake_gen_lowdev}")
    components.append(f"blrshrpn_{args.blursharpen}")

    name = "_".join(components)
    return name