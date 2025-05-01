from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import argparse
import random

from model import DDPM, ContextUnet
from dataset import diffusion_ds


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def training(args):
    max_steps = args.max_steps 
    batch_size = args.batch_size 
    n_T = args.n_T 
    n_feat = args.n_feat 
    lrate = args.lrate 
    beta = args.beta
    dataset = args.dataset 
    num_samples = args.num_samples 
    pixel_size = args.pixel_size
    n_sample = args.n_sample 
    type_attention = args.type_attention 
    seed = args.seed
    scheduler = args.scheduler
    weight_decay = args.weight_decay
    in_channels = 3

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    data_path = "./datasets/single-body_2d_3classes 2"
    train_labels = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0)]
    train_dataset, test_datasets = diffusion_ds(data_path, train_labels, args.pixel_size)

    n_classes = [2, 2, 2]

    # Build model
    save_dir = './models/ddpm/shapes'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, f"{num_samples}_{n_feat}_{n_T}_{max_steps}_{lrate}_{beta}_{seed}_{batch_size}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    ddpm = DDPM(
        nn_model=ContextUnet(
            in_channels=in_channels,
            n_feat=n_feat,
            n_classes=n_classes,
            type_attention=type_attention,
        ), 
        betas=(lrate, 0.02),
        # betas=(0.9, 0.99),
        n_T=n_T,
        device=device,
        drop_prob=0.1,
        n_classes=n_classes,
    )
    ddpm.to(device)

    # Create dataloaders
    # train_dataset = load_dataset.my_dataset(tf, num_samples, dataset, configs=configs["train"], training=True, alpha=alpha, remove_node=remove_node)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloaders = {k: DataLoader(v, batch_size=n_sample, shuffle=False, num_workers=8) for k, v in test_datasets.items()}

    log_dict = {
        'train_loss_per_batch': [],
        'test_loss_per_batch': {key: [] for key, _ in test_datasets.items()},
    }

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate, weight_decay=weight_decay)

    step = 0

    infinite_train_loader = infinite_dataloader(train_dataloader)

    pbar = tqdm(range(max_steps))
    for step in pbar:

        x, c = next(infinite_train_loader)
        # pbar = tqdm(train_dataloader)
        # for x, c in pbar:
        ddpm.train()

        optim.zero_grad()
        x = x.to(device)
        _c = [tmpc.to(device) for tmpc in c] # .values()]
        loss = ddpm(x, _c)
        # log_dict['train_loss_per_batch'].append(loss.item())
        loss.backward()
        loss_ema = loss.item()
        pbar.set_description(f"loss: {loss_ema:.4f}")
        optim.step()

        # with torch.no_grad():
        # for test_config, test_dl in test_dataloaders.items():  # [test_config]:
        #     for test_x, test_c in test_dl:
        #         test_x = test_x.to(device)
        #         _test_c = [tmptest_c.to(device) for tmptest_c in test_c]  # .values()]
        #         test_loss = ddpm(test_x, _test_c)
        #         log_dict['test_loss_per_batch'][test_config].append(test_loss.item())

        if step == 0 or (step + 1) % 50 == 0:
            ddpm.eval()
            with torch.no_grad():
                for test_config, test_dl in test_dataloaders.items():
                    print()
                    x_real, c_gen = next(iter(test_dl))
                    x_real = x_real.to(device)
                    c_gen = [tmptest_c.to(device) for tmptest_c in c_gen]  # .values()]
                    if scheduler == "DDIM":
                        x_gen, _ = ddpm.sample_ddim(
                            n_sample,
                            c_gen,
                            (in_channels, pixel_size, pixel_size),
                            device,
                        )
                    else:
                        x_gen, _ = ddpm.sample(
                            n_sample,
                            c_gen,
                            (in_channels, pixel_size, pixel_size),
                            device,
                            guide_w=0.0,
                        )

                    test_config = str(test_config)
                    np.savez_compressed(save_dir + f"/image_" + test_config + "_step" + str(step) + ".npz", x_gen=x_gen.detach().cpu().numpy()) 
                    print('saved image at ' + save_dir + f"/image_" + test_config + "_step" + str(step) + ".npz")

    # with open(save_dir + f"training_log_" + str(ep) + ".json", "w") as outfile:
    #     json.dump(log_dict, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--beta', default=2.0, type=float)
    parser.add_argument('--num_samples', default=5000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_T', default=500, type=int)
    parser.add_argument('--n_feat', default=256, type=int)
    parser.add_argument('--n_sample', default=64, type=int)
    parser.add_argument('--max_steps', default=20000, type=int)
    parser.add_argument('--remove_node', default="None", type=str)
    parser.add_argument('--type_attention', default="", type=str)
    parser.add_argument('--pixel_size', default=32, type=int)
    parser.add_argument('--dataset', default="single-body_2d_3classes", type=str)
    parser.add_argument('--scheduler', default="", type=str)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    training(args)
