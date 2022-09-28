import argparse

import torch
import torchvision
from tqdm import trange
from src.models.stylegan_module_finetune import StyleGANFinetuneModule
from pathlib import Path

parser = argparse.ArgumentParser(description='Dumps paired images sampled from stylegan.')
parser.add_argument('--generator-ckpt', help='Path to generator checkpoint')
parser.add_argument('--num-samples', help='Number of generated samples', type=int, default=10_000)
parser.add_argument('--device', help='Target device ("cpu", "cuda:0")', default='cuda:0')
parser.add_argument('--output-dir', help='Output folder for sampled images', default='sampled_images/faces')
parser.add_argument('--batch-size', help='Batch size', type=int, default=16)

args = parser.parse_args()

REAL_FOLDER = 'real'
TRANSFER_FOLDER = 'transfer'

if __name__ == '__main__':
    model = StyleGANFinetuneModule.load_from_checkpoint(checkpoint_path=args.generator_ckpt, map_location=args.device, is_train=False, strict=False)
    model.eval()
    model.freeze()
    model = model.to(args.device)
    model.update_k_layered_gen()

    output_folder = Path(args.output_dir)
    output_folder.mkdir(exist_ok=True, parents=True)
    real_folder = output_folder / REAL_FOLDER
    real_folder.mkdir(exist_ok=True, parents=True)
    transfer_folder = output_folder / TRANSFER_FOLDER
    transfer_folder.mkdir(exist_ok=True, parents=True)

    z_generator = torch.Generator(args.device)
    i = 0
    num_iters = args.num_samples // args.batch_size + 1
    for _ in trange(num_iters):
        gen_z = torch.randn([args.batch_size, model.z_dim], generator=z_generator, device=args.device,
                            dtype=model.dtype)
        real, transfered = model.generate_pair_images(gen_z)

        for el in range(args.batch_size):
            if i > args.num_samples:
                break
            el_real = real[el: el + 1]
            el_transfered = transfered[el: el + 1]

            torchvision.utils.save_image(el_real, str(real_folder / (str(i) + '.png')))
            torchvision.utils.save_image(el_transfered, str(transfer_folder / (str(i) + '.png')))
            i += 1

