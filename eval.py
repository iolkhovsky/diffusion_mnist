import argparse
import cv2
import numpy as np
import os
import torch

from ddpm import DDPM
from utils import get_available_device
from utils import images_to_grid, save_gif


def parse_args():
    parser = argparse.ArgumentParser(prog='DDPM evaluator')
    parser.add_argument(
        '--device', type=str,
        default=get_available_device(),
        help='Execution device',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default=os.path.join('logs', 'experiment', 'version_0', 'checkpoints', 'epoch-0000-step-469.ckpt'),
        help='Abs path to fine-tuning checkpoint',
    )
    parser.add_argument(
        '--guide_w', type=float,
        default=0.5,
        help='Guidence weight for diffusion',
    )
    return parser.parse_args()


@torch.no_grad()
def eval(args):
    model = DDPM.load_from_checkpoint(
        args.checkpoint,
        map_location=torch.device(args.device)
    ).eval()
    guide_w = torch.tensor(args.guide_w).to(args.device)
    contexts = torch.tensor(
        list(range(10)) * 10
    ).long().to(args.device)
    print('Generating...')
    _, history = model.forward(contexts, guide_w, True)
    print(f'Done!')

    history = [
        cv2.resize(
            images_to_grid(np.transpose(x, axes=(0, 2, 3, 1))),
            (640, 640)
        ) 
        for x in history
    ]
    save_gif(history, f'evaluation.gif')

    frame_idx = 0
    while True:
        frame_text = f'Step #{frame_idx + 1}/{len(history)}'
        vis = cv2.putText(history[frame_idx], frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Diffusion', vis)

        if frame_idx < len(history) - 1:
            frame_idx += 1

        key = cv2.waitKey(50)
        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_idx = 0


if __name__ == '__main__':
    eval(parse_args())
