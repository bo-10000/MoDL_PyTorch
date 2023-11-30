import argparse
import yaml, os, time
import h5py

from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from get_instances import *
from utils import *

def setup(args):
    config_path = args.config
    with open(config_path, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)
    device = 'cuda'

    #read configs =================================
    n_layers = configs['n_layers']
    k_iters = configs['k_iters']

    dataset_name = configs['dataset_name']
    dataset_params = configs['dataset_params']

    batch_size = configs['batch_size'] if args.batch_size is None else args.batch_size

    model_name = configs['model_name']
    model_params = configs.get('model_params', {})
    model_params['n_layers'] = n_layers
    model_params['k_iters'] = k_iters

    score_names = configs['score_names']

    config_name = configs['config_name']

    workspace = os.path.join(args.workspace, config_name) #workspace/config_name
    checkpoints_dir, log_dir = get_dirs(workspace) #workspace/config_name/checkpoints ; workspace/config_name/log.txt
    tensorboard_dir = os.path.join(args.tensorboard_dir, configs['config_name']) #runs/config_name
    logger = Logger(log_dir)
    writer = get_writers(tensorboard_dir, ['test'])['test']

    dataloader = get_loaders(dataset_name, dataset_params, batch_size, ['test'])['test']
    model = get_model(model_name, model_params, device)
    score_fs = get_score_fs(score_names)

    #restore
    saver = CheckpointSaver(checkpoints_dir)
    prefix = 'best' if configs['val_data'] else 'final'
    checkpoint_path = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.startswith(prefix)][0]
    model = saver.load_model(checkpoint_path, model)

    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

    return configs, device, workspace, logger, writer, dataloader, model, score_fs, tensorboard_dir

def main(args):
    configs, device, workspace, logger, writer, dataloader, model, score_fs, tensorboard_dir = setup(args)

    logger.write('\n')
    logger.write('test start: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    start = time.time()

    running_score = defaultdict(int)

    model.eval()
    for i, (x, y, csm, mask) in enumerate(tqdm(dataloader)):
        x, csm, mask = x.to(device), csm.to(device), mask.to(device)

        with torch.no_grad():
            y_pred = model(x, csm, mask).detach().cpu()

        y = np.abs(r2c(y.numpy(), axis=1))
        y_pred = np.abs(r2c(y_pred.numpy(), axis=1))

        f = h5py.File(tensorboard_dir + '/test/recon_' + str(i).zfill(3) + '.h5', 'w')
        f.create_dataset('recon', data=y_pred)
        f.create_dataset('gt', data=y)
        f.close()

        for score_name, score_f in score_fs.items():
            running_score[score_name] += score_f(y, y_pred) * y_pred.shape[0]
        if args.write_image > 0 and (i % args.write_image == 0):
            writer.add_figure('img', display_img(np.abs(r2c(x[-1].detach().cpu().numpy())), mask[-1].detach().cpu().numpy(), \
                y[-1], y_pred[-1], psnr(y[-1], y_pred[-1])), i)

    epoch_score = {score_name: score / len(dataloader.dataset) for score_name, score in running_score.items()}
    for score_name, score in epoch_score.items():
        writer.add_scalar(score_name, score, 0)
        logger.write('test {} score: {:.4f}'.format(score_name, score))

    writer.close()
    logger.write('-----------------------')
    logger.write('total test time: {:.2f} min'.format((time.time()-start)/60))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=False, default="configs/base_modl,k=1.yaml",
                        help="config file path")
    parser.add_argument("--workspace", type=str, default='./workspace')
    parser.add_argument("--tensorboard_dir", type=str, default='./runs')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--write_image", type=int, default=0)

    args = parser.parse_args()

    main(args)