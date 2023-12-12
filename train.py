import argparse
import os, time, yaml
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from utils import *
from get_instances import *

def setup(args):
    config_path = args.config
    with open(config_path, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #read configs =================================
    n_layers = configs['n_layers']
    k_iters = configs['k_iters']
    epochs = configs['epochs']

    dataset_name = configs['dataset_name']
    dataset_params = configs['dataset_params']
    val_data = configs['val_data']
    phases = ['train', 'val'] if val_data else ['train']

    batch_size = configs['batch_size']

    model_name = configs['model_name']
    model_params = configs.get('model_params', {})
    model_params['n_layers'] = n_layers
    model_params['k_iters'] = k_iters

    restore_weights = configs['restore_weights'] #'model', 'all', False

    loss_name = configs['loss_name']
    score_names = configs['score_names']
    optim_name = configs['optim_name']
    optim_params = configs.get('optim_params', {})
    scheduler_name = configs.get('scheduler_name', None)
    scheduler_params = configs.get('scheduler_params', {})

    # config_name = configs['config_name'] + '_' + datetime.now().strftime("%d%b%I%M%P") #ex) base_04Jun0243pm
    config_name = configs['config_name'] #ex) base

    #dirs, logger, writers, saver =========================================
    workspace = os.path.join(args.workspace, config_name) #workspace/config_name
    checkpoints_dir, log_dir = get_dirs(workspace, remake=True) #workspace/config_name/checkpoints ; workspace/config_name/log.txt
    tensorboard_dir = os.path.join(args.tensorboard_dir, config_name) #runs/config_name
    logger = Logger(log_dir)
    writers = get_writers(tensorboard_dir, phases)
    saver = CheckpointSaver(checkpoints_dir)

    #dataloaders, model, loss f, score f, optimizer, scheduler================================
    dataloaders = get_loaders(dataset_name, dataset_params, batch_size, phases)
    model = get_model(model_name, model_params, device)
    loss_f = get_loss(loss_name)
    score_fs = get_score_fs(score_names)
    val_score_name = score_names[0]
    optim_params['params'] = model.parameters()
    optimizer, scheduler = get_optim_scheduler(optim_name, optim_params, scheduler_name, scheduler_params)

    #load weights ==========================================
    if restore_weights:
        restore_path = configs['restore_path']
        start_epoch, model, optimizer, scheduler = saver.load(restore_path, restore_weights, model, optimizer, scheduler)
    else:
        start_epoch = 0

    # if torch.cuda.device_count()>1:
    #     model = nn.DataParallel(model)

    return configs, device, epochs, start_epoch, phases, workspace, logger, writers, saver, dataloaders, model, loss_f, score_fs, val_score_name, optimizer, scheduler

def main(args):
    configs, device, epochs, start_epoch, phases, workspace, logger, writers, saver, \
        dataloaders, model, loss_f, score_fs, val_score_name, optimizer, scheduler = setup(args)
    """
    :start_epoch: The point at which epoch starts from. 0 if restore_weights is False
    :phases: list of phases. ['train', 'val'] if val_data is True, else ['train']
    :workspace: Where all data are saved.
    :checkpoints_dir: intermediate checkpoints and final model path are saved.
    :logger: can write log by using logger.write() method
    :writers: tensorboard writers
    :score_fs: dictionary of scoring functions
    """

    logger.write('config path: ' + args.config)
    logger.write('workspace: ' + workspace)
    logger.write('description: ' + configs['description'])
    logger.write('\n')
    logger.write('train start: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.write('-----------------------')

    start = time.time()
    if args.seed:
        set_seeds(args.seed)

    for epoch in range(start_epoch, epochs):
        for phase in phases: #['train', 'val'] or ['train']
            running_score = defaultdict(int)

            if phase == 'train': model.train()
            else: model.eval()

            for i, (x, y, csm, mask) in enumerate(tqdm(dataloaders[phase])):
                x, y, csm, mask = x.to(device), y.to(device), csm.to(device), mask.to(device)

                with torch.set_grad_enabled(phase=='train'):
                    y_pred = model(x, csm, mask)
                    loss = loss_f(y_pred, y)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    if configs['gradient_clip']:
                        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                    optimizer.step()

                running_score['loss'] += loss.item() * y_pred.size(0)
                y = np.abs(r2c(y.detach().cpu().numpy(), axis=1))
                y_pred = np.abs(r2c(y_pred.detach().cpu().numpy(), axis=1))
                for score_name, score_f in score_fs.items():
                    running_score[score_name] += score_f(y, y_pred) * y_pred.shape[0]

            #scheduler
            if phase == 'train' and scheduler:
                scheduler.step()

            #write log
            epoch_score = {score_name: score / len(dataloaders[phase].dataset) for score_name, score in running_score.items()}
            for score_name, score in epoch_score.items():
                writers[phase].add_scalar(score_name, score, epoch)
            if args.write_lr:
                writers[phase].add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            if args.write_image > 0 and (epoch % args.write_image == 0):
                writers[phase].add_figure('img', display_img(np.abs(r2c(x[-1].detach().cpu().numpy())), mask[-1].detach().cpu().numpy(), \
                    y[-1], y_pred[-1], epoch_score[val_score_name]), epoch)
            if args.write_lambda:
                print('lam:', model.dc.lam.item())
                writers['train'].add_scalar('lambda', model.dc.lam.item(), epoch)

            logger.write('epoch {}/{} {} score: {:.4f}\tloss: {:.4f}'.format(epoch, epochs, phase, epoch_score[val_score_name], epoch_score['loss']))

        #save model
        if phase == 'val':
            saver.save_model(model, epoch_score[val_score_name], epoch, final=False)
        if epoch % args.save_step == 0:
            saver.save_checkpoints(epoch, model, optimizer, scheduler)

    if phase == 'train':
        saver.save_model(model, epoch_score[val_score_name], epoch, final=True)

    for phase in phases:
        writers[phase].close()

    logger.write('-----------------------')
    logger.write('total train time: {:.2f} min'.format((time.time()-start)/60))
    logger.write('best score: {:.4f}'.format(saver.best_score))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=False, default="configs/base_modl,k=10.yaml",
                        help="config file path")
    parser.add_argument("--workspace", type=str, default='./workspace')
    parser.add_argument("--tensorboard_dir", type=str, default='./runs')
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--write_lr", type=bool, default=False)
    parser.add_argument("--write_image", type=int, default=0)
    parser.add_argument("--write_lambda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    main(args)
