import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

sys.path.insert(0, os.path.abspath(''))
from utils.s3 import s3store, Connect
from utils.jsonutil import WriteDictJson, ReadDictJson
from datasets.citytorch import CityDataset
from segment_nas.trainargs import trainargs
from networks.cell2d import Cell, ConvBR

def make_data_loader(args, s3, dataset_list):

    training_list = list(filter(lambda d: d.get('set') == 'training', dataset_list['dataset']))
    test_list = list(filter(lambda d: d.get('set') == 'test', dataset_list['dataset']))

    training_data = CityDataset(s3, training_list, classes=args.classes)
    test_data = CityDataset(s3, test_list, classes=args.classes)

    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    return train_loader, test_loader


def train(epoch):
    epoch_loss = 0
    epoch_error = 0
    valid_iteration = 0
    
    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), (batch[2])
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        target=torch.squeeze(target,1)
        mask = target < args.maxdisp
        mask.detach_()
        valid = target[mask].size()[0]
        train_start_time = time()
        if valid > 0:
            model.train()
    
            optimizer.zero_grad()
            disp = model(input1,input2) 
            loss = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
            loss.backward()
            optimizer.step()
            
            error = torch.mean(torch.abs(disp[mask] - target[mask])) 
            train_end_time = time()
            train_time = train_end_time - train_start_time

            epoch_loss += loss.item()
            valid_iteration += 1
            epoch_error += error.item()
            print("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Error: ({:.4f}), Time: ({:.2f}s)".format(epoch, iteration, len(training_data_loader), loss.item(), error.item(), train_time))
            sys.stdout.flush()                        
    print("===> Epoch {} Complete: Avg. Loss: ({:.4f}), Avg. Error: ({:.4f})".format(epoch, epoch_loss / valid_iteration, epoch_error/valid_iteration))

def val():
    epoch_error = 0
    valid_iteration = 0
    three_px_acc_all = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
        target=torch.squeeze(target,1)
        mask = target < args.maxdisp
        mask.detach_()
        valid=target[mask].size()[0]
        if valid>0:
            with torch.no_grad(): 
                disp = model(input1,input2)
                error = torch.mean(torch.abs(disp[mask] - target[mask])) 

                valid_iteration += 1
                epoch_error += error.item()              
                #computing 3-px error#                
                pred_disp = disp.cpu().detach() 
                true_disp = target.cpu().detach()
                disp_true = true_disp
                index = np.argwhere(true_disp<args.maxdisp)
                disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
                correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 1)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
                three_px_acc = 1-(float(torch.sum(correct))/float(len(index[0])))

                three_px_acc_all += three_px_acc
    
                print("===> Test({}/{}): Error: ({:.4f} {:.4f})".format(iteration, len(testing_data_loader), error.item(), three_px_acc))
                sys.stdout.flush()

    print("===> Test: Avg. Error: ({:.4f} {:.4f})".format(epoch_error/valid_iteration, three_px_acc_all/valid_iteration))
    return three_px_acc_all/valid_iteration


def main(args):

    print('Start training')

    creds = ReadDictJson(args.credentails)
    s3_creds = next(filter(lambda d: d.get('name') == args.s3_name, creds), None)
    s3 = Connect(s3_creds)
    s3_index = s3.GetDict(s3_creds['index']['bucket'],s3_creds['index']['prefix'] )
    dataset_dfn = next(filter(lambda d: d.get('name') == args.dataset, s3_index['sets']['dataset']['datasets']), None)
    dataset_index = s3.GetDict(dataset_dfn['bucket'],dataset_dfn['prefix'] )

    if args.classes is None and args.classes_file is not None :
        if '.json' in args.classes_file:
            args.classes = ReadDictJson(args.classes_file)

    cuda = args.cuda

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    print('===> Loading datasets')
    kwargs = {'num_workers': args.threads, 'pin_memory': True, 'drop_last':True}
    training_data_loader, testing_data_loader = make_data_loader(args, s3, dataset_index)

    print('===> Building model')

#     def __init__(self, steps, block_multiplier, prev_prev_fmultiplier, prev_fmultiplier_same,filter_multiplier):

    model = Cell()

    ## compute parameters
    #print('Total number of model parameters : {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    #print('Number of Feature Net parameters: {}'.format(sum([p.data.nelement() for p in model.feature.parameters()])))
    #print('Number of Matching Net parameters: {}'.format(sum([p.data.nelement() for p in model.matching.parameters()])))

    print('Total Params = %.2fMB' % count_parameters_in_MB(model))
    print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
    print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))
    
    #mult_adds = comp_multadds(model, input_size=(3,args.crop_height, args.crop_width)) #(3,192, 192))
    #print("compute_average_flops_cost = %.2fMB" % mult_adds)

    if cuda:
        model = torch.nn.DataParallel(model).cuda()

    torch.backends.cudnn.benchmark = True

    if args.solver == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    elif args.solver == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        error=100
        for epoch in range(1, args.nEpochs + 1):
            train(epoch)
            is_best = False
            loss=val()
            if loss < error:
                error=loss
                is_best = True
            if args.dataset == 'sceneflow':
                if epoch>=0:
                    save_checkpoint(args.save_path, epoch,{
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        }, is_best)
            else:
                if epoch%100 == 0 and epoch >= 3000:
                    save_checkpoint(args.save_path, epoch,{
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        }, is_best)
                if is_best:
                    save_checkpoint(args.save_path, epoch,{
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        }, is_best)

            scheduler.step()

        save_checkpoint(args.save_path, args.nEpochs,{
                'epoch': args.nEpochs,
                'state_dict': model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best)

if __name__ == '__main__':
    args = trainargs()
  
    if args.debug:
        print("Wait for debugger attach")
        import debugpy
        ''' https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        Launch application from console with -debug flag
        $ python3 train.py -debug
        "configurations": [
            {
                "name": "Python: Remote",
                "type": "python",
                "request": "attach",
                "port": 3000,
                "host": "localhost",
                "pathMappings": [
                    {
                        "localRoot": "${workspaceFolder}",
                        "remoteRoot": "."
                    }
                ],
                "justMyCode": false
            },
            ...
        Connet to vscode "Python: Remote" configuration
        '''

        debugpy.listen(address=('0.0.0.0', args.debug_port))
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")

    main(args)
