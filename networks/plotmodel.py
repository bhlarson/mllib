import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from pymlutil.jsonutil import ReadDict, WriteDict, str2bool
from pymlutil.s3 import s3store, Connect

from torchdatasetutil.imagenetstore import CreateImagenetLoaders
from torchdatasetutil.cifar10store import CreateCifar10Loaders
from torchdatasetutil.cocostore import CreateCocoLoaders
from torchdatasetutil.imstore import  CreateImageLoaders
from torchdatasetutil.cityscapesstore import CreateCityscapesLoaders

sys.path.insert(0, os.path.abspath(''))
#import networks.cell2d as cell2d # import load, MakeNetwork
#import networks.network2d as netework2d # import load, MakeNetwork
from networks.cell2d import load as resnet_load
from networks.cell2d import MakeNetwork as resnet_MakeNetwork

from networks.network2d import load as unet_load
from networks.network2d import MakeNetwork as unet_MakeNetwork
from networks.totalloss import TotalLoss, FenceSitterEjectors
from networks.network2d import Network2d
from networks.network2d import ModelSize as NetworkModelSize
from networks.cell2d import ModelSize as CellModelSize




def parse_arguments():
    parser = argparse.ArgumentParser(description="RIVA WebRTC demo")

    parser.add_argument('-d', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug', type=str2bool, default=False, help='Wait for debuggee attach')
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('--verbose', '-v', action='store_true',help='Extended output')

    parser.add_argument('-credentails', type=str, default='creds.yaml', help='Credentials file.')
    parser.add_argument('-s3_name', type=str, default='store', help='S3 name in credentials')

    parser.add_argument('-args', type=str, default='serverconfig.yaml', help='Configuration')
    parser.add_argument('-env', type=str, default='config/config.sh', help='Output environment file')
    parser.add_argument('-results',  type=str, default='test', help='Results directory')

    parser.add_argument('-network', type=str, default='unet', choices=['unet', 'resnet'], help='network')
    parser.add_argument('-model_type', type=str,  default='segmentation')
    parser.add_argument('-model_class', type=str,  default='ImgSegmentPrune')
    parser.add_argument('-dataset', type=str, default='cityscapes', choices=['cifar10', 'imagenet', 'coco', 'lit', 'cityscapes'], help='Dataset')
    parser.add_argument('-dataset_path', type=str, default='/data', help='Local dataset path')
    parser.add_argument('-obj_imagenet', type=str, default='data/imagenet', help='Local dataset path')

    parser.add_argument('-lit_dataset', type=str, default='data/lit/dataset.yaml', help='Image dataset file')
    parser.add_argument('-lit_class_dict', type=str, default='model/crisplit/lit.json', help='Model class definition file.')

    parser.add_argument('-coco_class_dict', type=str, default='model/segmin/coco.json', help='Model class definition file.')

    parser.add_argument('-cityscapes_data', type=str, default='data/cityscapes', help='Image dataset file')
    parser.add_argument('-cityscapes_class_dict', type=str, default='model/cityscapes/cityscapes8.json', help='Model class definition file.')
    parser.add_argument('-sampler', type=bool, default=False, help='Toggle to use WeightedRandomSampler')
    
    parser.add_argument('-batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('-num_workers', type=int, default=1, help='Data loader workers')

    parser.add_argument('-height', type=int, default=768, help='Batch image height')
    parser.add_argument('-width', type=int, default=512, help='Batch image width')
    parser.add_argument('-cuda', type=str2bool, default=True)
    parser.add_argument('-unet_depth', type=int, default=5, help='number of encoder/decoder levels to search/minimize')
    parser.add_argument('-max_cell_steps', type=int, default=3, help='maximum number of convolution cells in layer to search/minimize')
    parser.add_argument('-channel_multiple', type=float, default=2, help='maximum number of layers to grow per level')
    parser.add_argument('-k_accuracy', type=float, default=1.0, help='Accuracy weighting factor')
    parser.add_argument('-k_structure', type=float, default=0.5, help='Structure minimization weighting factor')
    parser.add_argument('-k_prune_basis', type=float, default=1.0, help='prune base loss scaling')
    parser.add_argument('-k_prune_exp', type=float, default=50.0, help='prune basis exponential weighting factor')
    parser.add_argument('-k_prune_sigma', type=float, default=1.0, help='prune basis exponential weighting factor')
    parser.add_argument('-target_structure', type=float, default=0.00, help='Structure minimization weighting factor')
    parser.add_argument('-batch_norm', type=str2bool, default=False)
    parser.add_argument('-dropout', type=str2bool, default=False, help='Enable dropout')
    parser.add_argument('-dropout_rate', type=float, default=0.0, help='Dropout probability gain')
    parser.add_argument('-weight_gain', type=float, default=5.0, help='Channel convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convolution channels weights')
    parser.add_argument('-feature_threshold', type=float, default=0.0, help='cell tanh pruning threshold')
    parser.add_argument('-convMaskThreshold', type=float, default=0.5, help='convolution channel sigmoid level to prune convolution channels')
    parser.add_argument('-residual', type=str2bool, default=False, help='Residual convolution functions')
    parser.add_argument('-ejector', type=FenceSitterEjectors, default=FenceSitterEjectors.prune_basis, choices=list(FenceSitterEjectors))
    parser.add_argument('-ejector_start', type=float, default=4, help='Ejector start epoch')
    parser.add_argument('-ejector_full', type=float, default=5, help='Ejector full epoch')
    parser.add_argument('-ejector_max', type=float, default=1.0, help='Ejector max value')
    parser.add_argument('-ejector_exp', type=float, default=3.0, help='Ejector exponent')
    parser.add_argument('-search_flops', type=str2bool, default=True)
    parser.add_argument('-model_src', type=str,  default='ImgSegmentPrune_cityscapes_20221110_101703_hiocnn_search_structure_03')
    parser.add_argument('-model_dest', type=str, default='ImgSegmentPrune_cityscapes_20221110_101703_hiocnn_search_structure_03_plot')
    parser.add_argument('-save_image', type=str, default='img/class_weights.svg')



    args = parser.parse_args()

    if args.d:
        args.debug = args.d

    # Load arguments from file 
    if args.args is not None and os.path.exists(args.args):
        savedargs = ReadDict(args.args)
        if savedargs is not None:
            args.__dict__.update(savedargs)

    # record arguments in file
    now = datetime.now()
    args.__dict__['exec_datetime'] = now.strftime("%Y%m%d_%H%M%S")
    args.__dict__['exec_timestamp'] = now.strftime("%Y%m%d_%H%M%S")
    if args.results is not None:
        Path(args.results).mkdir(parents=True, exist_ok=True)
        WriteDict(args.__dict__, '{}/{}_{}.yaml'.format(args.results,'plotmodel', args.__dict__['exec_datetime'], args.args))

    return args

class PlotWeights():
    def __init__(self, save_image = 'class_weights.svg', 
                 title = None, 
                 colormapname = 'jet', 
                 diameter = 1.0, 
                 width=3, 
                 height=9, 
                 lenght = 1.0, 
                 dpi=600, 
                 thickness=1, 
                 fontfamily='serif', 
                 fontsize=9,
                 full_color='grey'  ):
        self.save_image = save_image
        self.title = title
        self.colormapname = colormapname
        self.diameter = diameter
        self.width = width
        self.height = height
        self.dpi = dpi
        self.cm = plt.get_cmap(colormapname)
        self.clear_frames = True
        self.lenght = lenght
        self.thickness=thickness
        self.fontfamily=fontfamily
        self.fontsize=fontsize
        self.full_color = full_color

    def plot_weights(self, full_weights, pruned_weights, index = None):


        plt.rcParams.update({
            'font.family': self.fontfamily,
            'font.size': self.fontsize})

        height = 0
        width = 0
        if len(full_weights) > 0:
            for i,  cell, in enumerate(full_weights):
                if len(cell['cell_weight']) > 0:                   
                    for j, step in enumerate(cell['cell_weight']):
                        width += 1
                        height = max(height, len(step))
                        
        self.fig, self.ax = plt.subplots(figsize=(self.width,self.height), dpi=self.dpi) # note we must use plt.subplots, not plt.subplot       
        self.ax.clear()

        if self.title is not None:
            if index:
                title = '{} {}'.format(self.title, index)
            else:
                title = self.title

            self.ax.set_title(title)

        x = 0
        for i,  cell, in enumerate(full_weights):
            for j, step in enumerate(cell['cell_weight']):
                for k, gain in enumerate(step.cpu().detach().numpy()):
                    y = k

                    line = Line2D([x,x+self.lenght],[y,y], linewidth=self.thickness, color=self.full_color)
                    self.ax.add_line(line)
                x += self.lenght

        x = 0
        for i,  cell, in enumerate(pruned_weights):
            prune_weight = cell['prune_weight'].item()
            for j, step in enumerate(cell['cell_weight']):
                for k, gain in enumerate(step.cpu().detach().numpy()):
                    conv_gain = prune_weight*gain
                    y = k
                    
                    line = Line2D([x,x+self.lenght],[y,y], linewidth=self.thickness, color=self.cm(conv_gain))
                    self.ax.add_line(line)
                x += self.lenght

        dx_text = 0.5
        dy_text = -10
        x = 0
        iMax = 100
        y0 = 0.85*height
        x0 = 0.60*width
        for i in range(iMax):

            y = i+y0
            line = Line2D([x0,x0+self.lenght],[y,y], linewidth=self.thickness, color=self.cm(i/iMax))
            self.ax.add_line(line)
            x += self.lenght

        y_space = 15
        full_height = 10
        for i in range(full_height):

            y = i+y0+iMax+y_space
            line = Line2D([x0,x0+self.lenght],[y,y], linewidth=self.thickness, color=self.full_color)
            self.ax.add_line(line)
            x += self.lenght

        plt.text(x0+self.lenght+dx_text, y0, '$\sigma(s)=0$')
        plt.text(x0+self.lenght+dx_text, y0+iMax+dy_text, '$\sigma(s)=1$')
        plt.text(x0+self.lenght+dx_text, y0+iMax+full_height/2-5+y_space, 'unpruned')

        # self.ax.set_axis_off()
        #  self.ax.set_xlim((0, len(full_weights))
        self.ax.set_ylim((0, height))

        self.ax.set_xlim((0, width))
        self.ax.set_ylim((0, height))

        self.ax.set_xlabel('subnetwork (~convolution)')
        self.ax.set_ylabel('channel')

        plt.text(-2, 0.02*height, 'image input', rotation='vertical')
        plt.text(width+1, 0.02*height, 'output segmentation', rotation='vertical')

        self.fig.tight_layout() 
        self.fig.savefig(self.save_image)


def main(args):

    results = {
            'batches': 0,
            'initial_parameters': None,
            'initial_flops': None,
            'runs': {},
            'load': {},
            'prune': {},
            'store': {},
            'train': {},
            'test': {},
        }

    s3, _, s3def = Connect(args.credentails, s3_name=args.s3_name)

    # Load dataset
    class_dictionary = None
    dataset_bucket = s3def['sets']['dataset']['bucket']

    # Load dataset
    if args.dataset == 'cifar10':
        loaders = CreateCifar10Loaders(args.dataset_path, batch_size = args.batch_size,  
                                       num_workers=args.num_workers, 
                                       cuda = args.cuda, 
                                       rotate=args.augment_rotation, 
                                       scale_min=args.augment_scale_min, 
                                       scale_max=args.augment_scale_max, 
                                       offset=args.augment_translate_x,
                                       augment_noise=args.augment_noise,
                                       width=args.width, height=args.height)

    elif args.dataset == 'imagenet':
        loaders = CreateImagenetLoaders(s3, s3def, 
                                        args.obj_imagenet, 
                                        args.dataset_path+'/imagenet', 
                                        width=232, 
                                        height=args.height, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers,
                                        cuda = args.cuda,
                                        rotate=args.augment_rotation, 
                                        scale_min=args.augment_scale_min, 
                                        scale_max=args.augment_scale_max, 
                                        offset=args.augment_translate_x,
                                        augment_noise=args.augment_noise,
                                        normalize=False
                                       )

    elif args.dataset=='coco':
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.coco_class_dict)
        loaders = CreateCocoLoaders(s3, dataset_bucket, 
            class_dict=args.coco_class_dict, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cuda = args.cuda,
            height = args.height,
            width = args.width,
        )
    elif args.dataset=='lit':
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.lit_class_dict)
        loaders = CreateImageLoaders(s3, dataset_bucket, 
            dataset_dfn=args.lit_dataset,
            class_dict=args.lit_class_dict, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cuda = args.cuda,
            height = args.height,
            width = args.width,
        )
    elif args.dataset=='cityscapes':
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.cityscapes_class_dict)

        if 'sample_weights' in class_dictionary.keys() and args.sampler:
            train_sampler_weights = class_dictionary['sample_weights']['weights']
            upsampled_class = class_dictionary['sample_weights']['class']
        else:
            train_sampler_weights = None


        loaders = CreateCityscapesLoaders(s3, s3def, 
            src = args.cityscapes_data,
            dest = args.dataset_path+'/cityscapes',
            class_dictionary = class_dictionary,
            batch_size = args.batch_size, 
            num_workers=args.num_workers,
            height=args.height,
            width=args.width, 
            train_sampler_weights=train_sampler_weights,
            )
    else:
        raise ValueError("Unupported dataset {}".format(args.dataset))

    full_model = None
    pruned_model = None
    full_model_parameters = 0
    full_model_flops = 0
    pruned_model_parameters = 0
    pruned_model_flops = 0
    if args.network == 'unet':

        full_model = unet_MakeNetwork(class_dictionary, args)
        pruned_model, results = unet_load(s3, s3def, args, class_dictionary, loaders, results)

        full_model_parameters, full_model_flops = NetworkModelSize(args, full_model, class_dictionary)
        pruned_model_parameters, pruned_model_flops = NetworkModelSize(args, pruned_model, class_dictionary)

    plotSearch = PlotWeights(save_image = args.save_image)



    architecture_weights, total_trainable_weights, full_weights = full_model.ArchitectureWeights()
    architecture_weights, total_trainable_weights, pruned_weights = pruned_model.ArchitectureWeights()

    plotSearch.plot_weights(full_weights, pruned_weights)

if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach on {}:{}".format(args.debug_address, args.debug_port))
        import debugpy

        debugpy.listen(address=(args.debug_address, args.debug_port)) # Pause the program until a remote debugger is attached
        debugpy.wait_for_client() # Pause the program until a remote debugger is attached
        print("Debugger attached")

    result = main(args)
    sys.exit(result)