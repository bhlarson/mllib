import argparse
import json

def trainargs():

    # Training settings
    parser = argparse.ArgumentParser(description='LEStereo training...')
    parser.add_argument('-maxdisp', type=int, default=192, 
                        help="max disp")
    parser.add_argument('-crop_height', type=int, default=512, 
                        help="crop height")
    parser.add_argument('-crop_width', type=int, default=512, 
                        help="crop width")
    parser.add_argument('-resume', type=str, default='', 
                        help="resume from saved model")
    parser.add_argument('-batch_size', type=int, default=4, 
                        help='training batch size')
    parser.add_argument('-testBatchSize', type=int, default=8, 
                        help='testing batch size')
    parser.add_argument('-nEpochs', type=int, default=2048, 
                        help='number of epochs to train for')
    parser.add_argument('-solver', default='adam',choices=['adam','sgd'],
                        help='solver algorithms')
    parser.add_argument('-lr', type=float, default=0.001, 
                        help='Learning Rate. Default=0.001')
    parser.add_argument('-cuda', type=int, default=1, 
                        help='use cuda? Default=True')
    parser.add_argument('-gpu-ids', type=json.loads, default='[0,1,2,3]',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('-threads', type=int, default=1, 
                        help='number of threads for data loader to use')
    parser.add_argument('-seed', type=int, default=2019, 
                        help='random seed to use. Default=123')
    parser.add_argument('-shift', type=int, default=0, 
                        help='random shift of left image. Default=0')
    parser.add_argument('-save_path', type=str, default='./checkpoints/', 
                        help="location to save models")
    parser.add_argument('-milestones', default=[30,50,300], metavar='N', nargs='*', 
                        help='epochs at which learning rate is divided by 2')    
    parser.add_argument('-stage', type=str, default='train', choices=['search', 'train'])


    ######### LEStereo params ##################
    parser.add_argument('-fea_num_layers', type=int, default=6)
    parser.add_argument('-mat_num_layers', type=int, default=12)
    parser.add_argument('-fea_filter_multiplier', type=int, default=8)
    parser.add_argument('-mat_filter_multiplier', type=int, default=8)
    parser.add_argument('-fea_block_multiplier', type=int, default=4)
    parser.add_argument('-mat_block_multiplier', type=int, default=4)
    parser.add_argument('-fea_step', type=int, default=2)
    parser.add_argument('-mat_step', type=int, default=2)
    parser.add_argument('-net_arch_fea', default=None, type=str)
    parser.add_argument('-cell_arch_fea', default=None, type=str)
    parser.add_argument('-net_arch_mat', default=None, type=str)
    parser.add_argument('-cell_arch_mat', default=None, type=str)

    parser.add_argument('-debug', action='store_true', help='True, enable debug and stop at breakpoint')
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')

    ######### Cityscape Dataset Parsing ##################
    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-s3_name', type=str, default='mllib-s3', help='Credential file s3 name.')
    parser.add_argument('-dataset', type=str, default='cityscapes', help='Dataset name.')
    parser.add_argument('-set', type=str, default='training', help='Set to extract from dataset')

    parser.add_argument('-classes', type=json.loads, default=None, help='Class dictionary JSON.  Leave empty if classes_file points to a JSON file.')
    parser.add_argument('-classes_file', type=str, default='datasets/cityscapes.json', help='Class dictionary JSON file')

    args = parser.parse_args()
    return args
