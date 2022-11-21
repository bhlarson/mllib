from pymlutil.s3 import s3store, Connect
import torch
import io
import sys
import torch
from torchdatasetutil.cityscapesstore import CreateCityscapesLoaders
sys.path.append('../')
from networks.network2d import Network2d
import tqdm
import torch
import numpy as np 
import time

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0) 

if __name__ == '__main__':

    #TODO: make these args

    #specify class_dictionary
    class_dict = 'cityscapes_2classes'

    ### Use custom class weight (recommended) or leave as [-1] for automated class weight
    custom_class_weight = [0.005, 1]

    #sample weight parameters
    sample_class = 1 # modify the sampling for images of these classes
    sample_rate = 2 # rate to sample image of classes specified by sample_classes
    



    s3, _, s3def = Connect('creds.yaml')
    class_dict_path = f'model/cityscapes/{class_dict}'
    class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'], class_dict_path + '.json')


    #load data
    loaders = CreateCityscapesLoaders(s3, s3def, 
        src = 'data/cityscapes',
        dest = '/data5/bishal/cityscapes',
        class_dictionary = class_dictionary,
        batch_size = 1,
        num_workers=1,
        height=768,
        width=512,
        shuffle=False)
    trainloader = next(filter(lambda d: d.get('set') == 'train', loaders), None)

    weights = []
    class_counts = {}

    for i in range(class_dictionary['classes']):
        class_counts[i]=0

    for i, data in enumerate(trainloader['dataloader']):
        inputs, labels, mean, stdev = data
        #calculate sample weights
        weight = 1
        if sample_class in labels:
            weight = sample_rate
        weights.append(weight)

        
        #calculate class counts
        class_idx, counts = torch.unique(labels, return_counts=True)
        for i, class_idx in enumerate(class_idx):
            class_counts[class_idx.item()]+=counts[i].item()


    #create class weights that are inversely proportional to total class area
    class_weights = np.array(list(class_counts.values()))
    class_weights = 1/class_weights
    class_weights = class_weights / class_weights.max() 

    class_dictionary['sample_weights'] = {'class':sample_class, 'weights': weights}
    class_dictionary['class_weights'] = class_weights.tolist()



    if -1 not in custom_class_weight:
        class_dictionary['class_weights'] = custom_class_weight
    

    s3.PutDict(s3def['sets']['dataset']['bucket'], class_dict_path + '.json', class_dictionary)
    