import os
import sys
import io
import json
import yaml
import pandas as pd
import pathlib
from pathlib import Path, PurePath
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pymlutil.jsonutil import ReadDict, WriteDict, str2bool
import matplotlib
import matplotlib.pyplot as plt

# Extraction function
def ParseTBEvents(path: str) -> {}:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow event file
    Returns
    -------
    log dictionary
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    log_dict = {}
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            log_dict[tag] = event_acc.Scalars(tag)
            #values = list(map(lambda x: x.value, event_list))
            #step = list(map(lambda x: x.step, event_list))
            #r = {"metric": [tag] * len(step), "value": values, "step": step}
            #r = pd.DataFrame(r)
            #log_dict = pd.concat([log_dict, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
    return log_dict

def ReadTB(tb_path):
    p = Path(tb_path)
    events = list(p.glob('**/event*'))
    all_logs = {}
    for event in events:
        log = ParseTBEvents(event.as_posix())
        if log is not None:
            all_logs[event.parent.name] = log

    return all_logs

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-config', type=str, default='convergence.yaml', help='Configuration')
    parser.add_argument('-d', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug', type=str2bool, default=False, help='Wait for debuggee attach')
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-tb', '--tensorboard', type=str, default='/data/tb_logs/crispcityscapes_20221021_151053_hiocnn_tb', help='Debug port')
    parser.add_argument('-column_width', type=float, default=3.25, help='Colum width in Inches')
    parser.add_argument('-page_width', type=float, default=6.875, help='Colum width in Inches')
    parser.add_argument('-fontdir', type=str, default='./fonts', help='Path to fonts')

    # tensorboard --bind_all --logdir /data/tb_logs/crispcityscapes_20221010_112200_hiocnn_test

    args = parser.parse_args()

    if args.d:
        args.debug = args.d

    if args.config is not None and os.path.exists(args.config):
        config = ReadDict(args.config)
        if config is not None:
            args.__dict__.update(config)

        WriteDict(args.__dict__, args.config)

    return args
tableau_colors=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

test_plots = [
    {
        'tensorboard': '/data/tb_logs/crispcityscapes_20221021_151053_hiocnn_tb',
        'title': 'Multi-step Pruning',
        'sets': [
            {  'name':'cross_entropy_loss/train',
               'size':1,
               'marker':'.',
               'alpha':0.1 },
            {'name':'cross_entropy_loss/test',
               'size':1.,
               'marker':'.',
               'alpha':1.0 }
        ],
        'yscale': [0.0, 1.5],
        'colors':tableau_colors
    },
    {
        'tensorboard': '/data/tb_logs/crispcityscapes_20221018_231609_hiocnn_tb',
        'title': 'Multi-step Pruning',
        'sets': [
            {  'name':'cross_entropy_loss/train',
               'size':1,
               'marker':'.',
               'alpha':0.1 },
            {'name':'cross_entropy_loss/test',
               'size':1.,
               'marker':'.',
               'alpha':1.0 }
        ],
        'yscale': [0.0, 1.5],
        'colors':tableau_colors
    }
]

def PlotTraining(fontdir, figsize, plots=test_plots, fontfamily='serif',fontname='Times New Roman Cyr', fontsize=9):
    # plot the images in the batch, along with predicted and true labels

    for font in matplotlib.font_manager.findSystemFonts(fontdir):
        matplotlib.font_manager.fontManager.addfont(font)

    plt.rcParams.update({
        'font.family': fontfamily,
        'font.serif':fontname,
        'font.size': fontsize})

    fig = plt.figure(figsize=figsize)

    for idx, plot  in enumerate(plots):
        ax = fig.add_subplot(len(plots), 1, idx+1)
        ax.set_title(plot['title'])

        all_logs = ReadTB(plot['tensorboard'])

        for jdx, set_dfn in enumerate(plot['sets']):
            iColor = 0
            for log_key in all_logs:
                if set_dfn['name'] in all_logs[log_key]:
                    set_value = all_logs[log_key][set_dfn['name']]
                    values = list(map(lambda x: x.value, set_value))
                    step = list(map(lambda x: x.step, set_value))
                    ax.scatter(step,values, marker=set_dfn['marker'], color=plot['colors'][iColor], s=set_dfn['size'], alpha=set_dfn['alpha'])
                    ax.set_yscale('log')
                    iColor += 1
        if 'yscale' in plot and plot['yscale'] is not None:
            plt.ylim(plot['yscale'])

    # for idx, setname  in enumerate(sets):
    #     ax = fig.add_subplot(len(sets), 1, idx+1)
    #     tile = "{0}".format(setname)
    #     ax.set_title(tile, color=("green"))
    #     iColor = 0
    #     for log_key in all_logs:
    #         if setname in all_logs[log_key]:
    #             set_value = all_logs[log_key][setname]
    #             values = list(map(lambda x: x.value, set_value))
    #             step = list(map(lambda x: x.step, set_value))
    #             ax.scatter(step,values, marker='.', s=2.5, alpha=0.25)

    #     if yscale is not None:
    #         plt.ylim(yscale)

    fig.tight_layout()
    return fig


def main(args): 
    print(__file__)   

    fig = PlotTraining(args.fontdir, figsize=(args.page_width,4))
    fig.set_figwidth(args.column_width)
    fig.savefig('training.pdf', format="pdf", bbox_inches="tight")
    return 0



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