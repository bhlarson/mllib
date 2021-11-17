# %%
import sys, os
import json
import argparse
import pandas as pd
import numpy as np
from IPython.display import Markdown, display
import ipywidgets as widgets
import plotly
import plotly.graph_objects as graph_obj
import plotly.figure_factory as ff
import plotly.express as px

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
from utils.s3 import s3store, Connect
from utils.jsonutil import WriteDictJson, ReadDictJson

# %%
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')

    parser.add_argument('-model_type', type=str,  default='segmentation')
    parser.add_argument('-test_results', type=str, default='test_results.json')

    parser.add_argument('-trainingset', type=str, default='data/coco/annotations/instances_train2017.json', help='Coco dataset instance json file.')
    parser.add_argument('-validationset', type=str, default='data/coco/annotations/instances_val2017.json', help='Coco dataset instance json file.')
    parser.add_argument('-train_image_path', type=str, default='data/coco/train2017', help='Coco image path for dataset.')
    parser.add_argument('-val_image_path', type=str, default='data/coco/val2017', help='Coco image path for dataset.')
    parser.add_argument('-class_dict', type=str, default='model/deeplabv3/coco.json', help='Model class definition file.')

    parser.add_argument('-epochs', type=int, default=4, help='Training epochs')

    args = parser.parse_args("")
    return args

# %%
def PrepareResults(test_data):
    test_names = []
    overview = {}
    results = {}
    model_dict = {}
        
    for test in test_data:
        if 'name' in test:
            name = '{}: {}'.format(len(test_names)+1,test['name'])
        elif 'model_src' in test['config']:
            name = '{}: {}'.format(len(test_names)+1,test['config']['model_src'])
        else: 
            name = '{}: {}'.format(len(test_names)+1,test['date'])
        test_names.append(name)

        if test['config']['train']:
            model = test['config']['model_dest']
        else:
            model = test['config']['model_src']

        description = ''
        if 'description' in test['config']:
            if 'description' in test['config']['description']:
                description = test['config']['description']['description']

        
        test_overview = {
            'model type': test['config']['model_type'],
            'model class': test['config']['model_class'],
            'model': model,
            'test date': test['date'],
            'test images':test['results']['num images'],
            'mean IoU':test['results']['mean intersection over union'], 
            'inference time':test['results']['average time'],
            'description': description
            }
        overview[name]=test_overview

        similarity = {}
        for object in test['objects']:
            res = test['results']['similarity'][object]
            if res:
                similarity[test['objects'][object]['name']] = res['similarity']

        results[name] = {'objects':test['objects'], 'similarity':similarity, 'confusion':test['results']['confusion']}

        dict_key = '{} {}'.format(test['config']['model_type'], test['config']['model_class'])
        if dict_key not in model_dict:
            model_dict[dict_key] = []

        model_dict[dict_key].append(test)


    return test_names, overview, results, model_dict

#%%
def PlotConfusion(objects, confusion_matrix, colorscale='plasma'):
    c = np.array(confusion_matrix)
    norm_confusion = (c.T / c.astype(np.float).sum(axis=1)).T
    norm_confusion = norm_confusion.round(decimals=3)

    confusion_text = [[str(round(y, 3)) for y in x] for x in norm_confusion]

    classNames = [value['name'] for value in objects.values()]

    # set up figure 
    confusion_plot = plotly.figure_factory.create_annotated_heatmap(norm_confusion, x=classNames, y=classNames, colorscale=colorscale)

    # add custom xaxis title
    confusion_plot.add_annotation(dict(font=dict(color="black",size=16),
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    text="Predicted value",
                    xref="paper",
                    yref="paper"))

    # add custom yaxis title
    confusion_plot.add_annotation(dict(font=dict(color="black",size=16),
                    x=-0.2,
                    y=0.5,
                    showarrow=False,
                    text="Real value",
                    textangle=-90,
                    xref="paper",
                    yref="paper"))


    # adjust margins to make room for yaxis title
    confusion_plot.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    confusion_plot['data'][0]['showscale'] = True
    confusion_plot.update_layout(yaxis_autorange="reversed")

    return confusion_plot

def UpdateConfusion(plot, data):
    with plot.batch_update():
        
        c = np.array(data)
        norm_confusion = (c.T / c.astype(np.float).sum(axis=1)).T
        confusion_text = [str(round(x, 3)) for x in norm_confusion.flatten()]
        plot.data[0].z =  norm_confusion
        plot.update_layout(yaxis_autorange="reversed")
        
        for i in range(len(confusion_text)):
            plot.layout.annotations[i].text = confusion_text[i]

def PlotModels(models):
    data = []
    x=[]

    columns = []
    labels = {'x':'model', 'y':'similarity', 'title':''}

    if len(models)>0:

        columns.append('mean IoU')
        objects = models[0]['objects']
        for key, value in objects.items():
            columns.append(value['name'])

        for model in models:
            results = model['results']
            if model['config']['train']:
                model_name = model['config']['model_dest']
            else:
                model_name = model['config']['model_src']
            x.append(model_name)
            row = []
            row.append(results['mean intersection over union'])
            for value in results['similarity'].values():
                row.append(value['similarity'])



            data.append(row)
            
    df = pd.DataFrame(data, columns =columns)

    fig = px.line(df, x=x , y=columns , labels=labels)

    return fig

#%%            
def ClearOutput(b, output):
    output.clear_output()

# %%
def SelectTest(change, output, select, display, results):
    with output:
        print('SelectTest change={}',format(change))
    
        with display.batch_update():
            
            c = np.array(results[select.value]['confusion'])
            norm_confusion = (c.T / c.astype(np.float).sum(axis=1)).T
            confusion_text = [str(round(x, 3)) for x in norm_confusion.flatten()]
            display.data[0].z =  norm_confusion
            display.update_layout(yaxis_autorange="reversed")
            
            for i in range(len(confusion_text)):
                display.layout.annotations[i].text = confusion_text[i]

# %%
def SelectModel(change, output, select, display, results):
    with output:
        print('SelectModel change={}',format(change))
    
        with display.batch_update():
            
            c = np.array(results[select.value]['confusion'])
            norm_confusion = (c.T / c.astype(np.float).sum(axis=1)).T
            confusion_text = [str(round(x, 3)) for x in norm_confusion.flatten()]
            display.data[0].z =  norm_confusion
            display.update_layout(yaxis_autorange="reversed")
            
            for i in range(len(confusion_text)):
                display.layout.annotations[i].text = confusion_text[i]

#%%
def main(args):
    print('segmenttest args:{}'.format(args))

    s3, creds, s3def = Connect(args.credentails)

    test_path = '{}/{}/{}'.format(s3def['sets']['test']['prefix'], args.model_type, args.test_results)
    test_data = s3.GetDict(s3def['sets']['test']['bucket'], test_path)

    test_names, overview, results, model_dict = PrepareResults(test_data)
    display(pd.DataFrame(overview).T)
    print(pd.DataFrame(overview).T)

    if len(test_data) > 0:
        plot = PlotConfusion(test_data[0]['objects'], test_data[0]['results']['confusion'])
        confusion_display = graph_obj.FigureWidget(plot)
        display(confusion_display)
        confusion_display.show()

    if len(model_dict) > 0:
        model_key, model_values = list(model_dict.items())[0]
        if len(model_values) > 0:

            modelsplot = PlotModels(model_values)
            models_display = graph_obj.FigureWidget(modelsplot)
            display(models_display)
            models_display.show()

    test_select = widgets.Select(
        options=test_names,
        description='Test:',
        #rows=25,
    )
    test_select.observe (lambda change:SelectTest(change, 
                                                  output=output, 
                                                  select=test_select, 
                                                  display=confusion_display, 
                                                  results=results), 
                        names="value")
    model_select = widgets.Select(
        options=model_dict.keys(),
        description='Model:',
        #rows=25,
    )
    model_select.observe (lambda change:ModelsPlot(change, 
                                                    output=output, 
                                                    select=model_select, 
                                                    display=confusion_display, 
                                                    results=results), 
                        names="value")

    output = widgets.Output()
    clear_output = widgets.Button(description='Clear Output')
    clear_output.on_click(lambda b: ClearOutput(b, output=output))

    display_results = widgets.HBox([test_select, confusion_display])
    segment_interactive = widgets.VBox([display_results, clear_output, output])

    if len(test_names) > 0:
        test_select.options = test_names
        test_select.value = test_names[0]

    display(segment_interactive)

# %%
if __name__ == '__main__':
    import argparse
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy
        # https://code.visualstudio.com/docs/python/debugging#_remote-debugging

        debugpy.listen(address=(args.debug_address, args.debug_port))
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")

    # %%
    main(args)

