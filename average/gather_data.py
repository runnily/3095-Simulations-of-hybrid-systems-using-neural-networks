"""
    Arthur: Adanna Obibuaku
    Purpose: This module is used for gathering data. That is 30 data plots to be later used for
             boxplots.
    Date:   29/03/21
"""

import default

TITLE_LR = "Learning Rate"
TITLE_BATCH_SIZE = "Batch Size"
TITLE_NUM_EPOCHES = "Number Of Epoches"
TITLE_TIME_STEP = "Time Step"
TITLE_CUSTOME = "Custome items"

HIDDEN_LAYERS_0 = [50, 75, 100, 125, 150, 175, 200, 225]
HIDDEN_LAYERS_1 = [50, 250, 350, 450, 550, 650, 750, 850]
HIDDEN_LAYERS_2 = [600, 800, 100, 1000, 2]
HIDDEN_LAYERS_3 = [600, 800, 1000]

HIDDEN_LAYERS = [HIDDEN_LAYERS_0, HIDDEN_LAYERS_1, HIDDEN_LAYERS_2, HIDDEN_LAYERS_3]

DEFAULT_LAYER =  [{"in_features" :2, "out_features" : 50}, {"in_features" : 50, "out_features" : 100}, 
        {"in_features" : 100, "out_features" : 200},
        {"in_features" : 200, "out_features" : 400}, 
        {"in_features" : 400, "out_features" : 1}]

def thiry_data_items(title, model, filename, **para):
    """
        thirty_data_item: This denotes will gather the 30 plots data loss item corresponding to the parameter
        Args:
            title (string) : This will show the title for the plots being gathered
            model (class <LossUtilities>) : This will show the model being used
            filename (string) : This will save the filename
            ***para: This denotes the parameter that will be used.
    """
    print("--- %s  ---" % (title))
    values = list(para.values())
    if len(values) > 0:
        _, df = model.loss(len(values[0]), **para)
        df.to_csv(filename, index=False)

def append_inputs_outputs_layers(num_inputs, num_outputs):
    """
        append_inputs_outputs_layers:
            This is used to append the input and output layers
            to the hidden constant layers defined at the top
        Args:
            num_inputs (int) : The number of inputs the neural network hass
            num_outputs (int) : The number of outputs the neural network 
        Returns
            (class 'list') : A list of all the layers to test
    """
    test_layers = []
    for layer in HIDDEN_LAYERS:
            test = layer # assign test to layer
            test.insert(0, num_inputs) # insert number inputs to first postion
            test.append(num_outputs) # insert num inputs
            test_layers.append(test) # add to test layers 
    return test_layers

# This is the methods will perform the collection of data plots ...

def newton(lr=False, batch_size=False, num_epoches=False, custome=False, layer=False):
    """
        newton allows code executions.
    """
    if lr:
        para = {"lr" : [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}
        thiry_data_items(TITLE_LR, default.NewtonsLoss(),"../data/boxplots/newton/lr.csv", **para)

    if batch_size:
        para = {"batch_size" : [50, 100, 150, 200, 250, 300]}
        thiry_data_items(TITLE_BATCH_SIZE, default.NewtonsLoss(),"../data/boxplots/newton/batch_size.csv", **para)

    if num_epoches:
        para = {"num_epoches" : [50, 100, 150, 200, 250, 300]}
        thiry_data_items(TITLE_NUM_EPOCHES, default.NewtonsLoss(),"../data/boxplots/newton/num_epoches.csv", **para)
    if custome: 
        para = {"lr" : [0.000001, 0.000001, 0.001, 0.001], "batch_size" : [32, 150, 32, 150], "num_epoches": [200, 200, 200, 200]}
        thiry_data_items(TITLE_CUSTOME, default.NewtonsLoss(),"../data/boxplots/newton/custome.csv", **para)
    if layer:
        test_layers = append_inputs_outputs_layers(num_inputs=2, num_outputs=1)
        loss = default.NewtonsLoss()
        _, df = loss.loss_modeling(test_layers)
        df.to_csv("../data/boxplots/newton/layers.csv")

def van(lr = False, batch_size = False, num_epoches = False, time_step = False, custome = False, layer = False):
    if num_epoches:
        para = {"num_epoches" : [10, 20, 30,]}
        thiry_data_items(TITLE_NUM_EPOCHES, default.VanDerPol(),"../data/boxplots/van/num_epoches.csv", **para)

    if lr:
        para = {"lr" : [0.0000005]}
        thiry_data_items(TITLE_LR, default.VanDerPol(),"../data/boxplots/van/lr.csv", **para)

    if batch_size:
        para = {"batch_size" : [15, 20, 25, 30, 35, 40]}
        thiry_data_items(TITLE_BATCH_SIZE, default.VanDerPol(),"../data/boxplots/van/batch_size.csv", **para)

    if time_step:
        para = {"time_step" : [0.1, 0.01, 0.001]}
        thiry_data_items(TITLE_TIME_STEP, default.VanDerPol(),"../data/boxplots/van/time_step.csv", **para)

    if custome:
        para = {"lr" : [0.0001, 0.000001], "bs" : [35, 15], "num_epoches" : [50, 100]}
        thiry_data_items(TITLE_CUSTOME, default.VanDerPol(),"../data/boxplots/van/custome.csv", **para)
    if layer:
        test_layers = append_inputs_outputs_layers(num_inputs=3, num_outputs=2)
        loss = default.VanDerPol()
        _, df = loss.loss_modeling(test_layers)
        df.to_csv("../data/boxplots/van/layers.csv")


def laub(lr = False, batch_size = False, num_epoches = False, time_step = False, custome = False, layer = False):
    if lr:
        para = {"lr" : [0.0001, 0.00005, 0.000005, 0.0000005, 0.00000005, 0.000000005]}
        thiry_data_items(TITLE_LR, default.Laub(),"../data/boxplots/laub/lr.csv", **para)

    if batch_size:
        para = {"batch_size" : [500, 550, 600, 650, 700, 750]}
        thiry_data_items(TITLE_BATCH_SIZE, default.Laub(),"../data/boxplots/laub/batch_size.csv", **para)

    if num_epoches:
        para = {"num_epoches" : [10, 20, 30, 40, 50, 60]}
        thiry_data_items(TITLE_NUM_EPOCHES, default.Laub(),"../data/boxplots/laub/num_epoches.csv", **para)

    if time_step:
        para = {"time_step" : [0.1, 0.01]}
        thiry_data_items(TITLE_TIME_STEP, default.Laub(),"../data/boxplots/laub/time_step.csv", **para)
    
    if custome:
        para = {"lr" : [5e-8, 5e-5, 5e-8, 5e-5], "batch_size" : [30, 500, 500,30], "num_epoches" : [180, 60, 180, 60]}
        thiry_data_items(TITLE_CUSTOME, default.Laub(),"../data/boxplots/laub/custome.csv", **para)
    if layer:
        test_layers = append_inputs_outputs_layers(num_inputs=8, num_outputs=7)
        loss = default.Laub()
        _, df = loss.loss_modeling(test_layers)
        df.to_csv("../data/boxplots/van/layers.csv")


if __name__== "__main__":
    # Add data here for plots you want to collect
    print("..Collecting data..")

    newton(layer=True)
    #newton(False, False, False, True)