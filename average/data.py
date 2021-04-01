import default


def thiry_data_items(title, model, filename, **para):
    print("--- %s  ---" % (title))
    values = list(para.values())
    if len(values) > 0:
        data, df = model.loss(len(values[0]), **para)
        df.to_csv(filename, index=False)

def collect(lr, batch_size, num_epoches, model, para, filename):
    if lr:
        para = {"lr" : para}
        thiry_data_items("learning rate", model, filename, **para)

    if batch_size:
        para = {"batch_size" : para}
        thiry_data_items("batch size", model, filename, **para)
    
    if batch_size:
        para = {"num_epoches" : para}
        thiry_data_items("Number of epoches", model, filename, **para)

def newton(lr, batch_size, num_epoches):
    if lr:
        para = {"lr" : [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}
        thiry_data_items("learning rate", default.NewtonsLoss(), "../data/boxplots/newton/lr.csv", **para)

    if batch_size:
        para = {"batch_size" : [50, 100, 150, 200, 250, 300]}
        thiry_data_items("batch size", default.NewtonsLoss(), "../data/boxplots/newton/batch_size.csv", **para)

    if num_epoches:
        para = {"num_epoches" : [50, 100, 150, 200, 250, 300]}
        thiry_data_items("num epoches", default.NewtonsLoss(), "../data/boxplots/newton/num_epoches.csv", **para)

def van(lr, batch_size, num_epoches):
    if lr:
        para = {"lr" : [0.5, 0.005, 0.0005, 0.00005, 0.000005, 0.0000005]}
        thiry_data_items("learning rate", default.VanDerPol(), "../data/boxplots/van/lr.csv", **para)

    if batch_size:
        para = {"batch_size" : [15, 20, 25, 30, 35, 40]}
        thiry_data_items("batch size", default.VanDerPol(), "../data/boxplots/van/batch_size.csv", **para)

    if num_epoches:
        para = {"num_epoches" : [50, 100, 150, 200, 250, 300]}
        thiry_data_items("num epoches", default.VanDerPol(), "../data/boxplots/van/num_epoches.csv", **para)



if __name__== "__main__":
    pass