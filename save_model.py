from model import Model

def save_model(model, model_name):
    print("Saving trained model to models/{}".format(model_name))
    model.save_to('models/{}'.format(model_name))