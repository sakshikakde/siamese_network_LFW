from common_utils import *

def predict(model, input):
  model.eval()
  with torch.no_grad():
      outputs = model(input) 
      _, pred = outputs.topk(1, 1, True)
      pred = pred.t()
  return pred