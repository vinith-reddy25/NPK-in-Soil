class MyModel(nn.Module):
 def __init__(self, input_size, output_size ):
 super(MyModel, self).__init__()
 self.ins = input_size
 self.ops = output_size
self.hidden_layer_size = 50
self.layer1 = nn.Linear(self.ins, self.hidden_layer_size)
 self.activationS = nn.Sigmoid()
 #Activation function
 self.relu = nn.ReLU()
 self.layer2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
 self.layer3 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
 self.layer4 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
 self.layer5 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
 self.layer6 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
 self.layer7 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
 self.layer8 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
 self.layer9 = nn.Linear(self.hidden_layer_size, self.ops)
def forward(self,X):
 output = self.layer9(self.layer8(self.relu(self.layer7(self.relu(self.layer6(self.
 # print (X.type(), self.layer.weight.type())
 return output