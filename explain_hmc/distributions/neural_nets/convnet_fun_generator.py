from abstract.abstract_class_V import V
import torch.nn as nn
def convnet_fun_generator(dataset,problem_type,num_units_list,activations_list,is_skip_connections):
    X = dataset["input"]
    target = dataset["target"]
    if problem_type == "regression":
        criterion = nn.MSELoss
        output_dim = 1
    elif problem_type == "classification":
        criterion = nn.NLLLoss
        output_dim = num_unique_classes(target)
    class ConvNet(V):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(7 * 7 * 32, output_dim)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            loss = criterion(out, target)
            return(loss)
    return(ConvNet)