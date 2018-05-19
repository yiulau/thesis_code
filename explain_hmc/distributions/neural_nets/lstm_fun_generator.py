import torch,numpy
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from general_util.memory_util import to_pickle_memory
from abstract.abstract_class_V import V

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 3
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

def lstm_fun_generator(dataset,problem_type,num_units_list,activations_list,is_skip_connections):
    X = dataset["input"]
    target = dataset["target"]
    if problem_type == "regression":
        criterion = nn.MSELoss
        output_dim = 1
    elif problem_type == "classification":
        criterion = nn.NLLLoss

        output_dim = len(numpy.unique(target))
# Recurrent neural network (many-to-one)
    class RNN(V):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(V, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            # Set initial hidden and cell states
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

            # Decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            loss = criterion(out, target)
            return(loss)
    return(RNN)
# for i, (images, labels) in enumerate(train_loader):
#     images = images.view(-1, sequence_length, input_size)
#     labels = labels
# print(images.shape)
# print(labels.shape)

model = RNN(input_size, hidden_size, num_layers, num_classes)
mb_size = to_pickle_memory(model)
print(mb_size)
exit()
#print(model.named_parameters())
from torch.autograd import Variable
indata = images[:100,:,:]
indata = Variable(indata,requires_grad=False)
labels = Variable(labels)
#model(indata)
#exit()
criterion = nn.CrossEntropyLoss()
import time
for i in range(25):
    start = time.time()
    out = model(indata)

    loss = criterion(out, labels)
    loss.backward()
    end_time = time.time()-start
    print(end_time)
#print(out)
exit()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')