import torchvision
import torchvision.transforms as transforms
import torch,numpy
from sys import getsizeof
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=60000,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=10000,
                                          shuffle=False)

sequence_length = 28
input_size = 28

for i, (images, labels) in enumerate(train_loader):
    raw_train_X,raw_train_target = images,labels

print(raw_train_X.view(60000,28,28).shape)
print(raw_train_target.shape)

#train_X_matrix = raw_train_X.numpy()
#train_y_vector = raw_train_target.numpy()

#print(train_X_matrix[0,:,:].shape)
#import matplotlib.pyplot as plt
#plt.gray()
#plt.matshow(train_X_matrix[0,:,:].reshape((28,28)))
#plt.show()
#exit()
#for i, (images, labels) in enumerate(test_loader):
#    raw_test_X,raw_test_target = images,labels

# don't use getsizeof cuz numpy array is reference to memory
#print(train_X_matrix.size * train_X_matrix.dtype.itemsize/(1024*1024))
#print(train_y_vector.size*train_y_vector.dtype.itemsize/(1024*1024))

#raw_data = {"train_X_matrix":raw_train_X,"train_y_vector":raw_train_target}

for i, (images, labels) in enumerate(train_loader):
    images = images.view(-1, sequence_length, input_size)
    labels = labels
print(images.shape)
print(labels.shape)

raw_data = {"train_X_matrix":raw_train_X,"train_y_vector":raw_train_target}


# for both digits data and mnist data : extract a class-balanced subset of points

def subset_dataset(dataset,size_per_class):
    # dataset
    X = dataset["input"]
    y= dataset["target"]
    num_classes = len(numpy.unique(y))
    final_indices = []
    for i in range(num_classes):
        index_to_choose_from = [index for index in range(len(y)) if y[index]==i]
        assert len(index_to_choose_from)>=size_per_class
        final_indices+=numpy.random.choice(index_to_choose_from,size=size_per_class,replace=False).tolist()

    outX = X[final_indices,:]
    outy = y[final_indices]

    out = {"input":outX,"target":y}
    return(out)