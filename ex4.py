from gcommand_loader import GCommandLoader
import torch
import torch.nn.functional as F
# define all hyper-parameters here:
learning_rate = 0.005
num_epochs = 5
input_channels = 1
batch_size = 100


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 10, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)

        self.fc1 = torch.nn.Linear(20000, 100)
        self.fc2 = torch.nn.Linear(100, 30)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        ###
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 20 * 40 * 25)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def load_data():
    train_set = GCommandLoader('./data/train')
    valid_set = GCommandLoader('./data/valid')
    test_set = GCommandLoader('./data/test')
    test_file_names = []
    for i in range(len(test_set)):
        name = (test_set.spects)[i][0].split("\\")[2]
        test_file_names.append(name)


    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_set,batch_size=batch_size, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    return train_loader, valid_loader, test_loader, test_file_names


def test(x):
    model.eval()
    test_loss = 0
    correct = 0
    for example, label in x:
        output = model(example)
        test_loss += F.nll_loss(output, label, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    test_loss /= len(x.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(x.dataset),
    #     100. * correct / len(x.dataset)))
    str = 'Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(x.dataset),
        100. * correct / len(x.dataset))
    return str


def train_one_epoch(x, model, optimizer, loss_func):
    for batch_idx, (example, label) in enumerate(x):
        optimizer.zero_grad()
        output = model(example)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, test_file_names = load_data()

    # train the model:
    model = CNN()
    loss = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        train_one_epoch(train_loader, model, optimizer, loss)
        train_acc_str = test(train_loader)
        valid_acc_str = test(valid_loader)
        print('epoch: ', epoch)
        print('train: ', train_acc_str)
        print('validaition: ', valid_acc_str)

    # predict for test:
    model.eval()
    file = open('test_y', 'w+')
    for example, label in test_loader:
        output = model(example)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        for i in range(len(pred.data)):
            file.write(test_file_names[i] + ', ' + str(pred.data[i].item()) + '\n')

    file.close()


