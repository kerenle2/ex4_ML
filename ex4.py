from gcommand_loader import GCommandLoader
import torch
import torch.nn.functional as F
# define all hyper-parameters here:
learning_rate = 0.005
num_epochs = 50



class CNN(torch.nn.Module):
    # input channel = 1
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # x = x.view(-1, 18 * 16 * 16)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)

        return x


def load_data():
    train_set = GCommandLoader('./data/train')
    valid_set = GCommandLoader('./data/valid')
    # test_set = GCommandLoader('./data/test')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=100, shuffle=True,
        num_workers=0, pin_memory=True, sampler=None)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=100, shuffle=True,
        num_workers=0, pin_memory=True, sampler=None)
    return train_loader, valid_loader


def test(x):
    model.eval()
    test_loss = 0
    correct = 0
    for example, label in x:
        output = model(x)
        test_loss += torch.F.nll_loss(output, torch.target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(torch.target.data.view_as(pred)).cpu().sum()
    test_loss /= len(x.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(x.dataset),
        100. * correct / len(x.dataset)))


def train_one_epoch(x, model, optimizer):
    for batch_idx, (example, label) in enumerate(x): #not working!!!!!!!!!!!!!1
        # optimizer.zero_grad()
        output = model(example)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()

    return epoch_loss


if __name__ == "__main__":
    train_loader, valid_loader = load_data()
    # train the model:

    model = CNN()
    loss = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(train_loader, model, optimizer)
