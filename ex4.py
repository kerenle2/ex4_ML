from gcommand_loader import GCommandLoader
import torch

class CNN(torch.nn.Module):
    def __init__(self,image_size):
        super(CNN, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 1000)
        self.fc1 = torch.nn.Linear(1000, 50)
        self.fc2 = torch.nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.F.relu(self.fc0(x))
        x = torch.F.relu(self.fc1(x))
        x = torch.F.relu(self.fc2(x))
        return torch.F.log_softmax(x)

# define all hyper-parameters here:
learning_rate = 0.005
num_epochs = 50
weight_decay = 0

def load_data():
    train_set = GCommandLoader('./data/train')
    valid_set = GCommandLoader('./data/train')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)
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


def train_one_epoch(x,model,optimizer):
    model.train()
    for batch_idx, (example, label) in enumerate(x):
        optimizer.zero_grad()
        output = model(example)
        loss = torch.F.nll_loss(output, label)
        loss.backward()
        optimizer.step()

    return epoch_loss


if __name__ == "__main__":
    train_set, valid_set = load_data()
    # train the model:
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(train_set, model, optimizer)
