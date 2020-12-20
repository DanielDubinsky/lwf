from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import torch
from src.models.utils import load_model, get_dataset
import click
from tqdm import tqdm
import seaborn as sns

@click.command()
@click.argument('model', type=click.Path(exists=True))
@click.argument('ds', type=click.STRING)
@click.argument('out', type=click.STRING)
@click.option('-d', type=click.STRING, default='cpu', help='device')
def tsne_model(model, ds, out, d):
    device = torch.device(d)
    net = load_model(model).to(device)
    ds = get_dataset(ds, train=False)
    dl = torch.utils.data.DataLoader(ds, 64, shuffle=False, num_workers=1)
    features = []
    labels = []
    with torch.no_grad():
        for d in tqdm(dl):
            image, label = d
            _, f = net(image.to(device))
            labels.append(label)
            features.append(f[3])

    features = torch.cat(features).cpu()
    labels = torch.cat(labels)
    strings = list(map(lambda x: ds.classes[x], labels))
    tsne = TSNE(n_jobs=16)
    Y = tsne.fit_transform(features)
    vis_x = Y[:, 0]
    vis_y = Y[:, 1]
    # plt.scatter(vis_x, vis_y, c=labels, marker='.')
    # plt.clim(-0.5, 9.5)
    # plt.legend(ds.classes)
    # plt.savefig('bla.png')
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=vis_x, y=vis_y,
        hue=strings,
        palette=sns.color_palette("hls", 100),
        legend="full",
        alpha=0.3
    )
    plt.savefig(out)

if __name__ == '__main__':
    tsne_model()
