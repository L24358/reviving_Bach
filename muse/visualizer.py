import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def show(Melody_Snippet):
    img = mpimg.imread(str(Melody_Snippet.write("lily.png")))
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()
    
def plot_stdlog(stdlog, figname, title='', ylabel='std', save=True,
                show=True, figsize=(6,4)):
    '''
    Plots the std log.
    '''
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.plot(stdlog, 'k.', markersize=1)
    ax1.set_xlabel('Epochs', fontsize=12); ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title('Std Log', fontsize=14, fontweight='bold')
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    plt.suptitle(title); plt.tight_layout()
    if save: plt.savefig(figname, dpi=300)
    if show: plt.show(); plt.clf()
    
def plot_train_val_loss(train_losses, val_losses, figname,\
                        title='Loss', ylabel='CE Loss', save=True,\
                        labels=['training', 'validation'],\
                        figsize=(6,4), blank=False, show=True):
    '''
    Plots the training and validation loss
    '''
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.plot(train_losses, 'r', label=labels[0])
    ax1.plot(val_losses, 'b', label=labels[1])
    if not blank:
        ax1.set_xlabel('Epochs', fontsize=12); ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold'); ax1.legend()
        ax1.spines.right.set_visible(False)
        ax1.spines.top.set_visible(False)
    else:
        ax1.axis('off')
    plt.tight_layout()
    if save: plt.savefig(figname, dpi=300)
    if show: plt.show(); plt.clf()

def plot_train_val_sample(model, sample_train, sample_val, figname, title='',\
                          save=True, figsize=(8,4), blank=False, show=True):
    '''
    Plots the training and validation sample
    '''
    model.eval()
    recons_train = model(sample_train)[0]
    recons_val = model(sample_val)[0]
    
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.plot(sample_train.view(-1), 'k', label='input')
    ax1.plot(recons_train.detach().numpy().reshape(-1), 'g', label='reconstruction')
    ax2.plot(sample_val.view(-1), 'k', label='input')
    ax2.plot(recons_val.detach().numpy().reshape(-1), 'g', label='reconstruction')
    
    if not blank:
        ax1.set_xlabel('Interval', fontsize=12); ax1.set_ylabel('Melody', fontsize=12)
        ax1.set_title('Training', fontsize=14, fontweight='bold'); ax1.legend()
        ax2.set_xlabel('Interval', fontsize=12); ax2.set_ylabel('Melody', fontsize=12)
        ax2.set_title('Validation', fontsize=14, fontweight='bold'); ax2.legend()
        ax1.spines.right.set_visible(False)
        ax1.spines.top.set_visible(False)
        ax2.spines.right.set_visible(False)
        ax2.spines.top.set_visible(False)
        plt.suptitle(title)
    else:
        ax1.axis('off'); ax2.axis('off')
        
    plt.tight_layout()
    if save: plt.savefig(figname, dpi=300)
    if show: plt.show(); plt.clf()
    return recons_train, recons_val
    
def plot_gen_sample(new, figname,
                    save=True, figsize=(4,4), blank=False, show=False):
    '''
    Plots the newly generated sample
    '''
    
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.plot(new, color='darkgreen')
    
    if not blank:
        ax1.set_xlabel('Interval', fontsize=12); ax1.set_ylabel('Melody', fontsize=12)
        ax1.set_title('Generation', fontsize=14, fontweight='bold')
        ax1.spines.right.set_visible(False)
        ax1.spines.top.set_visible(False)
    else:
        ax1.axis('off')
        
    plt.tight_layout()
    if save: plt.savefig(figname, dpi=300)
    if show: plt.show(); plt.clf()