import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

color = '#2288ee'
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['text.color'] = color
plt.rcParams["legend.edgecolor"] = color
plt.rcParams['figure.facecolor'] = '#22222200'
plt.rcParams['axes.edgecolor'] = color
plt.rcParams['axes.labelcolor'] = color
plt.rcParams['axes.titlecolor'] = color
plt.rcParams['xtick.color'] = color
plt.rcParams['ytick.color'] = color
plt.rcParams['legend.borderpad'] = 0.3


def Plot_Classification(model,x_train,y_train,weights_history,history):
    fig = plt.figure(figsize=(12,6))
    ax1 = plt.subplot2grid((16,16),(0,0), colspan=7, rowspan=13)
    colors = np.array(['#ff0000','#0000ff'])
    y_pred = np.argmax(model.predict(x_train), axis=-1)
    ax1.scatter(x_train[:,0],x_train[:,1],s=30,facecolor=colors[y_pred],edgecolor='w',zorder = 1)
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    ax1.set_title('Model classification')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    z_pred = model.predict(xy)
    Z = (z_pred[:,0] - z_pred[:,1]).reshape(XX.shape)
    ax1.contourf(XX, YY, Z, cmap='coolwarm', vmin=-1, vmax=1,levels=np.linspace(-1,1,15),zorder = 0)

    n_epochs = len(history.history['loss'])

    ax2 = plt.subplot2grid((16,16),(0,9), colspan=7, rowspan=13)
    ax2.plot(history.history['loss'],ls='--')
    ax2.plot(history.history['val_loss'],ls='--')
    ax2.scatter(0,history.history['loss'][0],c='blue')
    ax2.scatter(0,history.history['val_loss'][0],c='orange')
    ax2.set_title('Model Loss / Epoch')
    ax2.set_ylabel('Categorical_Cross_Entropy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['training', 'validation'], loc='upper right')

    dataSlider_ax = plt.subplot2grid((16,16),(15,0), colspan=16, rowspan=1,facecolor='lightgray')
    dataSlider = Slider(ax=dataSlider_ax,label='epoch',valmin=0,valmax= n_epochs - 1, valinit=0,valstep=1,visible = False)
    dataSlider_ax.set_xticks(np.linspace(0,n_epochs - 1,15,dtype=int))
    dataSlider_ax.xaxis.set_visible(True)
    dataSlider_ax.plot([0,0],[0,20],linewidth = 10,color='blue')

    def update(val):
        ax1.collections.clear()
        model.set_weights(weights_history[dataSlider.val])
        y_pred = np.argmax(model.predict(x_train), axis=-1)
        ax1.scatter(x_train[:,0],x_train[:,1],s=30,facecolor=colors[y_pred],edgecolor='w',zorder = 1)
        z_pred = model.predict(xy)
        Z = (z_pred[:,0] - z_pred[:,1]).reshape(XX.shape)
        ax1.contourf(XX, YY, Z, cmap='coolwarm', vmin=-1, vmax=1,levels=np.linspace(-1,1,15),zorder = 0)
        

        ax2.collections.clear()
        ax2.scatter(dataSlider.val,history.history['loss'][dataSlider.val],c='blue')
        ax2.scatter(dataSlider.val,history.history['val_loss'][dataSlider.val],c='orange')

        dataSlider_ax.lines.clear()
        dataSlider_ax.plot([dataSlider.val,dataSlider.val],[0,20],linewidth = 10,color='blue')

        fig.canvas.draw_idle()
            

    dataSlider.on_changed(update)

    plt.show()