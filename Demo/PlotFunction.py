import os
import matplotlib.pyplot as plt
import torch


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class Visualizer():
    def __init__(self, dir = 'figures/'):
        makedirs(dir)
        self.dir = dir
        self.fig = plt.figure(figsize=(12, 4), facecolor='white')
        self.ax_traj = self.fig.add_subplot(121, frameon=False)
        self.ax_phase = self.fig.add_subplot(122, frameon=False)
        plt.show(block=False)

    def Plot_XY_Tra(self,true_y :torch.Tensor, pred_y: torch.Tensor, t:torch.Tensor, iter = None, figurename = None, realtime_draw = False):

        self.ax_traj.cla()
        self.ax_traj.set_title('Trajectories')
        self.ax_traj.set_xlabel('t')
        self.ax_traj.set_ylabel('x,y')
        self.ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], '-',label='True y[0]', color='mediumslateblue')
        self.ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], '-',label='True y[1]', color='orange')
        self.ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--',label='Pred y[0]', color='mediumslateblue')
        self.ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], '--',label='Pred y[1]', color='orange')
        self.ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        # self.ax_traj.set_ylim(-2, 2)
        self.ax_traj.legend()

        self.ax_phase.cla()
        self.ax_phase.set_title('Phase Portrait')
        self.ax_phase.set_xlabel('x')
        self.ax_phase.set_ylabel('y')
        self.ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-',label='True')
        self.ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--',label='Pred')
        # ax_phase.set_xlim(-2, 2)
        # ax_phase.set_ylim(-2, 2)
        self.ax_phase.axis('equal')
        self.ax_phase.legend()

        self.fig.tight_layout()
        
        figures_title = self.dir 

        if figurename is not None:
            figures_title += figurename
        if iter is not None:
            figures_title += '{:03d}'.format(iter)
        if figurename is None and iter is None:
            figures_title += 'untitle'
        figures_title += '.pdf'
        plt.savefig(figures_title)

        if realtime_draw:
            self.fig.canvas.draw()
            plt.pause(0.001)