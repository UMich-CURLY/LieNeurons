import os
import matplotlib.pyplot as plt
import torch


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class Visualizer():
    def __init__(self, dir = 'png/'):
        makedirs(dir)
        self.dir = dir
        self.fig = plt.figure(figsize=(12, 4), facecolor='white')
        self.ax_traj = self.fig.add_subplot(121, frameon=False)
        self.ax_phase = self.fig.add_subplot(122, frameon=False)
        plt.show(block=False)

        self.fig3d = plt.figure(figsize=(12, 4), facecolor='white')
        self.ax_traj3d = self.fig3d.add_subplot(121, frameon=False)
        self.ax_phase3d = self.fig3d.add_subplot(122, projection='3d')

        

    def Plot_XY_Tra(self,true_y :torch.Tensor, pred_y: torch.Tensor, t:torch.Tensor, iter = None, figurename = None, realtime_draw = False, label1 = 'True y', label2 = 'Pred y'):

        self.ax_traj.cla()
        self.ax_traj.set_title('Trajectories')
        self.ax_traj.set_xlabel('t')
        self.ax_traj.set_ylabel('x,y')
        self.ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], '-',label= label1 + '[0]', color='mediumslateblue')
        self.ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], '-',label= label1 + '[1]', color='orange')

        self.ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--',label=label2 + '[0]', color='mediumslateblue')
        self.ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], '--',label=label2 + '[1]', color='orange')

        self.ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        # self.ax_traj.set_ylim(-2, 2)
        self.ax_traj.legend()

        self.ax_phase.cla()
        self.ax_phase.set_title('Phase Portrait')
        self.ax_phase.set_xlabel('x')
        self.ax_phase.set_ylabel('y')
        self.ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-',label=label1)
        self.ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--',label=label2)
        # ax_phase.set_xlim(-2, 2)
        # ax_phase.set_ylim(-2, 2)
        self.ax_phase.axis('equal')
        self.ax_phase.legend()

        self.fig.tight_layout()
        
        png_title = self.dir 

        if figurename is not None:
            png_title += figurename
        if iter is not None:
            png_title += '{:03d}'.format(iter)
        if figurename is None and iter is None:
            png_title += 'untitle'
        png_title += '.pdf'
        self.fig.savefig(png_title)

        if realtime_draw:
            self.fig.canvas.draw()
            plt.pause(0.001)

    def Plot_XYZ_Tra(self,true_y :torch.Tensor, pred_y: torch.Tensor, t:torch.Tensor, iter = None, figurename = None, realtime_draw = False, label1 = 'True y', label2 = 'Pred y'):


        self.ax_traj3d.cla()
        self.ax_traj3d.set_title('Trajectories')
        self.ax_traj3d.set_xlabel('t')
        self.ax_traj3d.set_ylabel('x,y,z')
        self.ax_traj3d.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], '-',label= label1 + '[0]', color='mediumslateblue')
        self.ax_traj3d.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], '-',label= label1 + '[1]', color='orange')
        self.ax_traj3d.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 2], '-',label= label1 + '[2]', color='tomato')

        self.ax_traj3d.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--',label=label2 + '[0]', color='mediumslateblue')
        self.ax_traj3d.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], '--',label=label2 + '[1]', color='orange')
        self.ax_traj3d.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 2], '--',label=label2 + '[2]', color='tomato')

        self.ax_traj3d.set_xlim(t.cpu().min(), t.cpu().max())
        # self.ax_traj.set_ylim(-2, 2)
        self.ax_traj3d.legend()

        self.ax_phase3d.cla()
        self.ax_phase3d.set_title('3D Trajecory')
        self.ax_phase3d.set_xlabel('x')
        self.ax_phase3d.set_ylabel('y')
        self.ax_phase3d.set_zlabel('z')
        self.ax_phase3d.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], true_y.cpu().numpy()[:, 0, 2], 'g-',label=label1)
        self.ax_phase3d.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], pred_y.cpu().numpy()[:, 0, 2], 'b--',label=label2)
        # ax_phase.set_xlim(-2, 2)
        # ax_phase.set_ylim(-2, 2)
        # ax_phase.set_zlim(-2, 2)

        self.ax_phase3d.axis('equal')
        self.ax_phase3d.legend()

        self.fig3d.tight_layout()
        
        png_title = self.dir 

        if figurename is not None:
            png_title += figurename
        if iter is not None:
            png_title += '{:03d}'.format(iter)
        if figurename is None and iter is None:
            png_title += 'untitle'
        png_title += '.pdf'
        self.fig3d.savefig(png_title)

        if realtime_draw:
            self.fig.canvas.draw()
            plt.pause(0.001)


