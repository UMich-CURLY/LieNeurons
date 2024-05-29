import torch 
import matplotlib.pyplot as plt
import os





class IMU_Dynamic_Plotter:
    def __init__(self, save_path = None) -> None:
        self.save_path = save_path
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        self.fig = plt.figure(figsize=(18, 6), facecolor='white')
        self.ax_Rx = self.fig.add_subplot(331, frameon=False)
        self.ax_Ry = self.fig.add_subplot(332, frameon=False)
        self.ax_Rz = self.fig.add_subplot(333, frameon=False)
        self.ax_vx = self.fig.add_subplot(334, frameon=False)
        self.ax_vy = self.fig.add_subplot(335, frameon=False)
        self.ax_vz = self.fig.add_subplot(336, frameon=False)
        self.ax_px = self.fig.add_subplot(337, frameon=False)
        self.ax_py = self.fig.add_subplot(338, frameon=False)
        self.ax_pz = self.fig.add_subplot(339, frameon=False)


    def pltR(self, x: torch.Tensor, data: torch.Tensor, title = "Orientation", data_label_1 = "x", data_label_2 = "y", data_label_3 = "z"):
        self.ax_Rx.plot(x.cpu().numpy(), data.cpu().numpy()[:,0], label=data_label_1)
        self.ax_Ry.plot(x.cpu().numpy(), data.cpu().numpy()[:,1], label=data_label_2)
        self.ax_Rz.plot(x.cpu().numpy(), data.cpu().numpy()[:,2], label=data_label_3)

        self.ax_Rx.set_title(title)
        self.ax_Ry.set_title(title)
        self.ax_Rz.set_title(title)
        self.ax_Rz.set_xlabel('time (s)')
        self.ax_Rx.set_ylabel(data_label_1)
        self.ax_Ry.set_ylabel(data_label_2)
        self.ax_Rz.set_ylabel(data_label_3)
        self.fig.tight_layout()
        plt.show(block=False)


    def addR(self, x: torch.Tensor, data: torch.Tensor, title = "Orientation", data_label_1 = "x", data_label_2 = "y", data_label_3 = "z"):
        self.ax_Rx.plot(x.cpu().numpy(), data.cpu().numpy()[:,0], label=data_label_1)
        self.ax_Ry.plot(x.cpu().numpy(), data.cpu().numpy()[:,1], label=data_label_2)
        self.ax_Rz.plot(x.cpu().numpy(), data.cpu().numpy()[:,2], label=data_label_3)

        self.ax_Rx.set_title(title)
        self.ax_Ry.set_title(title)
        self.ax_Rz.set_title(title)
        self.ax_Rz.set_xlabel('time (s)')
        self.ax_Rx.set_ylabel(data_label_1)
        self.ax_Ry.set_ylabel(data_label_2)
        self.ax_Rz.set_ylabel(data_label_3)
        self.fig.tight_layout()
        plt.show(block=False)
    """
    To be finished
    """
    

class vec3d_31_plotter:
    def __init__(self, save_path = None) -> None:
        self.save_path = save_path
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        self.fig = plt.figure(figsize=(18, 9), facecolor='white')
        self.ax_1 = self.fig.add_subplot(311, frameon=True)
        self.ax_2 = self.fig.add_subplot(312, frameon=True)
        self.ax_3 = self.fig.add_subplot(313, frameon=True)

    def pltvec3d(self, x: torch.Tensor, data: torch.Tensor, title = "Data", data_label_1 = "x", data_label_2 = "y", data_label_3 = "z"):
        self.ax_1.plot(x.cpu().numpy(), data.cpu().numpy()[:,0], label=data_label_1)
        self.ax_2.plot(x.cpu().numpy(), data.cpu().numpy()[:,1], label=data_label_2)
        self.ax_3.plot(x.cpu().numpy(), data.cpu().numpy()[:,2], label=data_label_3)

        self.ax_1.set_title(title)
        self.ax_3.set_xlabel('time (s)')
        self.ax_1.set_ylabel("x")
        self.ax_2.set_ylabel("y")
        self.ax_3.set_ylabel("z")

        self.fig.tight_layout()
        plt.show(block=False)

    def addvec3d(self, x: torch.Tensor, data: torch.Tensor, data_label_1 = "x", data_label_2 = "y", data_label_3 = "z"):
        self.ax_1.plot(x.cpu().numpy(), data.cpu().numpy()[:,0], label=data_label_1)
        self.ax_2.plot(x.cpu().numpy(), data.cpu().numpy()[:,1], label=data_label_2)
        self.ax_3.plot(x.cpu().numpy(), data.cpu().numpy()[:,2], label=data_label_3)
        
        self.ax_1.legend()
        self.ax_2.legend()
        self.ax_3.legend()
        self.fig.tight_layout()
        plt.show(block=False)

    def savefig(self, fig_name):
        if self.save_path:
            plt.savefig(self.save_path + fig_name+".pdf")
        else:
            print("No save path is given.")



if __name__ == "__main__":
    import numpy as np
    # Generate some sample data
    data = torch.rand((100,3))
    time = torch.arange(0,100)
    save_path = "figures/test/"
    Plotter = IMU_Dynamic_Plotter(save_path=save_path)
    Plotter.pltR(time,data,title="Orientation",data_label_1="x",data_label_2="y",data_label_3="z")
    data = torch.rand((100,3))
    Plotter.addR(time,data,title="Orientation",data_label_1="x",data_label_2="y",data_label_3="z")
    plt.show()

    plotter1 = vec3d_31_plotter(save_path=save_path)
    plotter1.pltvec3d(time,data,title="Data",data_label_1="true_x",data_label_2="true_y",data_label_3="true_z")
    data = torch.rand((100,3))
    plotter1.addvec3d(time,data,data_label_1="pred_x",data_label_2="pred_y",data_label_3="pred_z")
    plotter1.savefig("Test_figure")
    plt.show()