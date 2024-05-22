import matplotlib.pyplot as plt

from IMU.dataload import GroudtruethDataset

gtdata = GroudtruethDataset(csv_file="data/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv")
gtdata.convert_time_ns_to_s(set_init_time_to_zero=True)

gtdatanp = gtdata.get_data()
plot_len = 10000

fig, ax = plt.subplots(3,1)
ax[0].plot(gtdatanp[:plot_len,0], gtdatanp[:plot_len,1], label='px')
ax[1].plot(gtdatanp[:plot_len,0], gtdatanp[:plot_len,2], label='py')
ax[2].plot(gtdatanp[:plot_len,0], gtdatanp[:plot_len,3], label='pz')

ax[0].set_title('Position')
ax[2].set_xlabel('time (s)')
ax[0].set_ylabel('px')
ax[0].legend()

fig.tight_layout()
plt.show(block=False)

