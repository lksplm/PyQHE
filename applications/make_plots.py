from pyqhe.eigensystem import Eigensystem
import pickle
import matplotlib.pyplot as plt

Params = [(2,8), (3,8), (4,10)]
ext = ".pdf"
for N, m in Params:
    savedict = pickle.load(open("../../Results/result_{:d}_{:d}.p".format(m, N), "rb" ))
    eigsys = savedict["Esys"]

    path = "../../Results/plots/{:d}_{:d}_".format(N, m)

    f1, ax = eigsys.plot_observable("E")
    #ax.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
    ax.set_title('$N_{{\\uparrow={:d} }}, N_{{\\downarrow={:d} }}, m={:d}$'.format(N, N, m))
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$E$')
    f1.savefig(path+"Energies"+ext, dpi=300)

    f2, ax2 = eigsys.plot_observable("L", Mshow=3)
    ax2.set_title('$N_{{\\uparrow={:d} }}, N_{{\\downarrow={:d} }}, m={:d}$'.format(N, N, m))
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$L$')
    f2.savefig(path + "Ltot"+ext, dpi=300)

    f3, ax3 = eigsys.plot_observable("S", Mshow=3)
    ax3.set_title('$N_{{\\uparrow={:d} }}, N_{{\\downarrow={:d} }}, m={:d}$'.format(N, N, m))
    ax3.set_xlabel(r'$\alpha$')
    ax3.set_ylabel(r'$S$')
    f3.savefig(path + "Spin"+ext, dpi=300)

    f4, ax4 = plt.subplots()
    L = eigsys.get_observable("L")  # np.empty((self.M, *self.param_shape))
    Eint = eigsys.get_observable("Eint")

    ax4.plot(L[:,1:,:].flatten(), Eint[:,1:,:].flatten(), 'o', ms=5, markerfacecolor='none')
    ax4.set_xlabel('$L$')
    ax4.set_ylabel('$E_{int}/U$')
    ax4.set_title('$N_{{\\uparrow={:d} }}, N_{{\\downarrow={:d} }}, m={:d}$'.format(N, N, m))
    f4.savefig(path + "Spectrum"+ext, dpi=300)

    del eigsys, savedict