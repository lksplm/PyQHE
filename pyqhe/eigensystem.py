from .basis import *
from .cython.hamiltonian_cy import *
import matplotlib.pyplot as plt
from tqdm import tqdm

class Observable:
    def __init__(self, name, op, eigenvalues = None):
        self.name=name
        self.op = op
        self.eigenvalues = eigenvalues

class Eigensystem:
    def __init__(self, ops_list, param_list, M=10, simult_obs=None, simult_seed=[0., ], full=False):
        """
        Computes the eigensystem of a Hailtonian for a (number of) parameter(s) of the form
        :param ops_list: [H0, H1, ...]
        :param param_list:  [[c0], [c1_0, c1_1,...]]
        :param M: number of eigenstates to compute
        :return: eigen-energies [M, N0, N1, N2...]
                 eigen-states [eigenstate.shape, M, N0, N1, N2...]
        """
        Nop = len(ops_list)
        L = ops_list[0].shape[0] #size of the hamiltonian matrix
        self.M = M
        if full:
            self.M = L

        self.ops_list = ops_list
        self.param_list = param_list
        self.param_idx = [range(len(p)) for p in self.param_list]
        self.param_shape = [len(p) for p in self.param_list]
        self.param_array = np.empty((Nop, *self.param_shape))
        self.eigenenergy = np.empty((self.M, *self.param_shape))
        if simult_obs is not None:
            assert isinstance(simult_obs, Observable), "simult_obs must be an Observable"
            simult_ev = np.empty((self.M, *self.param_shape))
            Ms = M//len(simult_seed)
            tmp_state = np.empty((L, self.M), dtype=np.complex64)
            tmp_eige = np.empty(self.M)
            eig_simult = np.empty(self.M)
        self.eigenstate = np.empty((L, self.M, *self.param_shape), dtype=np.complex64)

        for j, p in tqdm(zip(product(*self.param_idx), product(*self.param_list)), total=np.array(self.param_shape).prod()):
            H = sp.coo_matrix(self.ops_list[0].shape)
            for i in range(Nop):
                H += ops_list[i].sp * p[i]
            if simult_obs is not None: #Perform simultaneous diagonalization
                H += 1j*simult_obs.op.matrix
                if full:
                    eige_cmplx, eigs = np.linalg.eig(H.todense())
                    eige = eige_cmplx.real
                    simult_ev[(slice(None),) + j] = eige_cmplx.imag
                else:
                    for i, sv in enumerate(simult_seed):
                        eige_cmplx, eigs = sp.linalg.eigs(H, k=Ms, which='LM', sigma=sv)
                        tmp_eige[Ms*i:Ms*(i+1)] = eige_cmplx.real
                        eig_simult[Ms*i:Ms*(i+1)] = eige_cmplx.imag
                        tmp_state[:,Ms*i:Ms*(i+1)] = eigs
                    sort_idx = np.argsort(tmp_eige) #sort values for one paramter along energy, otherwise scrambled by differnet seeds
                    eige, eigs = tmp_eige[sort_idx], tmp_state[:,sort_idx]
                    simult_ev[(slice(None),) + j] = eig_simult[sort_idx]
            else:
                if full:
                    eige, eigs = np.linalg.eigh(H.todense())
                else:
                    eige, eigs = sp.linalg.eigsh(H, k=M, which='LM', sigma=0.0)
            self.eigenenergy[(slice(None),) + j] = eige
            self.eigenstate[(slice(None), slice(None)) + j] = eigs
            self.param_array[(slice(None),) + j] = p

        self.Observables = {"E": Observable(name="E", op=self.ops_list, eigenvalues=self.eigenenergy)}
        if simult_obs is not None:
            simult_obs.eigenvalues = simult_ev
            self.Observables.update({simult_obs.name: simult_obs})

    @property
    def states(self):
        return np.squeeze(self.eigenstate)

    @property
    def energies(self):
        return np.squeeze(self.eigenenergy)

    def observe_eigenvalues(self, obs):
        """
        Computes the expectation value of an observable for the whole eigensystem
        :param obs: Observable to compute
        :return: sets the eigenvalues in the obs object, shape [M, N0, N1, N2...]
        """
        obs.eigenvalues = np.empty((self.M, *self.param_shape))
        S2 = obs.op.dot(obs.op) #squared operator
        for i in range(self.M):
            for j in product(*self.param_idx):
                st = self.eigenstate[(slice(None),) + (i,) + j].copy()
                v = obs.op.dot(st) #return eigenvalue*eigenvector

                norm = np.abs(st.conj() @ st)
                s = np.abs(st.conj() @ v) / norm
                var = np.abs(st.conj() @ (S2.dot(st))) / norm - s ** 2
                if var > 1.e-7:
                    s = np.nan
                obs.eigenvalues[(i,) + j] = s

    def observe_eigenvalues_old(self, obs):
        """
        Computes the expectation value of an observable for the whole eigensystem
        :param obs: Observable to compute
        :return: sets the eigenvalues in the obs object, shape [M, N0, N1, N2...]
        """
        obs.eigenvalues = np.empty((self.M, *self.param_shape))
        for i in range(self.M):
            for j in product(*self.param_idx):
                st = self.eigenstate[(slice(None),) + (i,) + j].copy()
                v = obs.op.dot(st) #return eigenvalue*eigenvector
                #v[np.abs(v) < 1.e-3] = np.nan#np.finfo(np.float32).eps*10
                st[np.abs(st) < 1.e-3] = np.nan
                e = v/st
                e = np.nanmean(e)
                if np.nanvar(e) > 1.e-4:
                    e = np.nan
                #if not np.allclose(e, np.ones_like(e)*e):
                #    e = np.nan
                obs.eigenvalues[(i,) + j] = e

    def add_observable(self, name, op):
        obs = Observable(name=name, op=op)
        self.observe_eigenvalues(obs)
        self.Observables.update({name: obs})

    def plot_energies(self, Mshow=5, axis1=None, axis2=None, ax=None):
        d=0
        if axis1 is not None:
            assert self.eigenenergy.shape[axis1] > 1, "Can't plot along axis with size 1"
            d=1
        if axis2 is not None:
            assert self.eigenenergy.shape[axis2] > 1, "Can't plot along axis with size 1"
            d=2
        if ax is None:
            f, ax = plt.subplots(1,1)

        if d==1:
            for i in range(self.M):
                #plt.plot(self.param_array.take(np.arange(self.eigenenergy.shape[axis1])), self.eigenenergy[i, :, 0])
                pass

    def get_observable(self, name):
        return self.Observables[name].eigenvalues

    def plot_observable(self, name, ax=None, Mshow=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        else:
            fig = None

        arr = self.Observables[name].eigenvalues
        sq_arr = np.squeeze(arr)
        assert  len(sq_arr.shape) == 2, "More than 1D, can not plot this"
        axis = np.where(np.array(arr.shape) > 1)[0][1] #get the axis that is larger than one and not the M index
        sq_param = np.array(self.param_list[axis-1])
        if Mshow is None:
            Mshow = self.M

        for i in range(Mshow):
            ax.plot(sq_param , sq_arr[i, :])

        return fig, ax
