"""
.. module:: mflike

:Synopsis: Definition of simplistic likelihood for Simons Observatory
:Authors: Thibaut Louis, Xavier Garrido, Max Abitbol,
          Erminia Calabrese, Antony Lewis, David Alonso.

"""
import os
from typing import Optional

import numpy as np
from cobaya.conventions import packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists

from .theoryforge_MFLike import TheoryForge_MFLike, TheoryForge_PlikMFLike


class MFLike(InstallableLikelihood):
    # _url = "https://portal.nersc.gov/cfs/sobs/users/MFLike_data"
    # _release = "v0.7.1"
    # install_options = {"download_url": f"{_url}/{_release}.tar.gz"}

    # attributes set from .yaml
    input_file: Optional[str]
    cov_Bbl_file: Optional[str]
    data: dict
    defaults: dict
    foregrounds: dict
    top_hat_band: dict
    systematics_template: dict

    def initialize(self):

        # Set default values to data member not initialized via yaml file
        self.l_bpws = None
        self.freqs = None
        self.spec_meta = []

        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, packages_path_input, None)):
            raise LoggedError(
                self.log,
                "No path given to MFLike data. Set the likelihood property "
                f"'path' or the common property '{packages_path_input}'.",
            )
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. "
                f"Check the given path [{self.data_folder}].",
            )

        # Read data
        self.prepare_data()

        # State requisites to the theory code
        self.requested_cls = ["tt"]
        self.lmax_theory = self.lmax_theory or 9000
        self.log.debug(f"Maximum multipole value: {self.lmax_theory}")
        use_acts = False
        use_acte = False
        use_sptg = False
        use_sptr = False

        for exp in self.experiments:
            if exp in ["acts_148", "acts_218"]:
                use_acts = True
            if exp in ["acte_148", "acte_218"]:
                use_acte = True
            if exp in ["sptg_90", "sptg_150", "sptg_220"]:
                use_sptg = True
            if exp in ["sptr_90", "sptr_150", "sptr_220"]:
                use_sptr = True

        self.use_acts = use_acts
        self.use_acte = use_acte
        self.use_sptg = use_sptg
        self.use_sptr = use_sptr

        if self.use_sptr and self.use_sptg:
            raise LoggedError(self.log("Using both Reichardt and George likelihood, please check your yaml file !"))

        self.expected_params_fg = ["a_tSZ", "a_kSZ", "a_c", "xi"] #common parameters only

        if use_acts or use_acte:
            self.expected_params_fg += ["aps_148", "aps_218", "rpsa"]
        if use_acts:
            self.expected_params_fg.append("a_gtt_as")
        if use_acte:
            self.expected_params_fg.append("a_gtt_ae")
        if use_sptg:
            self.expected_params_fg += ["aps_90", "aps_150", "aps_220", "rps0", "rps1", "rps2"]
        if use_sptr:
            self.expected_params_fg +=

        self.expected_params_nuis = [f"cal_{exp}" for exp in self.experiments]

        self.ThFo = TheoryForge_MFLike(self)
        self.log.info("Initialized!")

    def initialize_with_params(self):
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params,
            self.expected_params_fg + self.expected_params_nuis,
            name_A="given",
            name_B="expected",
        )
        # if differences:
        #     raise LoggedError(self.log, f"Configuration error in parameters: {differences}.")

    def get_requirements(self):
        # return dict(Cl={k: max(c, self.lmax_theory + 1) for k, c in self.lcuts.items()})
        return dict(Cl={k: self.lmax_theory + 1 for k, _ in self.lcuts.items()})

    def logp(self, **params_values):
        cl = self.theory.get_Cl(ell_factor=True)
        params_values_nocosmo = {
            k: params_values[k] for k in self.expected_params_fg + self.expected_params_nuis
        }
        return self.loglike(cl, **params_values_nocosmo)

    def loglike(self, cl, **params_values_nocosmo):
        ps_vec = self._get_power_spectra(cl, **params_values_nocosmo)
        delta = self.data_vec - ps_vec
        logp = -0.5 * (delta @ self.inv_cov @ delta)
        logp += self.logp_const
        self.log.debug(
            f"Log-likelihood value computed = {logp} (Χ² = {-2 * (logp - self.logp_const)})"
        )
        return logp

    def prepare_data(self):
        import sacc

        data = self.data
        # Read data
        input_fname = os.path.join(self.data_folder, self.input_file)
        s = sacc.Sacc.load_fits(input_fname)
        s_b = s
        try:
            default_cuts = self.defaults
        except AttributeError:
            raise KeyError("You must provide a list of default cuts")

        # Translation betwen TEB and sacc C_ell types
        pol_dict = {"T": "0", "E": "e", "B": "b"}
        ppol_dict = {
            "TT": "tt",
            "EE": "ee",
            "TE": "te",
            "ET": "te",
            "BB": "bb",
            "EB": "eb",
            "BE": "eb",
            "TB": "tb",
            "BT": "tb",
            "BB": "bb",
        }

        def get_cl_meta(spec):
            # For each of the entries of the `spectra` section of the
            # yaml file, extract the relevant information: experiments,
            # polarization combinations, scale cuts and
            # whether TE should be symmetrized.
            exp_1, exp_2 = spec["experiments"]
            # Read off polarization channel combinations
            pols = spec.get("polarizations", default_cuts["polarizations"]).copy()
            # Read off scale cuts
            scls = spec.get("scales", default_cuts["scales"]).copy()

            return exp_1, exp_2, pols, scls

        def get_sacc_names(exp_1, exp_2):
            # Translate the polarization combination and experiment
            # names of a given entry in the `spectra`
            # part of the input yaml file into the names expected
            # in the SACC files. (BB 03/23): For ACTS, the data are
            # given season by season, and need special treatment.
            exp = exp_1[:4]
            if exp == "acts":
                seasons_auto = ["season2sxseason2s", "season2sxseason3s", "season2sxseason4s",
                                "season3sxseason3s", "season3sxseason4s", "season4sxseason4s"]
                seasons_cross = ["season2sxseason2s", "season2sxseason3s", "season2sxseason4s",
                                 "season3sxseason2s", "season3sxseason3s", "season3sxseason4s",
                                 "season4sxseason2s", "season4sxseason3s", "season4sxseason4s"]
                if exp_1 == exp_2:
                    seasons = seasons_auto
                else:
                    seasons = seasons_cross
                tname_1 = [exp_1[5:] + '_' + sxs.split("x")[0] for sxs in seasons]
                tname_2 = [exp_2[5:] + '_' + sxs.split("x")[1] for sxs in seasons]
            elif exp == "acte":
                seasons_auto = ["season3exseason3e", "season3exseason4e", "season4exseason4e"]
                seasons_cross = ["season3exseason3e", "season3exseason4e", "season4exseason3e",
                                 "season4exseason4e"]
                if exp_1 == exp_2:
                    seasons = seasons_auto
                else:
                    seasons = seasons_cross
                tname_1 = [exp_1[5:] + '_' + sxs.split("x")[0] for sxs in seasons]
                tname_2 = [exp_2[5:] + '_' + sxs.split("x")[1] for sxs in seasons]
            elif exp == "sptg":
                tname_1 = [exp_1[5:] + '_spt']
                tname_2 = [exp_2[5:] + '_spt']
            elif exp == "sptr":
                tname_1 = [exp_1[5:] + '_spt']
                tname_2 = [exp_2[5:] + '_spt']

            dtype = "cl_00"
            return tname_1, tname_2, dtype

        # First we trim the SACC file so it only contains
        # the parts of the data we care about.
        # Indices to be kept
        indices = []
        # Length of the final data vector
        len_compressed = 0
        for spectrum in data["spectra"]:
            exp_1, exp_2, pols, scls = get_cl_meta(spectrum)
            tname_1, tname_2, dtype = get_sacc_names(exp_1, exp_2)
            for t1, t2 in zip(tname_1, tname_2):
                lmin, lmax = scls["TT"]
                ind = s.indices(
                    dtype,  # Power spectrum type
                    (t1, t2),  # Channel combinations
                    ell__gt=lmin,
                    ell__lt=lmax,
                )  # Scale cuts
                indices += list(ind)
                len_compressed += ind.size

                self.log.debug(f"{tname_1} {tname_2} {dtype} {ind.shape} {lmin} {lmax}")

        # Get rid of all the unselected power spectra.
        # Sacc takes care of performing the same cuts in the
        # covariance matrix, window functions etc.
        s.keep_indices(np.array(indices))

        # Now create metadata for each spectrum
        len_full = s.mean.size
        # These are the matrices we'll use to compress the data if
        # `symmetrize` is true.
        # Note that a lot of the complication in this function is caused by the
        # symmetrization option, for which SACC doesn't have native support.
        mat_compress = np.zeros([len_compressed, len_full])

        self.lcuts = {k: c[1] for k, c in default_cuts["scales"].items()}
        index_sofar = 0

        for spectrum in data["spectra"]:
            exp_1, exp_2, pols, scls = get_cl_meta(spectrum)
            for k in scls.keys():
                self.lcuts[k] = max(self.lcuts[k], scls[k][1])
            tname_1, tname_2, dtype = get_sacc_names(exp_1, exp_2)
            for t1, t2 in zip(tname_1, tname_2):
                # The only reason why we need indices is the symmetrization.
                # Otherwise all of this could have been done in the previous
                # loop over data["spectra"].
                ls, cls, ind = s.get_ell_cl(dtype, t1, t2, return_ind=True)
                ws = s.get_bandpower_windows(ind)

                if self.l_bpws is None:
                    # The assumption here is that bandpower windows
                    # will all be sampled at the same ells.
                    self.l_bpws = ws.values

                for i, j1 in enumerate(ind):
                    mat_compress[index_sofar + i, j1] = 1
                # The fields marked with # below aren't really used, but
                # we store them just in case.
                self.spec_meta.append(
                    {
                        "ids": (index_sofar + np.arange(cls.size, dtype=int)),
                        "pol": ppol_dict["TT"],
                        "hasYX_xsp": "TT"
                        in ["ET", "BE", "BT"],  # This is necessary for handling symmetrization
                        "t1": exp_1,
                        "t2": exp_2,
                        "leff": ls,  #
                        "cl_data": cls,  #
                        "bpw": ws,
                    }
                )
                index_sofar += cls.size
        # Put data and covariance in the right order.
        self.data_vec = np.dot(mat_compress, s.mean)
        self.cov = np.dot(mat_compress, s_b.covariance.covmat.dot(mat_compress.T))
        self.inv_cov = np.linalg.inv(self.cov)
        self.logp_const = np.log(2 * np.pi) * (-len(self.data_vec) / 2)
        self.logp_const -= 0.5 * np.linalg.slogdet(self.cov)[1]

        self.experiments = data["experiments"]
        self.bands = {
            name: {"nu": tracer.nu, "bandpass": tracer.bandpass}
            for name, tracer in s.tracers.items()
        }

        # Put lcuts in a format that is recognisable by CAMB.
        self.lcuts = {k.lower(): c for k, c in self.lcuts.items()}
        if "et" in self.lcuts:
            del self.lcuts["et"]

        self.log.info(f"Number of bins used: {self.data_vec.size}")

    def _get_power_spectra(self, cl, **params_values_nocosmo):
        # Get Cl's from the theory code
        Dls = {}
        for s, _ in self.lcuts.items():
            dl = np.zeros_like(self.l_bpws)
            dl[:self.lmax_theory+1] = cl[s][:self.lmax_theory+1]
            Dls[s] = dl
        # Dls = {s: cl[s][self.l_bpws] for s, _ in self.lcuts.items()}
        DlsObs = self.ThFo.get_modified_theory(Dls, **params_values_nocosmo)
        ps_vec = np.zeros_like(self.data_vec)
        for m in self.spec_meta:
            p = m["pol"]
            i = m["ids"]
            w = m["bpw"].weight.T
            clt = w @ DlsObs[p, m["t1"], m["t2"]]
            ps_vec[i] = clt

        return ps_vec


class PlikMFLike(InstallableLikelihood):

    data_folder: Optional[str]
    cov_Bbl_file: Optional[str]
    weightfile:Optional[str]
    minfile: Optional[str]
    maxfile: Optional[str]
    covfile: Optional[str]
    specfile: Optional[str]
    leakfile: Optional[str]
    corrfile: Optional[str]
    subpixfile: Optional[str]
    foregrounds: dict
    defaults: dict

    lmin: Optional[int]
    lmax_theory: Optional[int]

    def initialize(self):
        self.expected_params = [
            # TT parameters
            'cib_index', 'a_c', 'xi', 'a_tSZ', 'a_kSZ',
            'gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217',

            'aps_100_100', 'aps_143_143', 'aps_143_217', 'aps_217_217',

            # TE parameters
            'galf_TE_index',
            'galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217', 'galf_TE_A_143', 'galf_TE_A_143_217',
            'galf_TE_A_217',

            # EE parameters
            'galf_EE_index',
            'galf_EE_A_100', 'galf_EE_A_100_143', 'galf_EE_A_100_217', 'galf_EE_A_143', 'galf_EE_A_143_217',
            'galf_EE_A_217',

            # calibration parameters
            'calib_100T', 'calib_217T', 'calib_100P', 'calib_143P', 'calib_217P', 'A_planck',

            # These parameters aren't used, but they are kept for backwards compatibility.
            # 'A_sbpx_100_100_TT', 'A_sbpx_143_143_TT', 'A_sbpx_143_217_TT', 'A_sbpx_217_217_TT',
            # 'A_cnoise_e2e_100_100_EE', 'A_cnoise_e2e_100_143_EE', 'A_cnoise_e2e_100_217_EE', 'A_cnoise_e2e_143_143_EE',
            # 'A_cnoise_e2e_143_217_EE', 'A_cnoise_e2e_217_217_EE'
        ]
        self.enable_tt = False
        self.enable_te = False
        self.enable_ee = False

        if "TT" in self.defaults["polarizations"]:
            self.enable_tt = True
        if "TE" in self.defaults["polarizations"]:
            self.enable_te = True
        if "EE" in self.defaults["polarizations"]:
            self.enable_ee = True

        self.prepare_data()
        self.requested_cls = ["tt", "te", "ee"]
        self.ThFo = TheoryForge_PlikMFLike(self)
        print(f"Fg init: {t_init_end - t_fg_start} s")
        print(f"Total initilisation: {t_init_end - t_init_start} s")

    def prepare_data(self):
        try:
            default_cuts = self.defaults
        except AttributeError:
            raise KeyError("You must provide a list of default cuts")

        self.lcuts = {k: c[1] for k, c in default_cuts["scales"].items()}
        self.lmax_win = max([self.lcuts[k] for k, _ in default_cuts["scales"].items()]) #TODO fix for different lmax
        self.nmin = [[1, 1, 1, 1], [1, 1, 60, 1, 60, 60], [1, 1, 60, 1, 60, 60]]
        self.nmax = [[136, 199, 215, 215], [114, 114, 114, 199, 199, 199], [114, 114, 114, 199, 199, 199]]

        self.nbintt = [b - a + 1 for a, b in zip(self.nmin[0], self.nmax[0])]
        self.nbinte = [b - a + 1 for a, b in zip(self.nmin[1], self.nmax[1])]
        self.nbinee = [b - a + 1 for a, b in zip(self.nmin[2], self.nmax[2])]

        self.crosstt = [(0, 0), (1, 1), (1, 2), (2, 2)]
        self.crosste = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.crossee = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

        self.freqs = [100, 143, 217]

        self.sys_vec = None
        self.inv_cov = None

        self.b_ell = None
        self.b_dat = None
        self.win_func = None

        self.log.debug('Loading windows.')
        self.load_windows_pliklike(self.weightfile, self.minfile, self.maxfile, self.nmin, self.nmax,
                                   data_dir=self.data_folder)

        self.log.debug('Loading inv cov.')
        self.inv_cov = np.loadtxt(self.data_folder + self.covfile, dtype=float)[:self.nbin, :self.nbin]
        self.covmat = np.linalg.inv(self.inv_cov)

        self.log.debug('Loading spectrum data.')
        self.b_dat = np.loadtxt(self.data_folder + self.specfile, dtype=float)[:self.nbin, 1]

        self.log.debug('Loading systematics.')
        self.load_systematics(self.leakfile, self.corrfile, self.subpixfile, data_dir=self.data_folder)

        self.log.debug('Done preparing all data!')

    def load_windows_pliklike(self, weightfile, minfile, maxfile, bin_starts, bin_ends, data_dir=''):
        # Because of the way the plik files store the window function, I wrote this function to load in the window function into a matrix form.
        # It's not the nicest code I have ever written, but it does what it needs to do.
        # For optimal use, call this function once, output the resulting win_func to a text file, and then load that in using load_plaintext every time.
        blmin = np.loadtxt(data_dir + minfile).astype(int) + self.lmin
        blmax = np.loadtxt(data_dir + maxfile).astype(int) + self.lmin
        bweight = np.concatenate([np.zeros((self.lmin - 1)), np.loadtxt(data_dir + weightfile)])

        blens = [[b - a + 1 for a, b in zip(x, y)] for x, y in zip(bin_starts, bin_ends)]
        bweight = np.repeat(bweight[np.newaxis, :], max(blens[0]), axis=0)

        # Basically, bweight temporarily stores the full window function, and we will take slices from it and put that in the full window function.
        for i in np.arange(bweight.shape[0]):
            bweight[i, :blmin[i] - 1] = 0.0
            bweight[i, blmax[i]:] = 0.0

        xmin = []
        xmax = []
        for a, b in zip(bin_starts, bin_ends):
            xmin += a
            xmax += b

        xmin = np.array(xmin) - 1
        xmax = np.array(xmax)
        xlen = xmax - xmin

        self.win_func = np.zeros((sum([sum(x) for x in blens]), self.shape))

        for i in np.arange(len(xmin)):
            xstart = np.sum(xlen[0:i])
            xend = xstart + xlen[i]

            self.win_func[xstart:xend, :] = bweight[xmin[i]:xmax[i], 1:self.shape + 1]

        del bweight

        self.nmin = bin_starts
        self.nmax = bin_ends
        self.b_ell = self.win_func @ np.arange(2, self.lmax_win + 1)

    def load_systematics(self, leak_filename, corr_filename, subpix_filename, data_dir=''):
        leakage = np.loadtxt(data_dir + leak_filename)[:, 1:]
        corr = np.loadtxt(data_dir + corr_filename)[:, 1:]
        subpix = np.loadtxt(data_dir + subpix_filename)[:, 1:]

        sum_vec = (leakage + corr + subpix)
        sys_vec = np.zeros((self.shape, sum_vec.shape[1]))

        sys_vec[:sum_vec.shape[0], :] = sum_vec[:, :]

        sys_vec = self.win_func @ sys_vec

        self.sys_vec = np.zeros((self.win_func.shape[0]))

        k = 0
        for j, tt in enumerate(self.nbintt):
            self.sys_vec[k:k + tt] = sys_vec[k:k + tt, j]
            k += tt

        # The sys vector is sorted TT-TE-EE, but it should be sorted TT-EE-TE, so we swap ordering here a bit.
        k = 0
        k0 = sum(self.nbintt)
        k1 = sum(self.nbintt) + sum(self.nbinte)
        j1 = len(self.nbintt) + len(self.nbinte)
        for j, ee in enumerate(self.nbinee):
            self.sys_vec[k + k0:k + k0 + ee] = sys_vec[k + k1:k + k1 + ee, j + j1]
            k += ee

        k = 0
        k0 = sum(self.nbintt) + sum(self.nbinee)
        k1 = sum(self.nbintt)
        j1 = len(self.nbintt)
        for j, te in enumerate(self.nbinte):
            self.sys_vec[k + k0:k + k0 + te] = sys_vec[k + k1:k + k1 + te, j + j1]
            k += te

    # def get_requirements(self):
    #     return {
    #         'Cl': {
    #             'tt': self.lmax_theory,
    #             'te': self.lmax_theory,
    #             'ee': self.lmax_theory
    #         }
    #     }
    def get_requirements(self):
        # return dict(Cl={k: max(c, self.lmax_theory + 1) for k, c in self.lcuts.items()})
        return dict(Cl={k: self.lmax_theory + 1 for k, _ in self.lcuts.items()})

    def get_model(self, cl, **params_values):
        self.log.debug('Start calculating model.')
        l0 = int(2 - cl['ell'][0])
        ls = cl['ell'][l0:self.shape + l0]
        cl_tt = cl['tt'][l0:self.shape + l0]
        cl_te = cl['te'][l0:self.shape + l0]
        cl_ee = cl['ee'][l0:self.shape + l0]

        fg = self.ThFo.get_Planck_foreground(params_values, ls)

        fg_tt = np.zeros((self.nspectt, self.shape))
        fg_te = np.zeros((self.nspecte, self.shape))
        fg_ee = np.zeros((self.nspecee, self.shape))

        for i, (c1, c2) in enumerate(self.crosstt):
            f1, f2 = int(self.frequencies[c1]), int(self.frequencies[c2])
            fg_tt[i, :] = fg['tt', 'all', f1, f2][:self.shape]

        for i, (c1, c2) in enumerate(self.crosste):
            f1, f2 = int(self.frequencies[c1]), int(self.frequencies[c2])
            fg_te[i, :] = fg['te', 'all', f1, f2][:self.shape]

        for i, (c1, c2) in enumerate(self.crossee):
            f1, f2 = int(self.frequencies[c1]), int(self.frequencies[c2])
            fg_ee[i, :] = fg['ee', 'all', f1, f2][:self.shape]

        self.log.debug('Summing theory = CMB + foreground.')

        x_theory = np.zeros((self.nspec, self.shape))

        x_theory[0: self.nspectt, :self.shape] = np.tile(cl_tt, (self.nspectt, 1)) + fg_tt
        x_theory[self.nspectt: self.nspectt + self.nspecee, :self.shape] = np.tile(cl_ee, (self.nspecee, 1)) + fg_ee
        x_theory[self.nspectt + self.nspecee: self.nspectt + self.nspecee + self.nspecte, :self.shape] = np.tile(cl_te,
                                                                                                                 (
                                                                                                                 self.nspecte,
                                                                                                                 1)) + fg_te

        self.log.debug('Completed theory vector. Now binning.')

        x_model = np.zeros((self.nbin))

        # TT modes
        for j in range(self.nspectt):
            x_model[sum(self.nbintt[0:j]): sum(self.nbintt[0:j + 1])] = self.win_func[sum(self.nbintt[0:j]): sum(
                self.nbintt[0:j + 1]), :] @ x_theory[j, :]  # TT

        # EE modes
        for j in range(self.nspecee):
            i0 = sum(self.nbintt)
            j0 = self.nspectt
            x_model[i0 + sum(self.nbinee[0:j]): i0 + sum(self.nbinee[0:j + 1])] = self.win_func[
                                                                                  i0 + sum(self.nbinee[0:j]): i0 + sum(
                                                                                      self.nbinee[0:j + 1]),
                                                                                  :] @ x_theory[j0 + j, :]  # EE

        # TE modes
        for j in range(self.nspecte):
            i0 = sum(self.nbintt) + sum(self.nbinee)
            j0 = self.nspectt + self.nspecee
            x_model[i0 + sum(self.nbinte[0:j]): i0 + sum(self.nbinte[0:j + 1])] = self.win_func[
                                                                                  i0 + sum(self.nbinte[0:j]): i0 + sum(
                                                                                      self.nbinte[0:j + 1]),
                                                                                  :] @ x_theory[j0 + j, :]  # TE

        # x = x / [ l (l + 1) / 2 pi ]
        ll = np.arange(self.shape) + 2
        ell_factor = (ll.astype(float) * (ll + 1.0)) / (2.0 * np.pi)
        x_model = x_model / (self.win_func @ ell_factor)

        self.log.debug('Adding systematics.')

        x_model += self.sys_vec

        self.log.debug('Calibrating.')

        ct = 1.0 / np.sqrt(np.array([params_values['calib_100T'], 1.0, params_values['calib_217T']]))
        yp = 1.0 / np.sqrt(
            np.array([params_values['calib_100P'], params_values['calib_143P'], params_values['calib_217P']]))

        # Calibration
        for i in np.arange(len(self.nbintt)):
            # Mode T[i]xT[j] should be calibrated using CT[i] * CT[j]
            m1, m2 = self.crosstt[i]
            x_model[sum(self.nbintt[0:i]): sum(self.nbintt[0:i + 1])] = x_model[sum(self.nbintt[0:i]): sum(
                self.nbintt[0:i + 1])] * ct[m1] * ct[m2]

        for i in np.arange(len(self.nbinee)):
            # Mode E[i]xE[j] should be calibrated using YP[i] * YP[j]
            m1, m2 = self.crossee[i]
            i0 = sum(self.nbintt)
            x_model[i0 + sum(self.nbinee[0:i]): i0 + sum(self.nbinee[0:i + 1])] = x_model[
                                                                                  i0 + sum(self.nbinee[0:i]): i0 + sum(
                                                                                      self.nbinee[0:i + 1])] * yp[m1] * \
                                                                                  yp[m2]

        for i in np.arange(len(self.nbinte)):
            # Mode T[i]xE[j] should be calibrated using CT[i] * YP[j]
            # Because of symmetric, it is calibrated using 1/2 * ( CT[i] * YP[j] + CT[j] * YP[i] )
            m1, m2 = self.crosste[i]
            i0 = sum(self.nbintt) + sum(self.nbinee)
            x_model[i0 + sum(self.nbinte[0:i]): i0 + sum(self.nbinte[0:i + 1])] = x_model[
                                                                                  i0 + sum(self.nbinte[0:i]): i0 + sum(
                                                                                      self.nbinte[0:i + 1])] * (
                                                                                              0.5 * ct[m1] * yp[
                                                                                          m2] + 0.5 * yp[m1] * ct[m2])

        # Calibrating for the overall Planck calibration parameter.
        x_model /= (params_values['A_planck'] ** 2.0)

        self.log.debug('Done calculating model.')

        return x_model

    def logp(self, **params_values):
        cl = self.theory.get_Cl(ell_factor=True)
        return self.loglike(cl, **params_values)

    def loglike(self, cl, **params_values):
        x_model = self.get_model(cl, **params_values)
        diff_vec = self.b_dat - x_model

        if self.use_tt and self.use_te and self.use_ee:
            tmp = self.inv_cov @ diff_vec
            return -0.5 * np.dot(tmp, diff_vec)

        cov = self.covmat
        mask = np.ones(cov.shape[0], dtype=bool)
        if not self.use_tt: mask[0:sum(self.nbintt)] = False
        if not self.use_ee: mask[sum(self.nbintt):sum(self.nbintt) + sum(self.nbinee)] = False
        if not self.use_te: mask[sum(self.nbintt) + sum(self.nbinee):sum(self.nbintt) + sum(self.nbinte) + sum(
            self.nbinee)] = False

        cov = cov[mask, :][:, mask]
        diff_vec = diff_vec[mask]
        tmp_cov = np.linalg.inv(cov)

        tmp = tmp_cov @ diff_vec
        return -0.5 * np.dot(tmp, diff_vec)

    @property
    def use_tt(self):
        return self.enable_tt

    @use_tt.setter
    def use_tt(self, val):
        self.enable_tt = val

    @property
    def use_te(self):
        return self.enable_te

    @use_te.setter
    def use_te(self, val):
        self.enable_te = val

    @property
    def use_ee(self):
        return self.enable_ee

    @use_ee.setter
    def use_ee(self, val):
        self.enable_ee = val

    @property
    def frequencies(self):
        return self.freqs

    @property
    def nspectt(self):
        return len(self.nbintt)

    @property
    def nspecte(self):
        return len(self.nbinte)

    @property
    def nspecee(self):
        return len(self.nbinee)

    @property
    def nbin(self):
        # total number of bins
        return sum(self.nbintt) + sum(self.nbinte) + sum(self.nbinee)

    @property
    def nspec(self):
        # total number of spectra
        return self.nspectt + self.nspecte + self.nspecee

    @property
    def shape(self):
        return self.lmax_win - 1

    @property
    def input_shape(self):
        return self.tt_lmax - 1

