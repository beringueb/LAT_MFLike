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

from .theoryforge_MFLike import TheoryForge_MFLike


class MFLikehighL(InstallableLikelihood):
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

        for exp in self.experiments:
            if exp in ["acts_148", "acts_218"]:
                use_acts = True
            if exp in ["acte_148", "acte_218"]:
                use_acte = True
            if exp in ["sptg_90", "sptg_150", "sptg_220"]:
                use_sptg = True

        self.use_acts = use_acts
        self.use_acte = use_acte
        self.use_sptg = use_sptg

        self.expected_params_fg = ["a_tSZ", "a_kSZ", "a_c", "xi"] #common parameters only

        if use_acts or use_acte:
            self.expected_params_fg += ["aps_148", "aps_218", "rpsa"]
        if use_acts:
            self.expected_params_fg.append("a_gtt_as")
        if use_acte:
            self.expected_params_fg.append("a_gtt_ae")
        if use_sptg:
            self.expected_params_fg += ["aps_90", "aps_150", "aps_220", "rps0", "rps1", "rps2"]

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
        if differences:
            raise LoggedError(self.log, f"Configuration error in parameters: {differences}.")

    def get_requirements(self):
        return dict(Cl={k: max(c, self.lmax_theory + 1) for k, c in self.lcuts.items()})

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
        Dls = {s: cl[s][self.l_bpws] for s, _ in self.lcuts.items()}
        DlsObs = self.ThFo.get_modified_theory(Dls, **params_values_nocosmo)
        ps_vec = np.zeros_like(self.data_vec)
        for m in self.spec_meta:
            p = m["pol"]
            i = m["ids"]
            w = m["bpw"].weight.T
            clt = w @ DlsObs[p, m["t1"], m["t2"]]
            ps_vec[i] = clt

        return ps_vec
