import os
from itertools import product

import numpy as np
from cobaya.log import LoggedError


# Converts from cmb units to brightness. Numerical factors not included,
# it needs proper normalization when used.
def _cmb2bb(nu):
    # NB: numerical factors not included
    from scipy import constants

    T_CMB = 2.72548
    x = nu * constants.h * 1e9 / constants.k / T_CMB
    return np.exp(x) * (nu * x / np.expm1(x)) ** 2


class TheoryForge_MFLike:
    def __init__(self, mflike=None):

        if mflike is None:
            import logging

            self.log = logging.getLogger(self.__class__.__name__.lower())
            self.data_folder = None
            self.experiments = np.array(["LAT_93", "LAT_145", "LAT_225"])
            self.foregrounds = {
                "normalisation": {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725},
                "components": {
                    "tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
                    "te": ["radio", "dust"],
                    "ee": ["radio", "dust"],
                },
            }
            self.l_bpws = np.arange(2, 6002)
            self.requested_cls = ["tt", "te", "ee"]
            self.bandint_freqs = np.array([93.0, 145.0, 225.0])
            self.use_top_hat_band = False
        else:
            self.log = mflike.log
            self.data_folder = mflike.data_folder
            self.experiments = mflike.experiments
            self.foregrounds = mflike.foregrounds
            self.bands = mflike.bands
            self.l_bpws = mflike.l_bpws
            self.requested_cls = mflike.requested_cls
            self.expected_params_fg = mflike.expected_params_fg
            self.expected_params_nuis = mflike.expected_params_nuis
            self.spec_meta = mflike.spec_meta
            self.defaults_cuts = mflike.defaults
            self.use_acts = mflike.use_acts
            self.use_acte = mflike.use_acte
            self.use_sptg = mflike.use_sptg
            self.freqs = {"acts": [148, 220],
                          "acte": [148, 220],
                          "sptg": [90, 150, 220],
                          "sptr": [90, 150, 220],
                          }

            # Initialize foreground model
            self._init_foreground_model()

    def get_modified_theory(self, Dls, **params):
        fg_params = {k: params[k] for k in self.expected_params_fg}
        nuis_params = {k: params[k] for k in self.expected_params_nuis}

        fg_dict = self._get_foreground_model(**fg_params)

        cmbfg_dict = {}
        # Sum CMB and FGs
        for exp1, exp2 in product(self.experiments, self.experiments):
            # if exp1[:4] != exp2[:4]:  #no cross experiments spectra
            #     continue
            for s in self.requested_cls:
                cmbfg_dict[s, exp1, exp2] = Dls[s] + fg_dict[s, "all", exp1, exp2]

        # Apply alm based calibration factors
        cmbfg_dict = self._get_calibrated_spectra(cmbfg_dict, **nuis_params)

        # Built theory
        dls_dict = {}
        for m in self.spec_meta:
            p = m["pol"]
            if p in ["tt", "ee", "bb"]:
                dls_dict[p, m["t1"], m["t2"]] = cmbfg_dict[p, m["t1"], m["t2"]]
            else:  # ['te','tb','eb']
                if m["hasYX_xsp"]:  # not symmetrizing
                    dls_dict[p, m["t1"], m["t2"]] = cmbfg_dict[p, m["t2"], m["t1"]]
                else:
                    dls_dict[p, m["t1"], m["t2"]] = cmbfg_dict[p, m["t1"], m["t2"]]

                if self.defaults_cuts["symmetrize"]:  # we average TE and ET (as we do for data)
                    dls_dict[p, m["t1"], m["t2"]] += cmbfg_dict[p, m["t2"], m["t1"]]
                    dls_dict[p, m["t1"], m["t2"]] *= 0.5

        return dls_dict

    ###########################################################################
    ## This part deals with foreground construction and bandpass integration ##
    ###########################################################################

    # Initializes the foreground model. It sets the SED and reads the templates
    def _init_foreground_model(self):

        from fgspectra import cross as fgc
        from fgspectra import frequency as fgf
        from fgspectra import power as fgp

        template_path = os.path.join(os.path.dirname(os.path.abspath(fgp.__file__)), "data")
        cibc_file = "/Users/benjaminberingue/Documents/Research/ACT/project/0223_mflike-highL/highL_2015/data/Fg/cib_extra.dat"
        ksz_file = "/Users/benjaminberingue/Documents/Research/ACT/project/0223_mflike-highL/highL_2015/data/Fg/cl_ksz_Trac_nt20.dat"
        tsz_file = "/Users/benjaminberingue/Documents/Research/ACT/project/0223_mflike-highL/highL_2015/data/Fg/tsz_143_eps0.50ext.dat"
        tszxcib_file = "/Users/benjaminberingue/Documents/Research/ACT/project/0223_mflike-highL/highL_2015/data/Fg/sz_x_cib_template.dat"

        tsz_file_reichardt = "/Users/benjaminberingue/Documents/Research/ACT/project/0623_reichardt_mflike/likelihood_fortran/ptsrc/dl_shaw_tsz_s10_153ghz_norm1_fake25000.txt"
        ksz_file_reichardt = "/Users/benjaminberingue/Documents/Research/ACT/project/0623_reichardt_mflike/likelihood_fortran/ptsrc/dl_ksz_CSFplusPATCHY_13sep2011_norm1_fake25000.txt"


        # set pivot freq and multipole
        self.fg_nu_0 = self.foregrounds["normalisation"]["nu_0"]
        self.fg_ell_0 = self.foregrounds["normalisation"]["ell_0"]

        self.cirrus_act = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        self.cirrus_spt = fgc.FactorizedCrossSpectrum(fgf.FreeSED(), fgp.PowerLaw())
        self.ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.PowerSpectrumFromFile(ksz_file))
        self.poisson = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.PowerLaw())
        self.tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.PowerSpectrumFromFile(tsz_file))
        self.cibc = fgc.FactorizedCrossSpectrum(fgf.FreeSED(), fgp.PowerSpectrumFromFile(cibc_file))
        self.tSZ_and_CIB = fgc.CorrelatedFactorizedCrossSpectrum(fgf.Join(fgf.ThermalSZ(),fgf.FreeSED()),
                                                   fgp.PowerSpectraAndCovariance(
                                                       fgp.PowerSpectrumFromFile(tsz_file),
                                                       fgp.PowerSpectrumFromFile(cibc_file),
                                                       fgp.PowerSpectrumFromFile(tszxcib_file)))

        self.tsz_spt_reichardt = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.PowerSpectrumFromFile(tsz_file_reichardt))
        self.ksz_spt_reichardt = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.PowerSpectrumFromFile(ksz_file_reichardt))


        components = self.foregrounds["components"]
        self.fg_component_list = {s: components[s] for s in self.requested_cls}

    # Gets the actual power spectrum of foregrounds given the passed parameters
    def _get_foreground_model(self, ell=None, freqs_order=None, **fg_params):
        # if ell = None, it uses the l_bpws, otherwise the ell array provided
        # useful to make tests at different l_max than the data
        if not hasattr(ell, "__len__"):
            ell = self.l_bpws
        ell_0 = self.fg_ell_0
        nu_0 = self.fg_nu_0

        # Normalisation of radio sources
        ell_clp = ell * (ell + 1.0)
        ell_0clp = ell_0 * (ell_0 + 1.0)

        model = {}
        if self.use_acts:
            model["acts", "tt", "kSZ"] = self.ksz({"nu": np.array([146.9, 220.2])},
                                                  {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_kSZ"]},
                                                  )
            poisson_amp = np.array([[fg_params["aps_148"], np.sqrt(fg_params["aps_148"]*fg_params["aps_218"]) * fg_params["rpsa"]],
                                   [np.sqrt(fg_params["aps_148"]*fg_params["aps_218"]) * fg_params["rpsa"], fg_params["aps_218"]]])
            model["acts", "tt", "poisson"] = self.poisson({"nu": np.array([146.9, 220.2])},
                                                          {"ell": ell_clp, "ell_0": ell_0clp,
                                                           "alpha": 1., "amp": poisson_amp},
                                                          )
            model["acts", "tt", "tSZ"] = self.tsz({"nu": np.array([146.9, 220.2]), "nu_0": 143.},
                                                  {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_tSZ"]},
                                                  )
            model["acts", "tt", "cibc"] = self.cibc({"nu": np.array([149.7,219.6]), "sed": np.array([0.12, 0.89])**.5},
                                                    {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_c"]},
                                                    )
            model["acts", "tt", "cirrus"] = self.cirrus_act({"nu": np.array([149.7, 219.6]), "nu_0": nu_0, "beta": 3.8-2.},
                                                            {"ell": ell, "ell_0": ell_0, "alpha": -0.7, "amp": fg_params["a_gtt_as"]},
                                                            )
            model["acts", "tt", "tSZ_and_CIB"] = self.tSZ_and_CIB(
                {"kwseq": ({"nu": np.array([146.9, 220.2]), "nu_0": 143.},
                           {"nu": np.array([146.9, 220.2]), "sed": np.array([0.12, 0.89])**.5})},
                {"kwseq": ({"ell": ell, "ell_0": ell_0, "amp": fg_params["a_tSZ"]},
                           {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_c"]},
                           {"ell": ell, "ell_0": ell_0, "amp": -fg_params["xi"] * np.sqrt(fg_params["a_tSZ"] * fg_params["a_c"])},
                           )
                },
            )

        if self.use_acte:
            model["acte", "tt", "kSZ"] = self.ksz({"nu": np.array([146.9, 220.2])},
                                                  {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_kSZ"]},
                                                  )
            poisson_amp = np.array([[fg_params["aps_148"], np.sqrt(fg_params["aps_148"]*fg_params["aps_218"]) * fg_params["rpsa"]],
                                   [np.sqrt(fg_params["aps_148"]*fg_params["aps_218"]) * fg_params["rpsa"], fg_params["aps_218"]]])
            model["acte", "tt", "poisson"] = self.poisson({"nu": np.array([146.9, 220.2])},
                                                          {"ell": ell_clp, "ell_0": ell_0clp,
                                                           "alpha": 1., "amp": poisson_amp},
                                                          )
            model["acte", "tt", "tSZ"] = self.tsz({"nu": np.array([146.9, 220.2]), "nu_0": 143.},
                                                  {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_tSZ"]},
                                                  )
            model["acte", "tt", "cibc"] = self.cibc({"nu": np.array([149.7,219.6]), "sed": np.array([0.12, 0.89])**.5},
                                                    {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_c"]},
                                                    )
            model["acte", "tt", "cirrus"] = self.cirrus_act({"nu": np.array([149.7, 219.6]), "nu_0": nu_0, "beta": 3.8-2.},
                                                            {"ell": ell, "ell_0": ell_0, "alpha": -0.7, "amp": fg_params["a_gtt_ae"]},
                                                            )
            model["acte", "tt", "tSZ_and_CIB"] = self.tSZ_and_CIB(
                {"kwseq": ({"nu": np.array([146.9, 220.2]), "nu_0": 143.},
                           {"nu": np.array([146.9, 220.2]), "sed": np.array([0.12, 0.89])**.5})},
                {"kwseq": ({"ell": ell, "ell_0": ell_0, "amp": fg_params["a_tSZ"]},
                           {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_c"]},
                           {"ell": ell, "ell_0": ell_0, "amp": -fg_params["xi"] * np.sqrt(fg_params["a_tSZ"] * fg_params["a_c"])},
                           )
                },
            )
        if self.use_sptg:
            model["sptg", "tt", "kSZ"] = self.ksz({"nu": np.array([97.6, 153.1, 218.1])},
                                                  {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_kSZ"]},
                                                  )
            poisson_amp = np.array([[fg_params["aps_90"],
                                     fg_params["rps0"] * np.sqrt(fg_params["aps_90"] * fg_params["aps_150"]),
                                     fg_params["rps1"] * np.sqrt(fg_params["aps_90"] * fg_params["aps_220"])],
                                    [fg_params["rps0"] * np.sqrt(fg_params["aps_90"] * fg_params["aps_150"]),
                                     fg_params["aps_150"],
                                     fg_params["rps2"] * np.sqrt(fg_params["aps_220"] * fg_params["aps_150"])],
                                    [fg_params["rps1"] * np.sqrt(fg_params["aps_90"] * fg_params["aps_220"]),
                                     fg_params["rps2"] * np.sqrt(fg_params["aps_220"] * fg_params["aps_150"]),
                                     fg_params["aps_220"]]])
            model["sptg", "tt", "poisson"] = self.poisson({"nu": np.array([97.6, 153.1, 218.1])},
                                                          {"ell": ell_clp, "ell_0": ell_0clp,
                                                           "alpha": 1., "amp": poisson_amp},
                                                          )
            model["sptg", "tt", "tSZ"] = self.tsz({"nu": np.array([97.6, 153.1, 218.1]), "nu_0": 143.},
                                                  {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_tSZ"]},
                                                  )
            model["sptg", "tt", "cibc"] = self.cibc({"nu": np.array([97.6, 153.1, 218.1]),
                                                     "sed": np.array([0.026, 0.14, 0.91])**.5},
                                                    {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_c"]},
                                                    )
            model["sptg", "tt", "cirrus"] = self.cirrus_spt({"nu": np.array([97.6, 153.1, 218.1]),
                                                             "sed": np.array([0.16,0.21,2.19])**.5},
                                                            {"ell": ell, "ell_0": ell_0, "alpha": -0.7, "amp": 1.},
                                                        )
            model["sptg", "tt", "tSZ_and_CIB"] = self.tSZ_and_CIB(
                {"kwseq": ({"nu": np.array([97.6, 153.1, 218.1]), "nu_0": 143.},
                           {"nu": np.array([97.6, 153.1, 218.1]), "sed": np.array([0.026, 0.14, 0.91])**.5})},
                {"kwseq": ({"ell": ell, "ell_0": ell_0, "amp": fg_params["a_tSZ"]},
                           {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_c"]},
                           {"ell": ell, "ell_0": ell_0, "amp": -fg_params["xi"] * np.sqrt(fg_params["a_tSZ"] * fg_params["a_c"])},
                           )
                },
            )

        if self.use_sptr:
            model["sptg", "tt", "kSZ"] = self.ksz_spt_reichardt({"nu": self.freqs["sptr"]},
                                                                {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_kSZ"]},
                                                                )
            poisson_amp = np.array([[fg_params["aps_90"],
                                     fg_params["rps0"] * np.sqrt(fg_params["aps_90"] * fg_params["aps_150"]),
                                     fg_params["rps1"] * np.sqrt(fg_params["aps_90"] * fg_params["aps_220"])],
                                    [fg_params["rps0"] * np.sqrt(fg_params["aps_90"] * fg_params["aps_150"]),
                                     fg_params["aps_150"],
                                     fg_params["rps2"] * np.sqrt(fg_params["aps_220"] * fg_params["aps_150"])],
                                    [fg_params["rps1"] * np.sqrt(fg_params["aps_90"] * fg_params["aps_220"]),
                                     fg_params["rps2"] * np.sqrt(fg_params["aps_220"] * fg_params["aps_150"]),
                                     fg_params["aps_220"]]])
            model["sptg", "tt", "poisson"] = self.poisson({"nu": np.array([97.6, 153.1, 218.1])}, #TODO
                                                          {"ell": ell_clp, "ell_0": ell_0clp,
                                                           "alpha": 1., "amp": poisson_amp},
                                                          )
            model["sptg", "tt", "tSZ"] = self.tsz_spt_reichardt({"nu": np.array([96.55, 152.26,220.10]), "nu_0": 143.},
                                                                {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_tSZ"]},
                                                                )
            model["sptg", "tt", "cibc"] = self.cibc({"nu": np.array([97.6, 153.1, 218.1]), #TODO
                                                     "sed": np.array([0.026, 0.14, 0.91])**.5},
                                                    {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_c"]},
                                                    )
            model["sptg", "tt", "cirrus"] = self.cirrus_spt({"nu": np.array([97.6, 153.1, 218.1]), #TODO
                                                             "sed": np.array([0.16,0.21,2.19])**.5},
                                                            {"ell": ell, "ell_0": ell_0, "alpha": -0.7, "amp": 1.},
                                                        )
            model["sptg", "tt", "tSZ_and_CIB"] = self.tSZ_and_CIB(   #TODO
                {"kwseq": ({"nu": np.array([97.6, 153.1, 218.1]), "nu_0": 143.},
                           {"nu": np.array([97.6, 153.1, 218.1]), "sed": np.array([0.026, 0.14, 0.91])**.5})},
                {"kwseq": ({"ell": ell, "ell_0": ell_0, "amp": fg_params["a_tSZ"]},
                           {"ell": ell, "ell_0": ell_0, "amp": fg_params["a_c"]},
                           {"ell": ell, "ell_0": ell_0, "amp": -fg_params["xi"] * np.sqrt(fg_params["a_tSZ"] * fg_params["a_c"])},
                           )
                },
            )

        fg_dict = {}
        if not hasattr(freqs_order, "__len__"):
            experiments = self.experiments
        else:
            experiments = freqs_order
        for exp1 in experiments:
            for exp2 in experiments:
                if exp1[:4] != exp2[:4]:
                    exp = "sptg"  #Dodgy ... set all cross correlations between experiemn
                    # ts to irrelevant values
                    c1=0
                    c2=0
                else:
                    exp = exp1[:4]
                    c1 = self.freqs[exp].index(int(exp1[5:]))
                    c2 = self.freqs[exp].index(int(exp2[5:]))
                for s in self.requested_cls:
                    fg_dict[s, "all", exp1, exp2] = np.zeros(len(ell))
                    for comp in self.fg_component_list[s]:
                        if comp == "tSZ_and_CIB":
                            fg_dict[s, "tSZ", exp1, exp2] = model[exp, s, "tSZ"][c1, c2]
                            fg_dict[s, "cibc", exp1, exp2] = model[exp, s, "cibc"][c1, c2]
                            fg_dict[s, "tSZxCIB", exp1, exp2] = (
                                model[exp, s, comp][c1, c2]
                                - model[exp, s, "tSZ"][c1, c2]
                                - model[exp, s, "cibc"][c1, c2]
                            )
                            fg_dict[s, "all", exp1, exp2] += model[exp, s, comp][c1, c2]
                            # fg_dict[s, "all", exp1, exp2] += fg_dict[s, "tSZxCIB", exp1, exp2]
                        else:
                            fg_dict[s, comp, exp1, exp2] = model[exp, s, comp][c1, c2]
                            fg_dict[s, "all", exp1, exp2] += fg_dict[s, comp, exp1, exp2]

        return fg_dict

    ###########################################################################
    ## This part deals with calibration factors
    ## Here we implement an alm based calibration
    ## Each field {T,E,B}{freq1,freq2,...,freqn} gets an independent
    ## calibration factor, e.g. calT_145, calE_154, calT_225, etc..
    ## plus a calibration factor per channel, e.g. cal_145, etc...
    ## A global calibration factor calG_all is also considered.
    ###########################################################################

    def _get_calibrated_spectra(self, dls_dict, **nuis_params):

        from syslibrary import syslib_mflike as syl

        cal_pars = {}
        if "tt" in self.requested_cls or "te" in self.requested_cls:
            cal = np.array([nuis_params[f"cal_{exp}"] for exp in self.experiments])
            cal_pars["tt"] = 1 / cal

        calib = syl.Calibration_alm(ell=self.l_bpws, spectra=dls_dict)

        return calib(cal1=cal_pars, cal2=cal_pars, nu=self.experiments)

    ###########################################################################
    ## This part deals with rotation of spectra
    ## Each freq {freq1,freq2,...,freqn} gets a rotation angle alpha_93, alpha_145, etc..
    ###########################################################################

    def _get_rotated_spectra(self, dls_dict, **nuis_params):

        from syslibrary import syslib_mflike as syl

        # rot_pars = [nuis_params[f"alpha_{exp}"] for exp in self.experiments]
        rot_pars = [0. for exp in self.experiments]


        rot = syl.Rotation_alm(ell=self.l_bpws, spectra=dls_dict)

        return rot(rot_pars, nu=self.experiments, cls=self.requested_cls)

    ###########################################################################
    ## This part deals with template marginalization
    ## A dictionary of template dls is read from yaml (likely to be not efficient)
    ## then rescaled and added to theory dls
    ###########################################################################

    # Initializes the systematics templates
    # This is slow, but should be done only once
    def _init_template_from_file(self):
        if not self.systematics_template.get("rootname"):
            raise LoggedError(self.log, "Missing 'rootname' for systematics template!")

        from syslibrary import syslib_mflike as syl

        # decide where to store systematics template.
        # Currently stored inside syslibrary package
        templ_from_file = syl.ReadTemplateFromFile(rootname=self.systematics_template["rootname"])
        self.dltempl_from_file = templ_from_file(ell=self.l_bpws)

    def _get_template_from_file(self, dls_dict, **nuis_params):

        # templ_pars=[nuis_params['templ_'+str(exp)] for exp in self.experiments]
        # templ_pars currently hard-coded
        # but ideally should be passed as input nuisance
        templ_pars = {
            cls: np.zeros((len(self.experiments), len(self.experiments)))
            for cls in self.requested_cls
        }

        for cls in self.requested_cls:
            for i1, exp1 in enumerate(self.experiments):
                for i2, exp2 in enumerate(self.experiments):
                    dls_dict[cls, exp1, exp2] += (
                        templ_pars[cls][i1][i2] * self.dltempl_from_file[cls, exp1, exp2]
                    )

        return dls_dict

class TheoryForge_PlikMFLike:
    def __init__(self, mflike=None):

        if mflike is None:
            import logging

            self.log = logging.getLogger(self.__class__.__name__.lower())
            self.data_folder = None
            self.experiments = np.array(["LAT_93", "LAT_145", "LAT_225"])
            self.foregrounds = {
                "normalisation": {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725},
                "components": {
                    "tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
                    "te": ["radio", "dust"],
                    "ee": ["radio", "dust"],
                },
            }
            self.l_bpws = np.arange(2, 6002)
            self.requested_cls = ["tt", "te", "ee"]
            self.bandint_freqs = np.array([93.0, 145.0, 225.0])
            self.use_top_hat_band = False
        else:
            self.log = mflike.log
            self.data_folder = mflike.data_folder
            self.foregrounds = mflike.foregrounds
            self.requested_cls = mflike.requested_cls
            self.expected_params = mflike.expected_params
            self.defaults_cuts = mflike.defaults
        from fgspectra import cross as fgc
        from fgspectra import frequency as fgf
        from fgspectra import power as fgp

        ksz_planck_file = fgp._get_power_file("ksz_planck")
        tsz_planck_file = fgp._get_power_file("tsz_planck")
        tszxcib_planck_file = fgp._get_power_file("sz_x_cib_planck")

        self.ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.PowerSpectrumFromFile(ksz_planck_file))
        self.tsz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.PowerSpectrumFromFile(tsz_planck_file))
        self.cib = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.CIB_Planck())
        self.ttps = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.PowerLaw())
        self.tszxcib = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.PowerSpectrumFromFile(tszxcib_planck_file))
        self.gal = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.gal_Planck())
        self.galte = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.PowerLaw())

    def get_Planck_foreground(self, fg_params, ell, requested_cls=['tt', 'te', 'ee']):
        components = self.foregrounds["components"]
        self.fg_component_list = {s: components[s] for s in self.requested_cls}

        # The spectra templates for the foregrounds.

        nu_0 = self.foregrounds["normalisation"]["nu_0"]
        ell_0 = self.foregrounds["normalisation"]["ell_0"]

        frequencies = np.asarray([100, 143, 217], dtype=int)

        tSZcorr = np.array([2.022, 0.95, 0.0000476])
        CIBcorr = np.array([0.0, 0.094, 1.0])

        model = {}

        # A lot of the foreground modeling is done very explicitly, due to
        # the way it is supposed to work with fgspectra and the way it
        # used to be done in plik.

        tsz_amp = np.zeros((len(frequencies), len(frequencies)))
        tsz_amp[0, 0] = fg_params['a_tSZ'] * tSZcorr[0]
        tsz_amp[1, 1] = fg_params['a_tSZ'] * tSZcorr[1]
        tsz_amp[1, 2] = fg_params['a_tSZ'] * np.sqrt(tSZcorr[2])
        tsz_amp[2, 2] = fg_params['a_tSZ'] * tSZcorr[2]

        ps_amp = np.zeros((len(frequencies), len(frequencies)))
        ps_amp[0, 0] = fg_params['aps_100_100']
        ps_amp[1, 1] = fg_params['aps_143_143']
        ps_amp[1, 2] = fg_params['aps_143_217']
        ps_amp[2, 2] = fg_params['aps_217_217']

        gal_amp = np.zeros((len(frequencies), len(frequencies)))
        gal_amp[0, 0] = fg_params['gal545_A_100']
        gal_amp[1, 1] = fg_params['gal545_A_143']
        gal_amp[1, 2] = fg_params['gal545_A_143_217']
        gal_amp[2, 1] = fg_params['gal545_A_143_217']
        gal_amp[2, 2] = fg_params['gal545_A_217']

        galte_amp = np.zeros((len(frequencies), len(frequencies)))
        galte_amp[0, 0] = fg_params['galf_TE_A_100']
        galte_amp[0, 1] = fg_params['galf_TE_A_100_143']
        galte_amp[0, 2] = fg_params['galf_TE_A_100_217']
        galte_amp[1, 0] = fg_params['galf_TE_A_100_143']
        galte_amp[2, 0] = fg_params['galf_TE_A_100_217']
        galte_amp[1, 1] = fg_params['galf_TE_A_143']
        galte_amp[1, 2] = fg_params['galf_TE_A_143_217']
        galte_amp[2, 1] = fg_params['galf_TE_A_143_217']
        galte_amp[2, 2] = fg_params['galf_TE_A_217']

        galee_amp = np.zeros((len(frequencies), len(frequencies)))
        galee_amp[0, 0] = fg_params['galf_EE_A_100']
        galee_amp[0, 1] = fg_params['galf_EE_A_100_143']
        galee_amp[0, 2] = fg_params['galf_EE_A_100_217']
        galee_amp[1, 0] = fg_params['galf_EE_A_100_143']
        galee_amp[2, 0] = fg_params['galf_EE_A_100_217']
        galee_amp[1, 1] = fg_params['galf_EE_A_143']
        galee_amp[1, 2] = fg_params['galf_EE_A_143_217']
        galee_amp[2, 1] = fg_params['galf_EE_A_143_217']
        galee_amp[2, 2] = fg_params['galf_EE_A_217']

        szcib_amp = np.zeros((len(frequencies), len(frequencies)))
        szcib_amp[0, 0] = -2.0 * fg_params['xi'] * np.sqrt(
            fg_params['a_tSZ'] * tSZcorr[0] * fg_params['a_c'] * CIBcorr[0])
        szcib_amp[1, 1] = -2.0 * fg_params['xi'] * np.sqrt(
            fg_params['a_tSZ'] * tSZcorr[1] * fg_params['a_c'] * CIBcorr[1])
        szcib_amp[1, 2] = -fg_params['xi'] * np.sqrt(
            fg_params['a_tSZ'] * tSZcorr[1] * fg_params['a_c'] * CIBcorr[2]) - fg_params['xi'] * np.sqrt(
            fg_params['a_tSZ'] * tSZcorr[2] * fg_params['a_c'] * CIBcorr[1])
        szcib_amp[2, 2] = -2.0 * fg_params['xi'] * np.sqrt(
            fg_params['a_tSZ'] * tSZcorr[2] * fg_params['a_c'] * CIBcorr[2])

        ell_clp = ell * (ell + 1.0)
        ell_0clp = ell_0 * (ell_0 + 1.0)

        model['tt', 'kSZ'] = fg_params['a_kSZ'] * self.ksz({'nu': frequencies}, {'ell': ell, 'ell_0': ell_0})
        model['tt', 'tSZ'] = tsz_amp[..., np.newaxis] * self.tsz({"nu": frequencies}, {"ell": ell, "ell_0": ell_0})
        model['tt', 'tSZxCIB'] = szcib_amp[..., np.newaxis] * self.tszxcib({"nu": frequencies}, {"ell": ell, "ell_0": ell_0})
        model['tt', 'poisson'] = ps_amp[..., np.newaxis] * self.ttps({"nu": frequencies},
                                                           {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1.})
        model['tt', 'CIB'] = fg_params['a_c'] * self.cib({"nu": frequencies},
                                                    {"ell": ell, "ell_0": ell_0, 'n_cib': fg_params['cib_index']})
        model['tt', 'galactic'] = gal_amp[..., np.newaxis] * self.gal({"nu": frequencies}, {"ell": ell})

        model['te', 'galactic'] = galte_amp[..., np.newaxis] * self.galte({"nu": frequencies}, {"ell": ell, "ell_0": 500.0,
                                                                                      "alpha": fg_params[
                                                                                                   "galf_TE_index"] + 2.0})
        model['ee', 'galactic'] = galee_amp[..., np.newaxis] * self.galte({"nu": frequencies}, {"ell": ell, "ell_0": 500.0,
                                                                                      "alpha": fg_params[
                                                                                                   "galf_EE_index"] + 2.0})

        fg_dict = {}

        for idx, (i, j) in enumerate([(0, 0), (1, 1), (1, 2), (2, 2)]):
            f1, f2 = frequencies[i], frequencies[j]

            fg_dict['tt', 'kSZ', f1, f2] = model['tt', 'kSZ'][i, j]
            fg_dict['tt', 'tSZ', f1, f2] = model['tt', 'tSZ'][i, j]
            fg_dict['tt', 'tSZxCIB', f1, f2] = model['tt', 'tSZxCIB'][i, j]
            fg_dict['tt', 'poisson', f1, f2] = model['tt', 'poisson'][i, j]
            fg_dict['tt', 'CIB', f1, f2] = model['tt', 'CIB'][i, j]  # Picking the right template.
            fg_dict['tt', 'galactic', f1, f2] = model['tt', 'galactic'][i, j]

        for i, f1 in enumerate(frequencies):
            for j, f2 in enumerate(frequencies):
                fg_dict['te', 'galactic', f1, f2] = model['te', 'galactic'][i, j]
                fg_dict['ee', 'galactic', f1, f2] = model['ee', 'galactic'][i, j]

        # component_list = {'tt': ['kSZ', 'tSZ', 'tSZxCIB', 'CIB', 'gal', 'ps'], 'te': ['gal'], 'ee': ['gal']}
        for c1, f1 in enumerate(frequencies):
            for c2, f2 in enumerate(frequencies):
                for s in requested_cls:
                    fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
                    for comp in self.fg_component_list[s]:
                        if (s, comp, f1, f2) in fg_dict:
                            fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]

        return fg_dict
