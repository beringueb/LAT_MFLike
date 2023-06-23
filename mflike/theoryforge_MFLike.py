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
            raise LoggedError(self.log("Using theory forge without mflike, not supported yet ... "))
        else:
            self.log = mflike.log
            self.data_folder = mflike.data_folder
            self.experiments = mflike.experiments
            self.foregrounds = mflike.foregrounds
            self.bands = mflike.bands
            self.l_bpws = mflike.l_bpws
            self.requested_cls = mflike.requested_cls
            # self.expected_params_fg = mflike.expected_params_fg #TODO: change to read from file ?
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
            self.exp = mflike.exp

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
        import copy
        import pickle
        self.copy = copy
        self.fgp = fgp
        self.fgc = fgc
        self.fgf = fgf

        def get_all_params(param_access):
            params = []
            for val in param_access.values():
                if isinstance(val, str):
                    params.append(val)
                elif isinstance(val, list):
                    for v in val:
                        params.extend(get_all_params(v))
                elif isinstance(val, dict):
                    params.extend(get_all_params(val))
            try:
                params.remove('ell')
            except ValueError:
                pass
            try:
                params.remove('ell_clp')
            except ValueError:
                pass
            return params

        components = {}
        components_list = {}
        expected_params_fg = []
        with open(self.foregrounds["fg_model"], 'rb') as file:
            fg_model = pickle.load(file)
        for exp in self.exp:
            for s in self.requested_cls:
                components_list[s] = [] #TODO component_list to depend on exp ? expected params as well ...
                for _, (key, value) in enumerate(fg_model[exp, s].items()):
                    components[exp, s, key, "model"] = value['model']
                    components[exp, s, key, "model"].set_defaults(**value['defaults'])
                    components[exp, s, key, "param_access"] = value['param_access']
                    components_list[s].append(key)
                    expected_params_fg.extend(get_all_params(value['param_access']))

        self.fg_component_list = components_list
        self.fgs = components
        self.expected_params_fg = list(set(expected_params_fg))
        if "tsz_and_cib" in components_list["tt"]:
            self.expected_params_fg.append("xi")
            self.expected_params_fg.remove("a_tszxcib")
        print_fgs = 'Will including the following fg components: \n'
        for s in self.requested_cls:
            print_fgs += f"{s} : "
            for c in self.fg_component_list[s]: print_fgs += f"{c}, "
            print_fgs += "\n"
        self.log.info(print_fgs)

    def _evaluate_fgs(self, model, param_access, fg_params):

        def search_for_params(kwargs, base=[]):
            for _, (key, val) in enumerate(kwargs.items()):
                if isinstance(val, list):
                    for v in val:
                        search_for_params(v, base)
                elif isinstance(val, dict):
                    search_for_params(val, base + [key])
                elif val in fg_params:
                    kwargs[key] = fg_params[val]

        modified_params = self.copy.deepcopy(param_access)
        search_for_params(modified_params)
        return model.eval(**modified_params)

    # Gets the actual power spectrum of foregrounds given the passed parameters
    def _get_foreground_model(self, ell=None, freqs_order=None, **fg_params):
        # if ell = None, it uses the l_bpws, otherwise the ell array provided
        # useful to make tests at different l_max than the data
        if not hasattr(ell, "__len__"):
            ell = self.l_bpws
        fg_params['ell'] = ell
        fg_params['ell_clp'] = ell*(ell+1.)
        fg_params["a_tszxcib"] = -fg_params["xi"] * np.sqrt(fg_params["a_tSZ"] * fg_params["a_CIB"])
        model = {}
        for exp in self.exp:
            for s in self.requested_cls:
                for c in self.fg_component_list[s]:
                    model[exp, s, c] = self._evaluate_fgs(self.fgs[exp, s, c, "model"],
                                                          self.fgs[exp, s, c, "param_access"], fg_params)

        fg_dict = {}
        if not hasattr(freqs_order, "__len__"):
            experiments = self.experiments
        else:
            experiments = freqs_order
        for exp1 in experiments:
            for exp2 in experiments:
                if exp1[:4] != exp2[:4]:
                    exp = "sptg"  # Dodgy ... set all cross correlations between experiemn
                    # ts to irrelevant values
                    c1 = 0
                    c2 = 0
                else:
                    exp = exp1[:4]
                    c1 = self.freqs[exp].index(int(exp1[5:]))
                    c2 = self.freqs[exp].index(int(exp2[5:]))
                for s in self.requested_cls:
                    fg_dict[s, "all", exp1, exp2] = np.zeros(len(ell))
                    for comp in self.fg_component_list[s]:
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
            raise LoggedError(self.log("Using theory forge without mflike, not supported yet ... "))
        else:
            self.log = mflike.log
            self.data_folder = mflike.data_folder
            self.foregrounds = mflike.foregrounds
            self.requested_cls = mflike.requested_cls
            self.expected_params_nuis = mflike.expected_params_nuis
            self.defaults_cuts = mflike.defaults
            self.exp = ["planck"]

            # Initialize foreground model
            self._init_foreground_model()

    # Initializes the foreground model. It sets the SED and reads the templates

    def _init_foreground_model(self):

        from fgspectra import cross as fgc
        from fgspectra import frequency as fgf
        from fgspectra import power as fgp
        import copy
        import pickle
        self.copy = copy

        def get_all_params(param_access):
            params = []
            for val in param_access.values():
                if isinstance(val, str):
                    params.append(val)
                elif isinstance(val, list):
                    for v in val:
                        params.extend(get_all_params(v))
                elif isinstance(val, dict):
                    params.extend(get_all_params(val))
            try:
                params.remove('ell')
            except ValueError:
                pass
            try:
                params.remove('ell_clp')
            except ValueError:
                pass
            return params

        components = {}
        components_list = {}
        expected_params_fg = []
        with open(self.foregrounds["fg_model"], 'rb') as file:
            fg_model = pickle.load(file)
        for exp in self.exp:
            for s in self.requested_cls:
                components_list[s] = []
                for _, (key, value) in enumerate(fg_model[exp, s].items()):
                    components[s, key, "model"] = value['model']
                    components[s, key, "model"].set_defaults(**value['defaults'])
                    components[s, key, "param_access"] = value['param_access']
                    components_list[s].append(key)
                    expected_params_fg.extend(get_all_params(value['param_access']))
        self.fg_component_list = components_list
        self.fgs = components
        self.expected_params_fg = list(set(expected_params_fg))
        if "tsz_and_cib" in components_list["tt"]:
            self.expected_params_fg.append("xi")
            self.expected_params_fg.remove("a_tszxcib")
        if "galactic" in components_list["tt"]:
            self.expected_params_fg.extend(['gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217'])
            self.expected_params_fg.remove("a_gtt_p")
        if "ps" in components_list["tt"]:
            self.expected_params_fg.extend(['ps_A_100_100', 'ps_A_143_143', 'ps_A_143_217', 'ps_A_217_217'])
            self.expected_params_fg.remove("a_ps_p")
        if "galactic" in components_list["te"]:
            self.expected_params_fg.extend(['galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217',
                                            'galf_TE_A_143', 'galf_TE_A_143_217', 'galf_TE_A_217'])
            self.expected_params_fg.remove("a_gte_p")
        if "galactic" in components_list["ee"]:
            self.expected_params_fg.extend(['galf_EE_A_100', 'galf_EE_A_100_143', 'galf_EE_A_100_217',
                                            'galf_EE_A_143', 'galf_EE_A_143_217', 'galf_EE_A_217'])
            self.expected_params_fg.remove("a_gee_p")

    def _evaluate_fgs(self, model, param_access, fg_params):

        def search_for_params(kwargs, base=[]):
            for _, (key, val) in enumerate(kwargs.items()):
                if isinstance(val, list):
                    for v in val:
                        search_for_params(v, base)
                elif isinstance(val, dict):
                    search_for_params(val, base + [key])
                elif val in fg_params:
                    kwargs[key] = fg_params[val]

        modified_params = self.copy.deepcopy(param_access)
        search_for_params(modified_params)
        return model.eval(**modified_params)

    def get_Planck_foreground(self, fg_params, ell, requested_cls=['tt', 'te', 'ee']):
        frequencies = np.asarray([100, 143, 217], dtype=int)
        fg_params['ell'] = ell
        fg_params['ell_clp'] = ell*(ell+1.)
        if "tsz_and_cib" in self.fg_component_list["tt"]:
            fg_params["a_tszxcib"] = -fg_params["xi"] * np.sqrt(fg_params["a_tSZ"] * fg_params["a_CIB"])
        if "galactic" in self.fg_component_list["tt"]:
            gal_amp = np.zeros((len(frequencies), len(frequencies)))
            gal_amp[0, 0] = fg_params['gal545_A_100']
            gal_amp[1, 1] = fg_params['gal545_A_143']
            gal_amp[1, 2] = fg_params['gal545_A_143_217']
            gal_amp[2, 1] = fg_params['gal545_A_143_217']
            gal_amp[2, 2] = fg_params['gal545_A_217']
            fg_params["a_gtt_p"] = gal_amp
        if "ps" in self.fg_component_list["tt"]:
            ps_amp = np.zeros((len(frequencies), len(frequencies)))
            ps_amp[0, 0] = fg_params['ps_A_100_100']
            ps_amp[1, 1] = fg_params['ps_A_143_143']
            ps_amp[1, 2] = fg_params['ps_A_143_217']
            ps_amp[2, 2] = fg_params['ps_A_217_217']
            fg_params['a_ps_p'] = ps_amp
        if "galactic" in self.fg_component_list["te"]:
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
            fg_params["a_gte_p"] = galte_amp
        if "galactic" in self.fg_component_list["ee"]:
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
            fg_params["a_gee_p"] = galee_amp

        model = {}
        for s in self.requested_cls:
            for c in self.fg_component_list[s]:
                model["planck", s, c] = self._evaluate_fgs(self.fgs[s, c, "model"],
                                                           self.fgs[s, c,  "param_access"], fg_params)

        fg_dict = {}
        for idx, (i, j) in enumerate([(0, 0), (1, 1), (1, 2), (2, 2)]):
            f1, f2 = frequencies[i], frequencies[j]
            for comp in self.fg_component_list["tt"]:
                fg_dict["tt", comp, f1, f2] = model["planck", "tt", comp][i, j]

        for i, f1 in enumerate(frequencies):
            for j, f2 in enumerate(frequencies):
                fg_dict['te', 'galactic', f1, f2] = model['planck', 'te', 'galactic'][i, j]
                fg_dict['ee', 'galactic', f1, f2] = model['planck', 'ee', 'galactic'][i, j]

        for c1, f1 in enumerate(frequencies):
            for c2, f2 in enumerate(frequencies):
                for s in requested_cls:
                    fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
                    for comp in self.fg_component_list[s]:
                        if (s, comp, f1, f2) in fg_dict:
                            fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]

        return fg_dict
