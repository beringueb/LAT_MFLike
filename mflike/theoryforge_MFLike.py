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

    def _construct_fgs(self, component):
        keys = list(component.keys())
        sed = getattr(self.fgf, keys[0])()
        sed_dict = component[keys[0]]
        cl_dict = component[keys[1]]
        cl_dict["ell"] = self.l_bpws
        sed_dict["nu"] = np.fromstring(sed_dict["nu"], dtype=float, sep=',')
        if keys[0] == "FreeSED":
            sed_dict["sed"] = np.fromstring(sed_dict["sed"], dtype=float, sep=',')
        if keys[1] == "PowerSpectrumFromFile":
            try:
                template = self.fgp._get_power_file(cl_dict["file"])
            except ValueError:
                raise LoggedError(self.log, f"Check if you have template {cl_dict['file']} for {keys[1]}")
            cl = getattr(self.fgp, keys[1])(template)
            del cl_dict["file"]
        elif keys[1][:-3] == "PowerLaw":
            cl = getattr(self.fgp, keys[1][:-3])()
        else:
            cl = getattr(self.fgp, keys[1])()
        params = component["params"]
        param_access = {'sed_kwargs': {}, 'cl_kwargs': {}}
        for key, value in sed_dict.items():
            if params is None: continue
            if value in params:
                param_access['sed_kwargs'][key] = value
        for key, value in cl_dict.items():
            if params is None: continue
            if value in params:
                param_access['cl_kwargs'][key] = value
        for key in param_access['sed_kwargs']:
            del sed_dict[key]
        for key in param_access['cl_kwargs']:
            del cl_dict[key]
        sed.set_defaults(**sed_dict)
        cl.set_defaults(**cl_dict)
        model = self.fgc.FactorizedCrossSpectrum(sed, cl)
        return model, param_access

    def _construct_tszxcib(self, component):
        sed_keys = list(component.keys())

        sed_list = [getattr(self. fgf, sed_key)() for sed_key in component[sed_keys[0]].keys()]

        sed_dict_keys = list(component[sed_keys[0]].keys())
        sed_dict_list = [component[sed_keys[0]][sed_dict_key] for sed_dict_key in sed_dict_keys]
        for d in sed_dict_list:
            d["nu"] = np.fromstring(d["nu"], dtype=float, sep=',')

        cl_dict_keys = list(component[sed_keys[1]].keys())
        cl_dict_list = [component[sed_keys[1]][cl_dict_key] for cl_dict_key in cl_dict_keys]
        cl_list = []
        for d in cl_dict_list:
            d["ell"] = self.l_bpws

        for i, cl_key in enumerate(cl_dict_keys):
            if cl_key[:-4] == "PowerSpectrumFromFile":
                file = cl_dict_list[i]["file"]
                try:
                    template = self.fgp._get_power_file(file)
                except ValueError:
                    raise LoggedError(self.log, f"Check if you have template {file} for {cl_key}")
                cl_list.append(getattr(self.fgp, cl_key[:-4])(template))
                del cl_dict_list[i]["file"]
            else:
                cl_list.append(getattr(self.fgp, cl_key)())

        params = component["params"]

        param_access = {
            'sed_kwargs': {
                "kwseq": [
                    {key: value for key, value in sed_dict.items() if value in params}
                    for sed_dict in sed_dict_list
                ]
            },
            'cl_kwargs': {
                "kwseq": [
                    {key: value for key, value in cl_dict.items() if value in params}
                    for cl_dict in cl_dict_list
                ]
            }
        }

        model = self.fgc.CorrelatedFactorizedCrossSpectrum(self.fgf.Join(*sed_list), self.fgp.PowerSpectraAndCovariance(*cl_list))
        model.set_defaults(**{"sed_kwargs": {"kwseq": sed_dict_list}, "cl_kwargs": {"kwseq": cl_dict_list}})

        return model, param_access

    def _init_foreground_model(self):

        from fgspectra import cross as fgc
        from fgspectra import frequency as fgf
        from fgspectra import power as fgp
        self.fgp = fgp
        self.fgc = fgc
        self.fgf = fgf

        def get_all_params(param_access):
            params = []
            for value in param_access.values():
                if isinstance(value, str):
                    params.append(value)
                elif isinstance(value, dict):
                    params.extend(get_all_params(value))
            return params
        components = {}
        components_list = {}
        expected_params_fg = []
        for exp in self.exp:
            for s in self.requested_cls:
                components_list[s] = []
                for _, (key, value) in enumerate(self.foregrounds["components"][s][exp].items()):
                    if key.lower() in ["tszxcibc", "tszxcib", "tsz_and_cibc", "tsz_and cib", "tsz_x_cibc", "tsz_x_cib"]:
                        model, param_access = self._construct_tszxcib(value)
                    else:
                        model, param_access = self._construct_fgs(value)
                    components[exp, s, key, "model"] = model
                    components[exp, s, key, "param_access"] = param_access
                    components_list[s].append(key)
                    expected_params_fg.extend(get_all_params(param_access))
        self.fg_component_list = components_list
        self.fgs = components
        self.expected_params_fg = list(set(expected_params_fg))
        if "tszxcibc" in components_list["tt"]:
            self.expected_params_fg.append("xi")
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

        modified_params = param_access.copy()
        search_for_params(modified_params)
        return model.eval(**modified_params)

    # Gets the actual power spectrum of foregrounds given the passed parameters
    def _get_foreground_model(self, ell=None, freqs_order=None, **fg_params):
        # if ell = None, it uses the l_bpws, otherwise the ell array provided
        # useful to make tests at different l_max than the data
        if not hasattr(ell, "__len__"):
            ell = self.l_bpws

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
                        if comp == "tszxcibc":
                            fg_dict[s, "tSZ", exp1, exp2] = model[exp, s, "tsz"][c1, c2]
                            fg_dict[s, "cibc", exp1, exp2] = model[exp, s, "cibc"][c1, c2]
                            fg_dict[s, "tSZxCIB", exp1, exp2] = (
                                model[exp, s, comp][c1, c2]
                                - model[exp, s, "tsz"][c1, c2]
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
            raise LoggedError(self.log("Using theory forge without mflike, not supported yet ... "))
        else:
            self.log = mflike.log
            self.data_folder = mflike.data_folder
            self.foregrounds = mflike.foregrounds
            self.requested_cls = mflike.requested_cls
            self.expected_params_nuis = mflike.expected_params_nuis
            self.defaults_cuts = mflike.defaults
            self.exp = ["planck"]
            self.l_bpws = mflike.l_bpws

            # Initialize foreground model
            self._init_foreground_model()

    # Initializes the foreground model. It sets the SED and reads the templates

    def _construct_fgs(self, component):
        keys = list(component.keys())
        sed = getattr(self.fgf, keys[0])()
        sed_dict = component[keys[0]]
        cl_dict = component[keys[1]]
        cl_dict["ell"] = self.l_bpws
        sed_dict["nu"] = np.fromstring(sed_dict["nu"], dtype=float, sep=',')
        if keys[1] == "PowerSpectrumFromFile":
            try:
                template = self.fgp._get_power_file(cl_dict["file"])
            except ValueError:
                raise LoggedError(self.log, f"Check if you have template {cl_dict['file']} for {keys[1]}")
            cl = getattr(self.fgp, keys[1])(template)
            del cl_dict["file"]
        elif keys[1][:-3] == "PowerLaw":
            cl = getattr(self.fgp, keys[1][:-3])()
        else:
            cl = getattr(self.fgp, keys[1])()
        params = component["params"]
        param_access = {'sed_kwargs': {}, 'cl_kwargs': {}}
        for key, value in sed_dict.items():
            if value in params:
                param_access['sed_kwargs'][key] = value
        for key, value in cl_dict.items():
            if value in params:
                param_access['cl_kwargs'][key] = value
        for key in param_access['sed_kwargs']:
            del sed_dict[key]
        for key in param_access['cl_kwargs']:
            del cl_dict[key]
        sed.set_defaults(**sed_dict)
        cl.set_defaults(**cl_dict)
        model = self.fgc.FactorizedCrossSpectrum(sed, cl)
        return model, param_access

    def _construct_tszxcib(self, component):
        sed_keys = list(component.keys())

        sed_list = [getattr(self. fgf, sed_key)() for sed_key in component[sed_keys[0]].keys()]

        sed_dict_keys = list(component[sed_keys[0]].keys())
        sed_dict_list = [component[sed_keys[0]][sed_dict_key] for sed_dict_key in sed_dict_keys]
        for d in sed_dict_list:
            d["nu"] = np.fromstring(d["nu"], dtype=float, sep=',')

        cl_dict_keys = list(component[sed_keys[1]].keys())
        cl_dict_list = [component[sed_keys[1]][cl_dict_key] for cl_dict_key in cl_dict_keys]
        cl_list = []
        for d in cl_dict_list:
            d["ell"] = self.l_bpws

        for i, cl_key in enumerate(cl_dict_keys):
            if cl_key[:-4] == "PowerSpectrumFromFile":
                file = cl_dict_list[i]["file"]
                try:
                    template = self.fgp._get_power_file(file)
                except ValueError:
                    raise LoggedError(self.log, f"Check if you have template {file} for {cl_key}")
                cl_list.append(getattr(self.fgp, cl_key[:-4])(template))
                del cl_dict_list[i]["file"]
            else:
                cl_list.append(getattr(self.fgp, cl_key)())

        params = component["params"]

        param_access = {
            'sed_kwargs': {
                "kwseq": [
                    {key: value for key, value in sed_dict.items() if value in params}
                    for sed_dict in sed_dict_list
                ]
            },
            'cl_kwargs': {
                "kwseq": [
                    {key: value for key, value in cl_dict.items() if value in params}
                    for cl_dict in cl_dict_list
                ]
            }
        }

        model = self.fgc.CorrelatedFactorizedCrossSpectrum(self.fgf.Join(*sed_list), self.fgp.PowerSpectraAndCovariance(*cl_list))
        model.set_defaults(**{"sed_kwargs": {"kwseq": sed_dict_list}, "cl_kwargs": {"kwseq": cl_dict_list}})

        return model, param_access

    def _init_foreground_model(self):

        from fgspectra import cross as fgc
        from fgspectra import frequency as fgf
        from fgspectra import power as fgp
        self.fgp = fgp
        self.fgc = fgc
        self.fgf = fgf

        def get_all_params(param_access):
            params = []
            for value in param_access.values():
                if isinstance(value, str):
                    params.append(value)
                elif isinstance(value, dict):
                    params.extend(get_all_params(value))
            return params
        components = {}
        components_list = {}
        expected_params_fg = []
        for exp in self.exp:
            for s in self.requested_cls:
                components_list[s] = []
                for _, (key, value) in enumerate(self.foregrounds["components"][s][exp].items()):
                    if key.lower() in ["tszxcibc", "tszxcib", "tsz_and_cibc", "tsz_and cib", "tsz_x_cibc", "tsz_x_cib"]:
                        model, param_access = self._construct_tszxcib(value)
                    else:
                        model, param_access = self._construct_fgs(value)
                    components[exp, s, key, "model"] = model
                    components[exp, s, key, "param_access"] = param_access
                    components_list[s].append(key)
                    expected_params_fg.extend(get_all_params(param_access))
        self.fg_component_list = components_list #TODO : Component list to depend on exp as well ?
        self.fgs = components
        self.expected_params_fg = list(set(expected_params_fg))
        if "tszxcibc" in components_list["tt"]:
            self.expected_params_fg.append("xi")
        if "gal_cirrus" in components_list["tt"]:
            self.expected_params_fg.extend(['gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217'])
        if "gal_cirrus" in components_list["te"]:
            self.expected_params_fg.extend(['galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217',
                                            'galf_TE_A_143', 'galf_TE_A_143_217', 'galf_TE_A_217'])
        if "gal_cirrus" in components_list["ee"]:
            self.expected_params_fg.extend(['galf_EE_A_100', 'galf_EE_A_100_143', 'galf_EE_A_100_217',
                                            'galf_EE_A_143', 'galf_EE_A_143_217', 'galf_EE_A_217'])


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

        modified_params = param_access.copy()
        search_for_params(modified_params)
        return model.eval(**modified_params)


    def get_Planck_foreground(self, fg_params, ell, requested_cls=['tt', 'te', 'ee']):
        frequencies = np.asarray([100, 143, 217], dtype=int)

        if "tszxcibc" in self.fg_component_list["tt"]:
            fg_params["a_tszxcib"] = -fg_params["xi"] * np.sqrt(fg_params["a_tSZ"] * fg_params["a_CIB"])
        if "gal_cirrus" in self.fg_component_list["tt"]:
            gal_amp = np.zeros((len(frequencies), len(frequencies)))
            gal_amp[0, 0] = fg_params['gal545_A_100']
            gal_amp[1, 1] = fg_params['gal545_A_143']
            gal_amp[1, 2] = fg_params['gal545_A_143_217']
            gal_amp[2, 1] = fg_params['gal545_A_143_217']
            gal_amp[2, 2] = fg_params['gal545_A_217']
            fg_params["amp_gal_tt"] = gal_amp
        if "gal_cirrus" in self.fg_component_list["te"]:
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
            fg_params["amp_gal_te"] = galte_amp
        if "gal_cirrus" in self.fg_component_list["ee"]:
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
            fg_params["amp_gal_ee"] = galee_amp

        model = {}
        for s in self.requested_cls:
            for c in self.fg_component_list[s]:
                model["planck", s, c] = self._evaluate_fgs(self.fgs["planck", s, c, "model"],
                                                           self.fgs["planck", s, c, "param_access"], fg_params)

        fg_dict = {}
        for idx, (i, j) in enumerate([(0, 0), (1, 1), (1, 2), (2, 2)]):
            f1, f2 = frequencies[i], frequencies[j]
            for comp in self.fg_component_list["tt"]:
                if comp == "tszxcibc":
                    fg_dict["tt", "tSZ", f1, f2] = model["planck", "tt", "tsz"][i, j]
                    fg_dict["tt", "cibc", f1, f2] = model["planck", "tt", "cibc"][i, j]
                    fg_dict["tt", "tSZxCIB", f1, f2] = (
                            model["planck", "tt", comp][i, j]
                            - model["planck", "tt", "tsz"][i, j]
                            - model["planck", "tt", "cibc"][i, j]
                    )
                    # fg_dict["tt", "all", f1, f2] += model["planck", "tt", comp][i, j]
                    # fg_dict[s, "all", exp1, exp2] += fg_dict[s, "tSZxCIB", exp1, exp2]
                else:
                    fg_dict["tt", comp, f1, f2] = model["planck", "tt", comp][i, j]
                    # fg_dict["tt", "all", f1, f2] += fg_dict["tt", comp, f1, f2]

        for i, f1 in enumerate(frequencies):
            for j, f2 in enumerate(frequencies):
                fg_dict['te', 'gal_cirrus', f1, f2] = model['planck', 'te', 'gal_cirrus'][i, j]
                fg_dict['ee', 'gal_cirrus', f1, f2] = model['planck', 'ee', 'gal_cirrus'][i, j]

        for c1, f1 in enumerate(frequencies):
            for c2, f2 in enumerate(frequencies):
                for s in requested_cls:
                    fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
                    for comp in self.fg_component_list[s]:
                        if (s, comp, f1, f2) in fg_dict:
                            fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]

        return fg_dict
