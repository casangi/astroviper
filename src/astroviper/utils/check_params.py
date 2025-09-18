import toolviper.utils.logger as logger


def check_params(
    parm_dict,
    string_key,
    acceptable_data_types,
    acceptable_data=None,
    acceptable_range=None,
    list_acceptable_data_types=None,
    list_len=None,
    default=None,
):
    """

    Parameters
    ----------
    parm_dict : dict
        The dictionary in which the a parameter will be checked
    string_key : str
        The key of the parameter to check
    acceptable_data_types : list
        A list of acceptable data types for the parameter
    acceptable_data : list
        A list of acceptable values for the parameter
    acceptable_range : list
        A list of two elements specifying the acceptable range for the parameter
    list_acceptable_data_types : list
        A list of acceptable data types for list elements
    list_len : int
        If list_len is -1 than the list can be any length.
    default : dict
        A dictionary of default values for the parameters

    Returns
    -------
    parm_passed : bool

    """

    import numpy as np

    if string_key in parm_dict:
        if (list in acceptable_data_types) or (np.array in acceptable_data_types):
            if (len(parm_dict[string_key]) != list_len) and (list_len != -1):
                print(
                    "######### ERROR:Parameter ",
                    string_key,
                    "must be a list of ",
                    list_acceptable_data_types,
                    " and length",
                    list_len,
                    ". Wrong length.",
                )
                return False
            else:
                list_len = len(parm_dict[string_key])
                for i in range(list_len):
                    type_check = False
                    for lt in list_acceptable_data_types:
                        if isinstance(parm_dict[string_key][i], lt):
                            type_check = True
                    if not (type_check):
                        print(
                            "######### ERROR:Parameter ",
                            string_key,
                            "must be a list of ",
                            list_acceptable_data_types,
                            " and length",
                            list_len,
                            ". Wrong type of",
                            type(parm_dict[string_key][i]),
                        )
                        return False

                    if acceptable_data is not None:
                        if not (parm_dict[string_key][i] in acceptable_data):
                            print(
                                "######### ERROR: Invalid",
                                string_key,
                                ". Can only be one of ",
                                acceptable_data,
                                ".",
                            )
                            return False

                    if acceptable_range is not None:
                        if (parm_dict[string_key][i] < acceptable_range[0]) or (
                            parm_dict[string_key][i] > acceptable_range[1]
                        ):
                            print(
                                "######### ERROR: Invalid",
                                string_key,
                                ". Must be within the range ",
                                acceptable_range,
                                ".",
                            )
                            return False
        elif dict in acceptable_data_types:
            params_passed = True

            if default is None:
                print(
                    "######### ERROR:Dictionary parameters must have a default. Please report bug."
                )
                return False
            # print('is a dict',default)
            for default_element in default:
                if default_element in parm_dict[string_key]:
                    if not (
                        check_params(
                            parm_dict[string_key],
                            default_element,
                            [type(default[default_element])],
                            default=default[default_element],
                        )
                    ):
                        params_passed = False
                else:
                    parm_dict[string_key][default_element] = default[default_element]
        else:
            type_check = False
            for adt in acceptable_data_types:
                if isinstance(parm_dict[string_key], adt):
                    type_check = True
            if not (type_check):
                print(
                    "######### ERROR:Parameter ",
                    string_key,
                    "must be of type ",
                    acceptable_data_types,
                )
                return False

            if acceptable_data is not None:
                if not (parm_dict[string_key] in acceptable_data):
                    print(
                        "######### ERROR: Invalid",
                        string_key,
                        ". Can only be one of ",
                        acceptable_data,
                        ".",
                    )
                    return False

            if acceptable_range is not None:
                if (parm_dict[string_key] < acceptable_range[0]) or (
                    parm_dict[string_key] > acceptable_range[1]
                ):
                    print(
                        "######### ERROR: Invalid",
                        string_key,
                        ". Must be within the range ",
                        acceptable_range,
                        ".",
                    )
                    return False
    else:
        if default is not None:
            # print(parm_dict, string_key,  default)
            parm_dict[string_key] = default
            # print("Setting default", string_key, " to ", parm_dict[string_key])
        else:
            print("######### ERROR:Parameter ", string_key, "must be specified")
            return False

    return True


def _check_dataset(vis_dataset, data_variable_name):
    """
    Check if a data variable exists in the visualization dataset.
    """
    try:
        temp = vis_dataset[data_variable_name]
    except KeyError:
        print(
            "######### ERROR Data array ",
            data_variable_name,
            "can not be found in dataset.",
        )
        return False
    return True


def lists_overlap(a, b):
    return bool(set(a) & set(b))


import xarray as xr


def check_sel_params_ps_xdt(
    ps_xdt: xr.DataTree,
    sel_params: dict,
    default_data_group_in_name: str = None,
    default_data_group_out_name: str = None,
    default_data_group_out_modified: dict = None,
):
    data_group_in_list = []
    data_group_out_list = []
    for ms_xdt in ps_xdt.values():
        data_group_in, data_group_out = check_sel_params(
            ms_xdt.ds,
            sel_params,
            default_data_group_in_name=default_data_group_in_name,
            default_data_group_out_name=default_data_group_out_name,
            default_data_group_out_modified=default_data_group_out_modified,
        )
        data_group_out_list.append(data_group_out)
        data_group_in_list.append(data_group_in)

    IGNORE_KEYS = {"date", "description"}

    def normalize(d):
        return {k: v for k, v in d.items() if k not in IGNORE_KEYS}

    assert all(
        normalize(d) == normalize(data_group_out_list[0]) for d in data_group_out_list
    ), "data_group_out must be the same for all measurement sets in the processing set."

    assert all(
        normalize(d) == normalize(data_group_in_list[0]) for d in data_group_in_list
    ), "data_group_in must be the same for all measurement sets in the processing set."

    return data_group_in_list[0], data_group_out_list[0]


def check_sel_params(
    xds: xr.Dataset,
    sel_params: dict,
    default_data_group_in_name: str = None,
    default_data_group_out_name: str = None,
    default_data_group_out_modified: dict = None,
):
    """Check selection parameters for imaging weights calculation.

    Parameters
    ----------
    xds : xr.Dataset
        The input dataset.
    sel_params : dict
        The selection parameters.
    default_data_group_in_name : str, optional
        The default input data group name, by default None
    default_data_group_out_name : str, optional
        The default output data group name, by default None
    default_data_group_out_modified : dict, optional
        The default modified output data group, by default {}
    sel_params : dict
        The selection parameters.
    default_data_group_in_name : str, optional
        The default input data group name, by default None
    default_data_group_out_name : str, optional
        The default output data group name, by default None
    default_data_group_out_modified : dict, optional
        The default modified output data group, by default {}

    Returns
    -------
    data_group_in : dict
        The input data group.
    data_group_out : dict
        The output data group.
    """

    if default_data_group_out_modified is None:
        default_data_group_out_modified = {}

    xds_dv_names = list(xds.data_vars)
    import copy

    xds_data_groups = copy.deepcopy(xds.attrs.get("data_groups", {}))

    # Check the data_group_in
    if "data_group_in_name" in sel_params:
        assert sel_params["data_group_in_name"] in xds_data_groups, (
            "Data group "
            + sel_params["data_group_in_name"]
            + " not found in xds data_groups: "
            + str(xds_data_groups.keys())
        )
        assert sel_params["data_group_in_name"] in xds_data_groups, (
            "Data group "
            + sel_params["data_group_in_name"]
            + " not found in xds data_groups: "
            + str(list(xds_data_groups.keys()))
        )
        data_group_in = xds_data_groups[sel_params["data_group_in_name"]]
        data_group_in["data_group_in_name"] = sel_params["data_group_in_name"]
    else:
        assert default_data_group_in_name in xds_data_groups, (
            "Default data group "
            + default_data_group_in_name
            + " not found in xds data_groups: "
            + str(xds_data_groups.keys())
        )
        data_group_in = xds_data_groups[default_data_group_in_name]
        data_group_in["data_group_in_name"] = default_data_group_in_name

    # Check if data_group_out. Three use cases
    if ("data_group_out_name" in sel_params) and ("data_group_out" in sel_params):
        data_group_out = sel_params["data_group_out"]
        data_group_out["data_group_out_name"] = sel_params["data_group_out_name"]
    elif (
        "data_group_out_name" in sel_params
    ):  # Only data_group_out_name is given so use defaults
        data_group_out = default_data_group_out_modified
        data_group_out["data_group_out_name"] = sel_params["data_group_out_name"]
    elif "data_group_out" in sel_params:  # Only data_group_out is given so use defaults
        data_group_out = sel_params["data_group_out"]
        data_group_out["data_group_out_name"] = default_data_group_out_name
    else:  # No data_group_out is given so use defaults
        data_group_out = default_data_group_out_modified
        data_group_out["data_group_out_name"] = default_data_group_out_name

    # Add any missing data variables from default. These get created or modified.
    # Keys in data_group_out will take precedence over default_data_group_out_modified if there are conflicts.
    data_group_out = {**default_data_group_out_modified, **data_group_out}

    if "overwrite" not in sel_params:
        sel_params["overwrite"] = False

    if sel_params["overwrite"]:

        for data_group_name, data_group in xds_data_groups.items():
            data_group_values = data_group.values()
            data_group_out_values = data_group_out.values()

            # print(data_group_out_values, data_group_values)
            if lists_overlap(data_group_values, data_group_out_values):
                logger.debug(
                    f"Warning: Overwriting data variables in existing data group {data_group_name} since overwrite=True."
                )
                # Delete old data group:
                del xds.attrs["data_groups"][data_group_name]
    else:
        for dv_name in data_group_out.values():
            assert (
                dv_name not in xds_dv_names
            ), f"Data variable {dv_name} already exists in xds."

    # Merge data_group_in and data_group_out.
    # Keys in data_group_out will take precedence over data_group_in if there are conflicts.
    data_group_out = {**data_group_in, **data_group_out}

    return data_group_in, data_group_out
