import numpy as np


def check_parms(
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
    parm_dict: dict
        The dictionary in which the a parameter will be checked
    string_key :
    acceptable_data_types : list
    acceptable_data : list
    acceptable_range : list (length of 2)
    list_acceptable_data_types : list
    list_len : int
        If list_len is -1 than the list can be any length.
    default :
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
            parms_passed = True

            if default is None:
                print(
                    "######### ERROR:Dictionary parameters must have a default. Please report bug."
                )
                return False
            # print('is a dict',default)
            for default_element in default:
                if default_element in parm_dict[string_key]:
                    # print('1.*******')
                    # print(parm_dict[string_key], default_element, [type(default[default_element])], default[default_element])
                    if not (
                        check_parms(
                            parm_dict[string_key],
                            default_element,
                            [type(default[default_element])],
                            default=default[default_element],
                        )
                    ):
                        parms_passed = False
                    # print('2.*******')
                else:
                    # print('parm_dict',default_element,string_key)
                    parm_dict[string_key][default_element] = default[default_element]
        #                    print(
        #                        "Setting default",
        #                        string_key,
        #                        "['",
        #                        default_element,
        #                        "']",
        #                        " to ",
        #                        default[default_element],
        #                    )
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
    try:
        temp = vis_dataset[data_variable_name]
    except:
        print(
            "######### ERROR Data array ",
            data_variable_name,
            "can not be found in dataset.",
        )
        return False
    return True


def check_sel_parms(
    xds,
    sel_parms,
    new_or_modified_data_variables={},
    default_data_group_out=None,
    skip_data_group_in=False,
    skip_data_group_out=False,
):
    """

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        Input vis.zarr multi dataset.
    sel_parms : dictionary
    Returns
    -------
    psf_dataset : xarray.core.dataset.Dataset
    """

    assert "data_groups" in xds.attrs, "No data_groups found in ms_xds."

    if "overwrite" not in sel_parms:
        sel_parms["overwrite"] = False
        overwrite = False
    else:
        overwrite = sel_parms["overwrite"]

    if not skip_data_group_in:
        if not ("data_group_in" in sel_parms):
            data_group_in_name = "base"
            sel_parms["data_group_in"] = {
                data_group_in_name: xds.attrs["data_groups"][data_group_in_name]
            }

        if isinstance(sel_parms["data_group_in"], str):
            data_group_in_name = sel_parms["data_group_in"]

            # print(data_group_in_name,xds.attrs["data_groups"])
            sel_parms["data_group_in"] = {
                sel_parms["data_group_in"]: xds.attrs["data_groups"][
                    sel_parms["data_group_in"]
                ]
            }
        else:
            data_group_in_name = list(sel_parms["data_group_in"].keys())[0]

        if data_group_in_name not in xds.attrs["data_groups"]:
            xds.attrs["data_groups"][data_group_in_name] = sel_parms["data_group_in"][
                data_group_in_name
            ]

        data_group_in = xds.attrs["data_groups"][data_group_in_name]
    else:
        data_group_in = None

    if default_data_group_out is not None:
        default_data_group_out_name = list(default_data_group_out.keys())[0]
        new_or_modified_data_variables = list(
            default_data_group_out[default_data_group_out_name].keys()
        )

        default_data_group_out[default_data_group_out_name] = {
            **sel_parms["data_group_in"][data_group_in_name],
            **default_data_group_out[default_data_group_out_name],
        }
    else:
        default_data_group_out = None

    if not skip_data_group_out:
        if not ("data_group_out_name" in sel_parms):
            sel_parms["data_group_out"] = default_data_group_out

        if isinstance(sel_parms["data_group_out"], str):
            sel_parms["data_group_out"] = {
                sel_parms["data_group_out"]: list(default_data_group_out.values)[0]
            }

        data_group_out_name = list(sel_parms["data_group_out"].keys())[0]

        if not overwrite:
            for nm_dv in new_or_modified_data_variables:
                assert (
                    sel_parms["data_group_out"][data_group_out_name][nm_dv] not in xds
                ), (
                    sel_parms["data_group_out"][data_group_out_name][nm_dv]
                    + " already present in xds. Set overwrite to True if data variable should be overwritten."
                )

            # assert data_group_out_name not in xds.attrs["data_groups"], "Data group " + data_group_out_name + " already in xds data_groups. Set overwrite to True if data group should be overwritten."

        xds.attrs["data_groups"][data_group_out_name] = sel_parms["data_group_out"][
            data_group_out_name
        ]

        data_group_out = xds.attrs["data_groups"][data_group_out_name]
    else:
        data_group_out = None

    return data_group_in, data_group_out


def _check_sub_sel_parms(sel_parms, select_defaults):
    parms_passed = True
    for sel in select_defaults:
        if not (
            check_parms(
                sel_parms,
                sel,
                [type(select_defaults[sel])],
                default=select_defaults[sel],
            )
        ):
            parms_passed = False
    return parms_passed


"""
def _check_sel_parms(sel_parms,select_defaults):
    parms_passed = True
    for sel_def in select_defaults:
        if isinstance(select_defaults[sel_def], dict):
            if sel_def in sel_parms:
                for sub_sel_def in select_defaults[sel_def]:
                        #print(sub_sel_def,select_defaults[sel_def])
                        #print(sel_parms[sel_def], sub_sel_def, select_defaults[sel_def][sub_sel_def])
                        if not(check_parms(sel_parms[sel_def], sub_sel_def, [str], default=select_defaults[sel_def][sub_sel_def])): parms_passed = False
            else:
                sel_parms[sel_def] = select_defaults[sel_def]
                print ('Setting default', string_key, ' to ', parm_dict[string_key])
        else:
            if not(check_parms(sel_parms, sel_def, [str], default=select_defaults[sel_def])): parms_passed = False
    return parms_passed
"""


def _check_existence_sel_parms(dataset, sel_parms):
    parms_passed = True
    for sel in sel_parms:
        if isinstance(sel_parms[sel], dict):
            if sel != "properties":
                _check_existence_sel_parms(dataset, sel_parms[sel])
        else:
            if (sel != "id") and (sel != "properties"):
                if not (_check_dataset(dataset, sel_parms[sel])):
                    parms_passed = False
    return parms_passed
