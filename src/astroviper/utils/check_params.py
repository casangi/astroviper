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
    """Check if a data variable exists in a dataset.

    Parameters
    ----------
    vis_dataset : xr.Dataset
        The dataset to search.
    data_variable_name : str
        Name of the data variable to look up.

    Returns
    -------
    bool
        ``True`` if the variable is present, ``False`` otherwise.
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
    """Check whether two lists share at least one common element.

    Parameters
    ----------
    a : list
        First list.
    b : list
        Second list.

    Returns
    -------
    bool
        ``True`` if the lists have a non-empty intersection.
    """
    return bool(set(a) & set(b))
