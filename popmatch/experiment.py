import inspect
from functools import wraps
import re
from pathlib import Path
import pickle


def dict_router(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            d = args[0]
            # Extract the inputs
            call_args = []
            call_kwargs = {}
            for name, parameter in inspect.signature(func).parameters.items():
                section, varname = name.split('_', 1)
                if section not in d:
                    raise ValueError('[{}] Section {} not in params dictionary'.format(name, section))
                if parameter.default == inspect._empty:
                    if varname not in d[section]:
                        raise ValueError('[{}] Var {} not in params dictionary'.format(name, varname))
                    call_args.append(d[section][varname])
                else:
                    if varname in d[section]:
                        call_kwargs[name] = d[section][varname]
            routed_func = func(*call_args, **call_kwargs)

            return routed_func(*args, **kwargs)
        else:
            # Call the function as usual
            return func(*args, **kwargs)
    return wrapper


def dict_wrapper(*out_names):
    names_to_fill = [re.findall('{([^}]+)}', out_name) for out_name in out_names]
    names_to_fill =  {i for j in names_to_fill for i in j}
    def accept_kwargs_from_dict(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 1 and isinstance(args[0], dict):
                d = args[0]
                # Check if all names to fill are in the kwargs
                assert(names_to_fill.issubset(kwargs.keys()))
                filled_out_names = [out_name.format(**kwargs) for out_name in out_names]
                # Extract the inputs
                call_args = []
                call_kwargs = {}
                for name, parameter in inspect.signature(func).parameters.items():
                    if name in kwargs:
                        # We allow overriding some parameters in the kwargs
                        call_kwargs[name] = kwargs[name]
                        continue
                    section, varname = name.split('_', 1)
                    if section in kwargs:
                        section = kwargs[section]
                    if section not in d:
                        raise ValueError('[{}] Section {} not in params dictionary'.format(name, section))
                    if parameter.default == inspect._empty:
                        if varname not in d[section]:
                            raise ValueError('[{}] Var {} not in params dictionary'.format(name, varname))
                        call_args.append(d[section][varname])
                    else:
                        if varname in d[section]:
                            call_kwargs[name] = d[section][varname]
                result = func(*call_args, **call_kwargs)
                if len(out_names) == 1:
                    result = [result]
                assert(len(result) == len(out_names))
                print(filled_out_names)
                for value, name in zip(result, filled_out_names):
                    section, varname = name.split('_', 1)
                    if section not in d:
                        d[section] = {}
                    d[section][varname] = value
                return result
            else:
                # Call the function as usual
                return func(*args, **kwargs)
        return wrapper
    return accept_kwargs_from_dict


def dict_cache(experiment, section, cache_path='./'):
    result_path = Path(cache_path) / (section + '.pkl')
    if not result_path.exists():
        yield None
        with open(str(result_path), 'wb') as file:
            pickle.dump(experiment[section], file)
    else:
        with open(str(result_path), 'rb') as file:
            experiment[section] = pickle.load(file)
    raise StopIteration
