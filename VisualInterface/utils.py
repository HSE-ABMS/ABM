# from __future__ import annotations

import typing

import AgentBasedModel
import inspect
from typing import get_type_hints


def get_constructor(class_type, class_name):
    return getattr(getattr(AgentBasedModel, class_type), class_name).__init__


def get_default_value(class_type, class_name, param):
    sig = None
    if class_type == "stats":
        sig = inspect.signature(getattr(AgentBasedModel.visualization, class_name))
    else:
        sig = inspect.signature(get_constructor(class_type, class_name))
    x = sig.parameters[param]
    if x.default is not inspect.Parameter.empty:
        return x.default
    return None


def get_fields(class_type, class_name):
    if class_type == "stats":
        return list(typing.get_type_hints(getattr(AgentBasedModel.visualization, class_name)).keys())
    return list(typing.get_type_hints(get_constructor(class_type, class_name)).keys())


def get_arg_type_dict(class_type, class_name):
    if class_type == "stats":
        return typing.get_type_hints(getattr(AgentBasedModel.visualization, class_name))
    print(typing.get_type_hints(get_constructor(class_type, class_name)))
    return typing.get_type_hints(get_constructor(class_type, class_name))


def get_parameter_info(class_type, class_name, param):
    docstring = None
    if class_type == "stats":
        docstring = inspect.getdoc(getattr(AgentBasedModel.visualization, class_name))
    else:
        docstring = inspect.getdoc(get_constructor(class_type, class_name))
    if docstring is not None:
        for s in docstring.split("\n"):
            if ":param " + param not in s:
                continue
            return s[s.find(":", 7) + 1:]
    return "No docs for this parameter"


def get_agents_list():
    for s in dir(AgentBasedModel.agents):
        if inspect.isclass(getattr(AgentBasedModel.agents, s)) and issubclass(getattr(AgentBasedModel.agents, s),
                                                                              AgentBasedModel.agents.Trader) and s != "Trader":
            yield s


def get_events_list():
    for s in dir(AgentBasedModel.events):
        if inspect.isclass(getattr(AgentBasedModel.events, s)) and issubclass(getattr(AgentBasedModel.events, s),
                                                                              AgentBasedModel.events.Event) and s != "Event":
            yield s


def get_brokers_list():
    for s in dir(AgentBasedModel.agents):
        if inspect.isclass(getattr(AgentBasedModel.agents, s)) and issubclass(getattr(AgentBasedModel.agents, s),
                                                                              AgentBasedModel.agents.Broker) and s != "Broker":
            yield s


def get_stats_list():
    for s in dir(AgentBasedModel.visualization):
        if s.startswith("plot_"):
            yield s


def can_bruteforce(class_type, class_name, param):
    if class_name == "stats":
        return False
    if param == "count":
        return True
    di = get_arg_type_dict(class_type, class_name)
    return di[param] in [int, float, float | int]


def is_list_parameter(class_type, class_name, param):
    return "List" in str(get_arg_type_dict(class_type, class_name)[param])
