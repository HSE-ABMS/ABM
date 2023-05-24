import AgentBasedModel


def load_simulator(cfg: dict) -> AgentBasedModel.Simulator:
    exchanges = [AgentBasedModel.ExchangeAgent(**exc) for exc in cfg['exchanges']]
    traders = []
    events = []

    dynamic_agents = {"IntradayTrader"}
    for params in cfg["traders"]:
        params = params.copy()
        _type = params.pop("type")
        _count = params.pop("count")
        params["markets"] = [exchanges[_] for _ in params["markets"]]

        AgentClass = getattr(AgentBasedModel.agents, _type)
        if _type in dynamic_agents:
            if params["trader_type"] in dynamic_agents:
                raise TypeError(f"{params['trader_type']} can't be used as IntradayTrader's trader_type")
            _trader_type = getattr(AgentBasedModel.agents, params.pop("trader_type"))
            AgentClass = type(_type, (_trader_type, AgentClass), {})
        traders.extend(
            AgentClass(**params) for _ in range(_count)
        )

    for params in cfg["events"]:
        params = params.copy()
        _type = params.pop("type")
        events.append(
            getattr(AgentBasedModel.events, _type)(**params)
        )

    simulator = AgentBasedModel.Simulator(
        exchanges=exchanges,
        traders=traders,
        events=events
    )
    return simulator
