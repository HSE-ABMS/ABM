from PyQt6.QtWidgets import *

import VisualInterface.app
from VisualInterface.app import MainWindow, recursive_bruteforce
from VisualInterface.utils import *
import sys

if __name__ == '__main__':
    # print(get_agent_arg_type("Chartist", "cash"))
    # from AgentBasedModel import *
    #
    # exchange = ExchangeAgent(volume=1000)
    # simulator = Simulator(**{
    #     'exchanges': [exchange],
    #     'traders': [Universalist([exchange], 10 ** 3, [0]) for _ in range(20)],
    #     'events': [MarketPriceShock(200, -10)]
    # })
    # info = simulator.info
    # simulator.simulate(500)
    #
    # plot_price_fundamental(info[0])


    # print(get_stats_list())
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

    # print(*VisualInterface.app.recursive_bruteforce([0, 0], 0, [[0, 1], [1, 2]]))
