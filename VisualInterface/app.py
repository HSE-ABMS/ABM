import datetime

from PyQt6.QtWidgets import QMessageBox, QMainWindow, QVBoxLayout, QLabel, QGroupBox, QTextBrowser, QFileDialog, \
    QProgressDialog
from PyQt6.QtCore import QCoreApplication
from VisualInterface.main_window_ui import Ui_MainWindow
from VisualInterface.form import Form
from VisualInterface.utils import *
import sys
import json
import AgentBasedModel
import matplotlib.pyplot as plt
import os
import logging

NAMES = ["traders", "events", "brokers", "stats"]


def fetch_form_parameters(form):
    res_type = form.type_box.currentText()
    params = dict()
    for i in range(form.form_layout.count() // 2):
        param_name = form.form_layout.itemAt(i * 2).widget().text()
        type_hint = param_name[param_name.find("(") + 1:-1]
        param = param_name[:param_name.find("(")]
        value = form.form_layout.itemAt(i * 2 + 1).widget().text()
        if "Optional" in type_hint and value == "":
            params[param] = None
            continue
        if type_hint == "int" or type_hint == "List[int]" or type_hint == "Optional[int]" or type_hint == "Optional[List[int]]":
            params[param] = list(map(int, value.split(",")))
            if type_hint == "int" or type_hint == "Optional[int]" and len(params[param]) == 1:
                params[param] = params[param][0]
        elif type_hint == "float" or type_hint == "float or int":
            params[param] = list(map(float, value.split(",")))
        else:
            sys.exit(1)
    return res_type, params


def dict_to_text_browser(class_name, di):
    di2 = di.copy()
    cnt = 1
    if "count" in di:
        cnt = di["count"]
        di2.pop("count")
    res = QTextBrowser()
    di_str = "\n".join([f"{k}: {v}" for k, v in di.items()])
    res.setText(f'{cnt} {class_name}: {di_str}')
    return res


def recursive_bruteforce(cur_params, cur_param_num, param_values):
    print(cur_params)
    if cur_param_num == len(param_values):
        yield cur_params[:]
        return
    for val in param_values[cur_param_num]:
        cur_params[cur_param_num] = val
        for x in recursive_bruteforce(cur_params, cur_param_num + 1, param_values):
            yield x


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # self.traders_layout = QVBoxLayout(parent=self.trader_scroll_area)
        for class_type in "trader", "event", "broker", "stats":
            setattr(self, f"{class_type}_layout", QVBoxLayout())
            getattr(self, f"{class_type}_scroll_area").setLayout(getattr(self, f"{class_type}_layout"))
            setattr(self, f"{class_type}_groupbox", QGroupBox())
            getattr(self, f"{class_type}_groupbox").setLayout(getattr(self, f"{class_type}_layout"))
            getattr(self, f"{class_type}_scroll_area").setWidget(getattr(self, f"{class_type}_groupbox"))
            getattr(self, f"add_{class_type}_button").clicked.connect(self.show_form(class_type))
        self.save_config_button.clicked.connect(self.save_config)
        self.load_config_button.clicked.connect(self.load_config)
        self.launch_button.clicked.connect(self.run_simulator)
        self.simulation_dict = {
            "traders": [],
            "events": [],
            "brokers": [],
            "stats": [],
            "iterations": 100
        }
        self.folder_name = ""
        self.simulation_name = ""
        self.cur_sim_number = 0
        self.sync_parameters()

    def update_iterations_count(self):
        self.simulation_dict["iterations"] = self.iterations_spinbox.value()

    def show_form(self, class_category):
        def f():
            print(1)
            form = Form(class_category)
            if form.exec() == 0:
                return
            item_type, params = fetch_form_parameters(form)
            class_type = class_category[:]
            if class_category in ["trader", "broker"]:
                class_type = "agents"
            elif class_category == "stats":
                class_type = "stats"
            else:
                class_type = "events"
            for k in params.keys():
                if not can_bruteforce(class_type, item_type, k) and is_list_parameter(class_type, item_type, k) and \
                        params[k] is not None:
                    params[k] = params[k][0]
            key = class_category[:]
            if not key.endswith("s"):
                key += "s"
            self.simulation_dict[key].append({
                "type": item_type,
                "params": params
            })
            self.sync_parameters()

        return f

    def clear_layouts(self):
        for class_type in "trader", "event", "broker", "stats":
            lay: QVBoxLayout = getattr(self, f"{class_type}_layout")
            while lay.count() > 0:
                wid = lay.takeAt(0)
                if wid.widget():
                    wid.widget().deleteLater()

    def sync_parameters(self):
        self.clear_layouts()
        self.iterations_spinbox.setValue(self.simulation_dict["iterations"])
        for class_type in "traders", "events", "brokers", "stats":
            for di in self.simulation_dict[class_type]:
                name2 = class_type[:]
                if name2 != "stats":
                    name2 = name2[:-1]
                getattr(self, f"{name2}_layout").addWidget(dict_to_text_browser(di["type"], di["params"]))

    def save_config(self):
        self.update_iterations_count()
        name = QFileDialog.getSaveFileName(self, 'Save File')[0]
        if name == "":
            return
        file = open(name, "w")
        json.dump(self.simulation_dict, file)

    def load_config(self):
        name = QFileDialog.getOpenFileName(self, "Open file")[0]
        self.simulation_dict = json.load(open(name))
        self.sync_parameters()

    def run_simulator(self):
        self.update_iterations_count()
        self.folder_name = QFileDialog.getExistingDirectory(self, 'Select folder to save results')
        self.simulation_name = self.simulation_name_input.text()
        try:
            os.mkdir(os.path.join(self.folder_name, self.simulation_name))
        except:
            QMessageBox.about(self, "Error", "Couldn't create directory")
            return
        params_values = []
        simulations_count = 1
        for name in NAMES:
            if name == "stats":
                continue
            class_type = "agents"
            if name == "events":
                class_type = "events"
            for di in self.simulation_dict[name]:
                class_name = di["type"]
                for param_name, value in di["params"].items():
                    if can_bruteforce(class_type, class_name, param_name) and type(value) == list:
                        params_values.append(value)
                        simulations_count *= len(value)
        progress_dialog = QProgressDialog("Simulation in progress", "cancel", 0, simulations_count, self)
        progress_dialog.setModal(True)
        self.cur_sim_number = 0
        progress_dialog.setValue(self.cur_sim_number)
        progress_dialog.show()
        QCoreApplication.processEvents()
        for cur_params in recursive_bruteforce([0] * len(params_values), 0, params_values):
            cur_sim_settings = dict()
            p = 0
            for name in NAMES:
                cur_sim_settings[name] = []
                class_type = name[:]
                if class_type == "brokers" or class_type == "traders":
                    class_type = "agents"
                for di in self.simulation_dict[name]:
                    class_name = di["type"]
                    cur_sim_settings[name].append({
                        "type": di["type"],
                        "params": dict()
                    })
                    for param_name, value in di["params"].items():
                        if name != "stats" and can_bruteforce(class_type, class_name, param_name):
                            cur_sim_settings[name][-1]["params"][param_name] = cur_params[p]
                            p += 1
                        elif value is None:
                            cur_sim_settings[name][-1]["params"][param_name] = None
                        elif is_list_parameter(class_type, class_name, param_name):
                            cur_sim_settings[name][-1]["params"][param_name] = value[:]
                        else:
                            cur_sim_settings[name][-1]["params"][param_name] = value[0]
            try:
                self.run_one_simulation(cur_sim_settings)
            except:
                progress_dialog.deleteLater()
                QMessageBox.about(self, "Error", "Error while running the simulation")

                return
            self.cur_sim_number += 1
            progress_dialog.setValue(self.cur_sim_number)
            QCoreApplication.processEvents()

    def run_one_simulation(self, simulation_settings):
        brokers = []
        traders = []
        events = []
        cur_dir = os.path.join(self.folder_name, self.simulation_name, str(self.cur_sim_number))
        os.mkdir(cur_dir)
        AgentBasedModel.Logger = AgentBasedModel.create_logger(
            filename=os.path.join(cur_dir, f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"))

        for di in simulation_settings["brokers"]:
            cur_params = di["params"].copy()
            cur_params.pop("count")
            for _ in range(di["params"]["count"]):
                brokers.append(getattr(AgentBasedModel, di["type"])(**cur_params))

        def convert_markets(di):
            markets = []
            for i in di["markets"]:
                markets.append(brokers[i])
            di["markets"] = markets
            return di

        for di in simulation_settings["traders"]:
            di2 = convert_markets(di["params"].copy())
            di2.pop("count")
            for _ in range(di["params"]["count"]):
                traders.append(getattr(AgentBasedModel, di["type"])(**di2))
        for di in simulation_settings["events"]:
            events.append(getattr(AgentBasedModel, di["type"])(**di["params"]))
        simulator = AgentBasedModel.Simulator(brokers, traders, events)
        infos = simulator.info
        simulator.simulate(int(self.iterations_spinbox.text()))
        plot_id = 0
        for di in simulation_settings["stats"]:
            di2 = di["params"].copy()
            di2["info"] = infos[di2["info"]]
            di2["filename"] = os.path.join(cur_dir, f"{plot_id}.png")
            getattr(AgentBasedModel, di["type"])(**di2)
            plot_id += 1
